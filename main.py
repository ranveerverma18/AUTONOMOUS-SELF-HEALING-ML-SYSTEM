from dataset.processed.preprocess_module import load_data, add_rul
from simulation.chaos_controller import inject_noise, inject_drift
from ml.train import train_model
from ml.predict import predict
from drift.error_monitor import ErrorMonitor
from drift.adwin_detector import DriftDetector
from decision.engine import DecisionEngine
from sklearn.model_selection import train_test_split
from collections import deque
from drift.data_drift import DataDriftDetector
import joblib
import mlflow
import numpy as np
import pandas as pd


def split_by_unit(df, stream_fraction=0.2, random_state=42):
    units = df['unit'].unique()
    train_units, stream_units = train_test_split(
        units,
        test_size=stream_fraction,
        random_state=random_state,
    )

    train_df = df[df['unit'].isin(train_units)].copy()
    stream_df = df[df['unit'].isin(stream_units)].copy()

    return train_df, stream_df


def create_engine_stream(df):
    engines = {}

    for unit in df['unit'].unique():
        engine_df = df[df['unit'] == unit].sort_values("cycle")
        engines[unit] = engine_df.to_dict("records")

    return engines


def interleaved_stream(df, max_cycles=200):
    engines = create_engine_stream(df)

    for i in range(max_cycles):
        for unit in engines:
            if i < len(engines[unit]):
                yield engines[unit][i]


def is_retrain_buffer_valid(df, min_rows=30, min_units=1):
    if len(df) <= min_rows:
        return False, f"need > {min_rows} buffered samples"

    if 'unit' not in df.columns or df['unit'].nunique() < min_units:
        return False, f"need at least {min_units} distinct units"

    if 'RUL' not in df.columns:
        return False, "missing RUL column"

    feature_df = df.drop(columns=['RUL', 'unit', 'cycle'], errors='ignore')

    if feature_df.empty:
        return False, "no features available for training"

    if feature_df.isnull().any().any() or df['RUL'].isnull().any():
        return False, "missing values found in retrain buffer"

    # Ensure we do not retrain on corrupted non-numeric values.
    finite_mask = np.isfinite(feature_df.to_numpy(dtype=float)).all()
    if not finite_mask:
        return False, "non-finite feature values found"

    return True, "ok"


def evaluate_on_holdout(model, scaler, holdout_df):
    """
    Compute MAE of `model` on a fixed holdout set that was never used for
    training.  This gives a fair, apples-to-apples comparison between the
    baseline and every retrain candidate.
    """
    from ml.predict import predict as _predict
    errors = []
    for _, row in holdout_df.iterrows():
        pred = _predict(model, scaler, row.to_dict())
        errors.append(abs(row["RUL"] - pred))
    return float(np.mean(errors)) if errors else float("inf")


def run_pipeline():
    # load data
    df = load_data("dataset/raw/train_FD001.txt")
    df = add_rul(df)

    train_df, stream_df = split_by_unit(df, stream_fraction=0.2, random_state=42)
    expected_retrain_columns = train_df.columns.tolist()

    # -----------------------------------------------------------------------
    # FIX 1: Carve out a fixed holdout set from train_df BEFORE training.
    #         This set is NEVER touched during training or retraining and is
    #         the single yardstick used to compare every model candidate.
    # -----------------------------------------------------------------------
    holdout_fraction = 0.15          # ~15 % of training units held out
    holdout_units, fit_units = train_test_split(
        train_df['unit'].unique(),
        test_size=(1 - holdout_fraction),
        random_state=0,
    )
    holdout_df = train_df[train_df['unit'].isin(holdout_units)].copy()
    fit_df     = train_df[train_df['unit'].isin(fit_units)].copy()

    # train model on fit_df only
    model, scaler = train_model(fit_df, model_name="train_model")

    # -----------------------------------------------------------------------
    # FIX 2: Seed current_best_mae from the ACTUAL holdout performance of the
    #         initial model, not float("inf").  This prevents a tiny,
    #         overfit retrain candidate from stealing the baseline with MAE=0.
    # -----------------------------------------------------------------------
    current_best_mae = evaluate_on_holdout(model, scaler, holdout_df)
    print(f"Initial model holdout MAE (baseline): {current_best_mae:.2f}\n")

    # monitors
    monitor = ErrorMonitor(window_size=5)
    detector = DriftDetector()
    data_drift_detector = DataDriftDetector(window_size=30)
    engine = DecisionEngine(error_threshold=15)

    # -----------------------------------------------------------------------
    # FIX 3: Raise minimum buffer size so the retrain set can be split into a
    #         proper train/validation partition without overfitting.
    #         50 rows → ~40 train / 10 val, which is still small but far less
    #         prone to zero-variance MAE than 32 rows with a single split.
    # -----------------------------------------------------------------------
    MIN_RETRAIN_ROWS = 50
    MAX_STREAM_EVENTS = 200

    buffer = deque(maxlen=200)       # increased cap so buffer grows healthily
    retrain_cooldown_cycles = 10
    last_retrain_cycle_idx = -retrain_cooldown_cycles
    retrain_error_threshold = engine.error_threshold
    skipped_retrains = 0
    logs = []
    log_buffer = deque(maxlen=200)

    print("\n--- STREAM + DRIFT DETECTION ---\n")

    for i, data in enumerate(interleaved_stream(stream_df, max_cycles=200)):
        if i >= MAX_STREAM_EVENTS:
            break

        # chaos
        data = inject_noise(data)
        if i > 10 and i<80:  # inject drift after some initial stable cycles
            data = inject_drift(data, shift=10)

        # Keep latest stream samples for possible drift-aware retraining.
        buffer.append(data.copy())

        # prediction
        actual = data['RUL']
        pred = predict(model, scaler, data)
        error = abs(actual - pred)

        # monitoring
        rolling_avg = monitor.update(error)
        trend = monitor.is_increasing()

        # ADWIN
        drift_input = rolling_avg if rolling_avg is not None else error
        drift = detector.update(drift_input)
        data_drift_result = data_drift_detector.update_with_details(data)
        data_drift = data_drift_result["drift_detected"]
        drift_score = data_drift_result["drift_score"]
        drifted_features = data_drift_result["drifted_features"]
        combined_drift = drift or data_drift

        action = engine.decide(combined_drift, rolling_avg, trend)
        logs.append(
            {
                "event": "cycle",
                "index": i,
                "unit": int(data["unit"]),
                "cycle": int(data["cycle"]),
                "error": float(error),
                "rolling_avg": None if rolling_avg is None else float(rolling_avg),
                "trend": bool(trend),
                "drift": bool(drift),
                "data_drift": bool(data_drift),
                "combined_drift": bool(combined_drift),
                "data_drift_score": float(drift_score),
                "drifted_features": list(drifted_features),
                "action": action,
            }
        )
        log_line = (
            f"Cycle: {int(data['cycle'])} | "
            f"Error: {error:.2f} | "
            f"Rolling Avg: {rolling_avg if rolling_avg is not None else '...'} | "
            f"Trend: {trend} | "
            f"ADWIN Drift: {drift} | "
            f"Data Drift: {data_drift} | "
            f"Drift Score: {drift_score:.3f} | "
            f"Drifted Features: {drifted_features} | "
            f"Action: {action}"
        )

        log_buffer.append(log_line)
        print(log_line)

        if i % 20 == 0 and i != 0:
            print(f"\n📊 SUMMARY at cycle {i}")
            print(f"Buffer size: {len(buffer)}")
            print(f"Current best MAE: {current_best_mae:.2f}\n")

        cooldown_elapsed = (i - last_retrain_cycle_idx) >= retrain_cooldown_cycles

        if action == "RETRAIN":
            print("\n🚨🚨 RETRAIN EVENT 🚨🚨\n")
            print("\n🔍 RETRAIN DEBUG INFO")
            print(f"Unit: {int(data['unit'])}")
            print(f"Cycle: {int(data['cycle'])}")
            print(f"Rolling Avg: {rolling_avg}")
            print(f"Threshold: {engine.error_threshold}")
            print(f"Cooldown OK: {cooldown_elapsed}")
            print(f"Drift Signal: {drift}")
            print(f"Buffer Size: {len(buffer)}")
            print(f"Current Best Holdout MAE: {current_best_mae:.2f}")

        if (
            action == "RETRAIN"
            and cooldown_elapsed
            and rolling_avg is not None
            and (rolling_avg > retrain_error_threshold or drift or data_drift)
        ):
            print("\nDrift detected -> Retraining candidate accepted\n")
            logs.append(
                {
                    "event": "retrain_candidate",
                    "index": i,
                    "cycle": int(data["cycle"]),
                    "buffer_size": len(buffer),
                }
            )

            new_df = pd.DataFrame(buffer)

            # Keep retrain preprocessing compatible with original training schema.
            if 'RUL' not in new_df.columns:
                new_df = add_rul(new_df)

            missing_cols = [c for c in expected_retrain_columns if c not in new_df.columns]
            if missing_cols:
                reason = f"schema mismatch, missing columns: {missing_cols}"
                print(f"Skipping retrain: {reason}\n")
                logs.append(
                    {
                        "event": "retrain_skipped",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "reason": reason,
                    }
                )
                continue

            # Drop unexpected fields from streaming payload and keep training schema.
            new_df = new_df.loc[:, expected_retrain_columns]

            is_valid, reason = is_retrain_buffer_valid(new_df, min_rows=MIN_RETRAIN_ROWS, min_units=2)
            if not is_valid:
                print(f"Skipping retrain: {reason}\n")
                logs.append(
                    {
                        "event": "retrain_skipped",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "reason": reason,
                    }
                )
                continue

            retrain_data_size = len(new_df)
            new_model, new_scaler, new_mae_internal = None, None, None
            print(f"Retrain decision -> data size: {retrain_data_size}")

            if retrain_data_size < MIN_RETRAIN_ROWS:
                print("❌ Retrain skipped: insufficient data")
                skipped_retrains += 1
                logs.append(
                    {
                        "event": "retrain_skipped",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "reason": f"insufficient retrain data (< {MIN_RETRAIN_ROWS})",
                        "buffer_size": retrain_data_size,
                    }
                )
                continue
            else:
                print("✅ Retrain executed")
                new_model, new_scaler, new_mae_internal = train_model(
                    new_df,
                    return_mae=True,
                    model_name="retrain_model",
                    cycle=int(data["cycle"]),
                    trigger="drift",
                )

            last_retrain_cycle_idx = i

            # -------------------------------------------------------------------
            # FIX 4: Evaluate the candidate on the SAME fixed holdout set used
            #         to measure the baseline.  Internal train/val MAE (which
            #         comes from train_model's own split of new_df) is NOT used
            #         for the accept/reject decision — only holdout MAE is.
            # -------------------------------------------------------------------
            if new_model is not None and new_scaler is not None:
                candidate_holdout_mae = evaluate_on_holdout(new_model, new_scaler, holdout_df)
                print(f"Candidate internal MAE : {new_mae_internal:.2f}" if new_mae_internal is not None else "")
                print(f"Candidate holdout  MAE : {candidate_holdout_mae:.2f}")
                print(f"Current  best MAE      : {current_best_mae:.2f}")

                if candidate_holdout_mae < current_best_mae + 5:
                    print("✅ New model is BETTER on holdout -> Accepting")

                    model = new_model
                    scaler = new_scaler
                    current_best_mae = candidate_holdout_mae
                    joblib.dump(model, "best_model.pkl")

                    logs.append(
                        {
                            "event": "model_accepted",
                            "index": i,
                            "cycle": int(data["cycle"]),
                            "holdout_mae": candidate_holdout_mae,
                            "internal_mae": new_mae_internal,
                            "buffer_size": len(buffer),
                        }
                    )

                    try:
                        with mlflow.start_run(run_name="model_selection"):
                            mlflow.set_tag("model_status", "accepted")
                            mlflow.log_param("cycle", int(data["cycle"]))
                            mlflow.log_metric("candidate_holdout_mae", candidate_holdout_mae)
                            if new_mae_internal is not None:
                                mlflow.log_metric("candidate_internal_mae", float(new_mae_internal))
                    except Exception as err:
                        print(f"MLflow model_status logging warning: {err}")

                    # reset drift detectors after model swap
                    detector = DriftDetector()
                    data_drift_detector = DataDriftDetector(window_size=30)
                    monitor = ErrorMonitor(window_size=5)

                    print("Model retrained successfully\n")
                else:
                    print("❌ New model is WORSE on holdout -> Rejecting")

                    logs.append(
                        {
                            "event": "model_rejected",
                            "index": i,
                            "cycle": int(data["cycle"]),
                            "holdout_mae": candidate_holdout_mae,
                            "internal_mae": new_mae_internal,
                            "buffer_size": len(buffer),
                        }
                    )

                    try:
                        with mlflow.start_run(run_name="model_selection"):
                            mlflow.set_tag("model_status", "rejected")
                            mlflow.log_param("cycle", int(data["cycle"]))
                            mlflow.log_metric("candidate_holdout_mae", candidate_holdout_mae)
                            if new_mae_internal is not None:
                                mlflow.log_metric("candidate_internal_mae", float(new_mae_internal))
                    except Exception as err:
                        print(f"MLflow model_status logging warning: {err}")
            else:
                print("Reject retrained model\n")
                logs.append(
                    {
                        "event": "retrain_rejected",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "buffer_size": len(buffer),
                        "reason": "train_model returned None",
                    }
                )

    retrain_count = sum(1 for entry in logs if entry["event"] == "model_accepted")
    rejected_count = sum(1 for entry in logs if entry["event"] == "model_rejected")
    skipped_count = sum(1 for entry in logs if entry["event"] == "retrain_skipped")
    print(
        f"Run complete | Retrains: {retrain_count} | "
        f"Rejected Retrains: {rejected_count} | "
        f"Skipped Retrains: {skipped_count} | Skipped Counter: {skipped_retrains}"
    )


if __name__ == "__main__":
    run_pipeline()