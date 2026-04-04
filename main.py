from dataset.processed.preprocess_module import load_data, add_rul
from simulation.data_generator import stream_data
from simulation.chaos_controller import inject_noise, inject_drift
from ml.train import train_model, train_model_with_holdout
from ml.predict import predict
from drift.error_monitor import ErrorMonitor
from drift.adwin_detector import DriftDetector
from decision.engine import DecisionEngine
from sklearn.model_selection import train_test_split
from collections import deque
from drift.data_drift import DataDriftDetector
import mlflow
import numpy as np


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


def run_pipeline():
    # load data
    df = load_data("dataset/raw/train_FD001.txt")
    df = add_rul(df)

    train_df, stream_df = split_by_unit(df, stream_fraction=0.2, random_state=42)
    expected_retrain_columns = train_df.columns.tolist()

    # train model
    model, scaler = train_model(train_df)

    # monitors
    monitor = ErrorMonitor(window_size=5)
    detector = DriftDetector()
    data_drift_detector = DataDriftDetector(window_size=30)
    engine = DecisionEngine(error_threshold=25)
    buffer = deque(maxlen=100)  # last 100 samples
    retrain_cooldown_cycles = 15
    last_retrain_cycle_idx = -retrain_cooldown_cycles
    retrain_error_threshold = engine.error_threshold
    skipped_retrains = 0
    logs = []

    print("\n--- STREAM + DRIFT DETECTION ---\n")

    for i, data in enumerate(stream_data(stream_df.head(200))):

        # chaos
        data = inject_noise(data)
        if i > 10:
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
        # print
        print(
            f"Cycle: {int(data['cycle'])} | "
            f"Error: {error:.2f} | "
            f"Rolling Avg: {rolling_avg if rolling_avg is not None else '...'} | "
            f"Trend: {trend} | "
            f"ADWIN Drift: {drift} |"
            f"Data Drift: {data_drift} | "
            f"Drift Score: {drift_score:.3f} | "
            f"Drifted Features: {drifted_features} | "
            f"Action: {action}"
        )

        cooldown_elapsed = (i - last_retrain_cycle_idx) >= retrain_cooldown_cycles

        if action == "RETRAIN":
            print("\n🔍 RETRAIN DEBUG INFO")
            print(f"Cycle: {int(data['cycle'])}")
            print(f"Rolling Avg: {rolling_avg}")
            print(f"Threshold: {engine.error_threshold}")
            print(f"Cooldown OK: {(i - last_retrain_cycle_idx) >= retrain_cooldown_cycles}")
            print(f"Drift Signal: {drift}")
            print(f"Buffer Size: {len(buffer)}")

        if (
            action == "RETRAIN"
            and cooldown_elapsed
            and rolling_avg is not None
            and rolling_avg > retrain_error_threshold
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

            import pandas as pd
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

            is_valid, reason = is_retrain_buffer_valid(new_df, min_rows=30, min_units=1)
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

            min_retrain_rows = 30
            retrain_data_size = len(new_df)
            new_model, new_scaler, new_mae = None, None, None
            print(f"Retrain decision -> data size: {retrain_data_size}")

            try:
                with mlflow.start_run(run_name="retrain_attempt"):
                    mlflow.log_param("trigger", "drift")
                    mlflow.log_metric("data_size", retrain_data_size)

                    if retrain_data_size < min_retrain_rows:
                        mlflow.log_param("status", "skipped")
                        print("❌ Retrain skipped: insufficient data")
                        print("Reason: insufficient data")
                    else:
                        mlflow.log_param("status", "executed")
                        print("✅ Retrain executed")
                        # train + holdout + model logging happens in train_model_with_holdout.
                        new_model, new_scaler, new_mae = train_model_with_holdout(
                            new_df,
                            min_retrain_rows=min_retrain_rows,
                        )
            except Exception as err:
                print(f"MLflow retrain_attempt logging warning: {err}")
                if retrain_data_size >= min_retrain_rows:
                    new_model, new_scaler, new_mae = train_model_with_holdout(
                        new_df,
                        min_retrain_rows=min_retrain_rows,
                    )

            if retrain_data_size < min_retrain_rows:
                skipped_retrains += 1
                logs.append(
                    {
                        "event": "retrain_skipped",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "reason": f"insufficient retrain data (< {min_retrain_rows})",
                        "buffer_size": retrain_data_size,
                    }
                )
                continue

            last_retrain_cycle_idx = i
            if new_mae is not None:
                print(f"Validation MAE after retrain: {new_mae:.2f}")

            if (
                new_model is not None
                and new_scaler is not None
            ):
                print("Accept retrained model")
                model = new_model
                scaler = new_scaler
                logs.append(
                    {
                        "event": "retrained",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "buffer_size": len(buffer),
                        "validation_mae": new_mae,
                    }
                )

                # reset systems after model swap.
                detector = DriftDetector()
                data_drift_detector = DataDriftDetector(window_size=30)
                monitor = ErrorMonitor(window_size=5)

                print("Model retrained successfully\n")
            else:
                print("Reject retrained model\n")
                logs.append(
                    {
                        "event": "retrain_rejected",
                        "index": i,
                        "cycle": int(data["cycle"]),
                        "buffer_size": len(buffer),
                        "candidate_mae": new_mae,
                        "reason": "insufficient retrain data or invalid candidate",
                    }
                )

    retrain_count = sum(1 for entry in logs if entry["event"] == "retrained")
    skipped_count = sum(1 for entry in logs if entry["event"] == "retrain_skipped")
    print(
        f"Run complete | Retrains: {retrain_count} | "
        f"Skipped Retrains: {skipped_count} | Skipped Counter: {skipped_retrains}"
    )


if __name__ == "__main__":
    run_pipeline()