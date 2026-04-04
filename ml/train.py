from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Default")


def _fit_model_from_xy(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def _log_mlflow_run(model, params, metrics, run_name):
    def _write_run():
        print("🔥 MLflow run starting")
        active_run = mlflow.active_run()
        with mlflow.start_run(run_name=run_name, nested=active_run is not None):
            logging_warnings = []

            try:
                for key, value in params.items():
                    mlflow.log_param(key, value)
            except Exception as err:
                logging_warnings.append(f"param_logging_failed: {err}")

            print("🔥 Logging metrics...")
            try:
                for key, value in metrics.items():
                    if value is not None:
                        mlflow.log_metric(key, float(value))
            except Exception as err:
                logging_warnings.append(f"metric_logging_failed: {err}")

            try:
                mlflow.sklearn.log_model(model, "model")
            except Exception as err:
                logging_warnings.append(f"model_logging_failed: {err}")

            if logging_warnings:
                warning_text = " | ".join(logging_warnings)[:500]
                try:
                    mlflow.set_tag("logging_warning", warning_text)
                except Exception:
                    pass
                print(f"MLflow logging warnings: {warning_text}")

    try:
        _write_run()
        return
    except Exception as err:
        # Some environments expose invalid tracking URIs (e.g., encoded spaces on Windows).
        # Fallback to a local workspace path so model training never crashes on logging.
        local_store = Path(__file__).resolve().parents[1] / "mlruns"
        local_store.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(local_store.as_uri())
        try:
            _write_run()
            print(f"MLflow fallback enabled at {local_store}")
            return
        except Exception as fallback_err:
            print(f"MLflow logging skipped: {err} | fallback failed: {fallback_err}")


def train_model(df, return_mae=False):
    df = df.dropna()
    print("Training rows:", len(df))
    print("Columns:", df.columns)

    # features
    X = df.drop(columns=['RUL', 'unit', 'cycle'])
    y = df['RUL']
    groups = df['unit']

    def fit_on_all_data(reason):
        model, scaler = _fit_model_from_xy(X, y)
        _log_mlflow_run(
            model=model,
            params={
                "model": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": 12,
                "training_mode": "full_buffer_fallback",
                "fallback_reason": reason,
                "rows": len(X),
            },
            metrics={},
            run_name="train_model_fallback",
        )

        print(f"Model trained on full retrain buffer ({reason})")
        if return_mae:
            return model, scaler, None
        return model, scaler

    # Split by engine unit to avoid leakage across cycles from the same unit.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    if len(X) < 10:
        return fit_on_all_data("too few samples")

    if groups.nunique() < 2:
        return fit_on_all_data("only one unit group")

    try:
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    except ValueError:
        return fit_on_all_data("group split failed")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model, scaler = _fit_model_from_xy(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    # evaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    _log_mlflow_run(
        model=model,
        params={
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 12,
            "splitter": "GroupShuffleSplit",
            "test_size": 0.2,
            "rows": len(X),
        },
        metrics={"mae": mae},
        run_name="train_model",
    )

    print(f"Model trained successfully | MAE: {mae:.2f}")

    if return_mae:
        return model, scaler, float(mae)
    return model, scaler


def train_model_with_holdout(
    df,
    val_fraction=0.2,
    min_val_rows=5,
    min_retrain_rows=30,
):
    df = df.dropna()
    print("Training rows:", len(df))
    print("Columns:", df.columns)

    if len(df) < min_retrain_rows:
        print("Too little data -> skipping retrain")
        return None, None, None

    units = df['unit'].dropna().unique()

    # Prefer unit-based holdout to prevent same-unit leakage.
    if len(units) >= 2:
        train_units, val_units = train_test_split(
            units,
            test_size=val_fraction,
            random_state=42,
        )
        train_df = df[df['unit'].isin(train_units)].copy()
        val_df = df[df['unit'].isin(val_units)].copy()
    else:
        # Fallback when only one unit exists in retrain buffer.
        ordered_df = df.sort_values(by='cycle').reset_index(drop=True)
        split_idx = int((1 - val_fraction) * len(ordered_df))
        split_idx = max(1, min(split_idx, len(ordered_df) - 1))
        train_df = ordered_df.iloc[:split_idx]
        val_df = ordered_df.iloc[split_idx:]

    X_train = train_df.drop(columns=['RUL', 'unit', 'cycle'])
    y_train = train_df['RUL']

    if len(val_df) < min_val_rows or len(train_df) < 2:
        model, scaler = _fit_model_from_xy(X_train, y_train)
        _log_mlflow_run(
            model=model,
            params={
                "model": "RandomForestRegressor",
                "n_estimators": 200,
                "max_depth": 12,
                "training_mode": "holdout_fallback_no_eval",
                "val_fraction": val_fraction,
                "min_val_rows": min_val_rows,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
            },
            metrics={},
            run_name="train_model_with_holdout_fallback",
        )
        print("Retrain completed without holdout evaluation (insufficient validation rows)")
        return model, scaler, None

    X_val = val_df.drop(columns=['RUL', 'unit', 'cycle'])
    y_val = val_df['RUL']

    model, scaler = _fit_model_from_xy(X_train, y_train)
    y_pred = model.predict(scaler.transform(X_val))
    val_mae = mean_absolute_error(y_val, y_pred)

    _log_mlflow_run(
        model=model,
        params={
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 12,
            "training_mode": "holdout",
            "val_fraction": val_fraction,
            "min_val_rows": min_val_rows,
            "train_rows": len(train_df),
            "val_rows": len(val_df),
        },
        metrics={"validation_mae": val_mae},
        run_name="train_model_with_holdout",
    )

    print(f"Retrain validation MAE: {val_mae:.2f}")
    return model, scaler, val_mae