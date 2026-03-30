from dataset.processed.preprocess_module import load_data, add_rul
from simulation.data_generator import stream_data
from simulation.chaos_controller import inject_noise, inject_drift
from ml.train import train_model
from ml.predict import predict
from drift.error_monitor import ErrorMonitor
from drift.adwin_detector import DriftDetector
from decision.engine import DecisionEngine
from sklearn.model_selection import train_test_split


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


def run_pipeline():
    # load data
    df = load_data("dataset/raw/train_FD001.txt")
    df = add_rul(df)

    train_df, stream_df = split_by_unit(df, stream_fraction=0.2, random_state=42)

    # train model
    model, scaler = train_model(train_df)

    # monitors
    monitor = ErrorMonitor(window_size=5)
    detector = DriftDetector()
    engine = DecisionEngine(error_threshold=40)

    print("\n--- STREAM + DRIFT DETECTION ---\n")

    for i, data in enumerate(stream_data(stream_df.head(200))):

        # chaos
        data = inject_noise(data)
        if i > 10:
            data = inject_drift(data, shift=10)

        # prediction
        actual = data['RUL']
        pred = predict(model, scaler, data)
        error = abs(actual - pred)

        # monitoring
        rolling_avg = monitor.update(error)
        trend = monitor.is_increasing()

        # ADWIN
        drift = detector.update(error)

        action = engine.decide(drift, rolling_avg, trend)
        # print
        print(
            f"Cycle: {int(data['cycle'])} | "
            f"Error: {error:.2f} | "
            f"Rolling Avg: {rolling_avg if rolling_avg is not None else '...'} | "
            f"Trend: {trend} | "
            f"ADWIN Drift: {drift} |"
            f"Action: {action}"
        )


if __name__ == "__main__":
    run_pipeline()