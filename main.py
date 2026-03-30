from dataset.processed.preprocess_module import load_data, add_rul
from simulation.data_generator import stream_data
from simulation.chaos_controller import inject_noise, inject_drift
from ml.train import train_model
from ml.predict import predict
from drift.error_monitor import ErrorMonitor
from drift.adwin_detector import DriftDetector

# load data
df = load_data("dataset/raw/train_FD001.txt")
df = add_rul(df)

# train model
model,scaler = train_model(df)

# monitors
monitor = ErrorMonitor(window_size=5)
detector = DriftDetector()

print("\n--- STREAM + DRIFT DETECTION ---\n")

for i, data in enumerate(stream_data(df.head(200))):

    # chaos
    data = inject_noise(data)
    if i > 10:
        data = inject_drift(data,shift=10)

    # prediction
    actual = data['RUL']
    pred = predict(model,scaler, data)
    error = abs(actual - pred)

    # monitoring
    rolling_avg = monitor.update(error)
    trend = monitor.is_increasing()

    # ADWIN
    drift = detector.update(rolling_avg if rolling_avg else error)

    # print
    print(
        f"Cycle: {int(data['cycle'])} | "
        f"Error: {error:.2f} | "
        f"Rolling Avg: {rolling_avg if rolling_avg else '...'} | "
        f"Trend: {trend} | "
        f"ADWIN Drift: {drift}"
    )