from dataset.processed.preprocess_module import load_data, add_rul
from simulation.data_generator import stream_data
from simulation.chaos_controller import inject_noise, inject_drift
from ml.train import train_model
from ml.predict import predict
from drift.error_monitor import ErrorMonitor

# load + preprocess
df = load_data("dataset/raw/train_FD001.txt")
df = add_rul(df)

# train model
model = train_model(df)

# initialize monitor
monitor = ErrorMonitor(window_size=5)

print("\n--- STREAM + ERROR MONITORING ---\n")

for i, data in enumerate(stream_data(df.head(30))):

    # chaos
    data = inject_noise(data)
    if i > 10:
        data = inject_drift(data)

    # prediction
    actual = data['RUL']
    pred = predict(model, data)
    error = abs(actual - pred)

    # update monitor
    rolling_avg = monitor.update(error)
    trend = monitor.is_increasing()

    # print
    print(
        f"Cycle: {int(data['cycle'])} | "
        f"Error: {error:.2f} | "
        f"Rolling Avg: {rolling_avg if rolling_avg else '...'} | "
        f"Trend Increasing: {trend}"
    )