from dataset.processed.preprocess_module import load_data, add_rul
from simulation.data_generator import stream_data

# load data
df = load_data("dataset/raw/train_FD001.txt")
df = add_rul(df)

# simulate stream
print("\n--- STREAM START ---\n")

for data in stream_data(df.head(10)):
    print(data)