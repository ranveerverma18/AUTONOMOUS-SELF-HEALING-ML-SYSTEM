import time

def stream_data(df):
    for _, row in df.iterrows():
        yield row.to_dict()
        time.sleep(0.05)