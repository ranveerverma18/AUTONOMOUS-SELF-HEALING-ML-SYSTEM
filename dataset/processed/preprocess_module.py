import pandas as pd

def load_data(path):
    cols = ['unit', 'cycle'] + \
           [f'op_setting_{i}' for i in range(1,4)] + \
           [f'sensor_{i}' for i in range(1,22)]

    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis=1)
    df.columns = cols

    return df


def add_rul(df):   #rul-remaining useful life
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']

    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']

    df['RUL']= df['RUL'].clip(upper=125)  # cap RUL at 125

    return df.drop(columns=['max_cycle'])