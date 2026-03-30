import pandas as pd


def load_data(path):
    cols = ['unit', 'cycle'] + \
           [f'op_setting_{i}' for i in range(1, 4)] + \
           [f'sensor_{i}' for i in range(1, 22)]

    # CMAPSS files can contain variable spacing and a trailing blank column.
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1)

    if df.shape[1] != len(cols):
        raise ValueError(
            f"Unexpected column count in {path}: got {df.shape[1]}, expected {len(cols)}"
        )

    df.columns = cols

    return df


def add_rul(df):   #rul-remaining useful life
    max_cycle = df.groupby('unit')['cycle'].max().reset_index()
    max_cycle.columns = ['unit', 'max_cycle']

    df = df.merge(max_cycle, on='unit')
    df['RUL'] = df['max_cycle'] - df['cycle']

    df['RUL']= df['RUL'].clip(upper=125)  # cap RUL at 125

    return df.drop(columns=['max_cycle'])