from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def train_model(df):
    X = df.drop(columns=['RUL','unit'])
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50)
    model.fit(X_train, y_train)

    print("Model trained successfully")

    return model