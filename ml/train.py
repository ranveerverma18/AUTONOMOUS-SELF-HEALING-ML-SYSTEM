from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import pandas as pd

def train_model(df):

    # features
    X = df.drop(columns=['RUL', 'unit', 'cycle'])
    y = df['RUL']

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train_scaled, y_train)

    # evaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model trained successfully | MAE: {mae:.2f}")

    return model, scaler