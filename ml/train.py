from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def train_model(df):

    # features
    X = df.drop(columns=['RUL', 'unit', 'cycle'])
    y = df['RUL']
    groups = df['unit']

    def fit_on_all_data(reason):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)

        print(f"Model trained on full retrain buffer ({reason})")
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

    # scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)

    # evaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model trained successfully | MAE: {mae:.2f}")

    return model, scaler