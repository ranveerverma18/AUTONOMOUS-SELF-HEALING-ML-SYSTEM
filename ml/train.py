from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def _fit_model_from_xy(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_model(df):

    # features
    X = df.drop(columns=['RUL', 'unit', 'cycle'])
    y = df['RUL']
    groups = df['unit']

    def fit_on_all_data(reason):
        model, scaler = _fit_model_from_xy(X, y)

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

    model, scaler = _fit_model_from_xy(X_train, y_train)
    X_test_scaled = scaler.transform(X_test)

    # evaluation
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Model trained successfully | MAE: {mae:.2f}")

    return model, scaler


def train_model_with_holdout(df, val_fraction=0.2, min_val_rows=5):
    units = df['unit'].dropna().unique()

    # Prefer unit-based holdout to prevent same-unit leakage.
    if len(units) >= 2:
        train_units, val_units = train_test_split(
            units,
            test_size=val_fraction,
            random_state=42,
        )
        train_df = df[df['unit'].isin(train_units)].copy()
        val_df = df[df['unit'].isin(val_units)].copy()
    else:
        # Fallback when only one unit exists in retrain buffer.
        ordered_df = df.sort_values(by='cycle').reset_index(drop=True)
        split_idx = int((1 - val_fraction) * len(ordered_df))
        split_idx = max(1, min(split_idx, len(ordered_df) - 1))
        train_df = ordered_df.iloc[:split_idx]
        val_df = ordered_df.iloc[split_idx:]

    X_train = train_df.drop(columns=['RUL', 'unit', 'cycle'])
    y_train = train_df['RUL']

    if len(val_df) < min_val_rows or len(train_df) < 2:
        model, scaler = _fit_model_from_xy(X_train, y_train)
        print("Retrain completed without holdout evaluation (insufficient validation rows)")
        return model, scaler, None

    X_val = val_df.drop(columns=['RUL', 'unit', 'cycle'])
    y_val = val_df['RUL']

    model, scaler = _fit_model_from_xy(X_train, y_train)
    y_pred = model.predict(scaler.transform(X_val))
    val_mae = mean_absolute_error(y_val, y_pred)

    print(f"Retrain validation MAE: {val_mae:.2f}")
    return model, scaler, val_mae