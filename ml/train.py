from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


def _fit_model_from_xy(X_train, y_train):
    """Fit a StandardScaler + RandomForest on pre-split X/y arrays."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_model_with_holdout(df, val_fraction=0.2, min_val_rows=5, min_retrain_rows=30):
    """
    Train with a unit-based holdout split.
    Returns (model, scaler, val_mae) or (None, None, None) if not enough data.
    """
    df = df.dropna()
    if len(df) < min_retrain_rows:
        print(f"  train_model_with_holdout: only {len(df)} rows, need {min_retrain_rows} — skip")
        return None, None, None

    units = df["unit"].dropna().unique()

    if len(units) >= 2:
        train_units, val_units = train_test_split(units, test_size=val_fraction, random_state=42)
        train_df = df[df["unit"].isin(train_units)].copy()
        val_df = df[df["unit"].isin(val_units)].copy()
    else:
        ordered = df.sort_values("cycle").reset_index(drop=True)
        split = max(1, min(int((1 - val_fraction) * len(ordered)), len(ordered) - 1))
        train_df = ordered.iloc[:split]
        val_df = ordered.iloc[split:]

    X_train = train_df.drop(columns=["RUL", "unit", "cycle"])
    y_train = train_df["RUL"]

    if len(val_df) < min_val_rows or len(train_df) < 2:
        model, scaler = _fit_model_from_xy(X_train, y_train)
        print("  train_model_with_holdout: trained without holdout validation")
        return model, scaler, None

    X_val = val_df.drop(columns=["RUL", "unit", "cycle"])
    y_val = val_df["RUL"]

    model, scaler = _fit_model_from_xy(X_train, y_train)
    y_pred = model.predict(scaler.transform(X_val))
    val_mae = mean_absolute_error(y_val, y_pred)
    print(f"  train_model_with_holdout | val MAE: {val_mae:.2f} | train: {len(train_df)} | val: {len(val_df)}")
    return model, scaler, float(val_mae)
