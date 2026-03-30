def predict(model, scaler, data):
    # convert dict → DataFrame-like structure
    import pandas as pd

    df = pd.DataFrame([data])

    # drop target + unit (VERY IMPORTANT)
    X = df.drop(columns=['RUL', 'unit','cycle'], errors='ignore')

    # Align incoming stream data with training-time feature order.
    expected_features = getattr(scaler, "feature_names_in_", None)
    if expected_features is not None:
        missing = [c for c in expected_features if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required features for prediction: {missing}")
        X = X.loc[:, expected_features]

    X_scale = scaler.transform(X)
    prediction = model.predict(X_scale)[0]

    return prediction