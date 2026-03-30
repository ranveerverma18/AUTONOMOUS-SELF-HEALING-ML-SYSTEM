def predict(model,scaler, data):
    # convert dict → DataFrame-like structure
    import pandas as pd

    df = pd.DataFrame([data])

    # drop target + unit (VERY IMPORTANT)
    X = df.drop(columns=['RUL', 'unit','cycle'], errors='ignore')
    X_scale = scaler.transform(X)
    prediction = model.predict(X_scale)[0]

    return prediction