def predict(model, data):
    # convert dict → DataFrame-like structure
    import pandas as pd

    df = pd.DataFrame([data])

    # drop target + unit (VERY IMPORTANT)
    df = df.drop(columns=['RUL', 'unit'], errors='ignore')

    prediction = model.predict(df)[0]

    return prediction