def predict(model, data):
    # remove target if present
    if 'RUL' in data:
        data = data.drop('RUL')

    prediction = model.predict([data])[0]
    return prediction