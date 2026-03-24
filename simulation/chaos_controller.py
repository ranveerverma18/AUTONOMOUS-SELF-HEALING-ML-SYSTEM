import numpy as np

def inject_noise(data, noise_level=0.1):
    for key in data:
        if "sensor" in key:
            data[key] += np.random.normal(0, noise_level)
    return data


def inject_drift(data, shift=1.0):
    for key in data:
        if "sensor" in key:
            data[key] += shift
    return data