import numpy as np

def naive_forecast(series, horizon):
    return np.repeat(series[-1], horizon)
