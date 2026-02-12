import xgboost as xgb
import numpy as np

def create_features(series, lags=[1,24,168]):
    X, y = [], []
    for i in range(max(lags), len(series)):
        X.append([series[i-l] for l in lags])
        y.append(series[i])
    return np.array(X), np.array(y)

def train_xgboost(series):
    X, y = create_features(series)
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05
    )
    model.fit(X, y)
    return model

def forecast_xgboost(model, history, horizon):
    preds = []
    hist = list(history)
    for _ in range(horizon):
        x = np.array([[hist[-1], hist[-24], hist[-168]]])
        y_hat = model.predict(x)[0]
        preds.append(y_hat)
        hist.append(y_hat)
    return np.array(preds)
