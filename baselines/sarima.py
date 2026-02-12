from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(train_series, horizon):
    model = SARIMAX(
        train_series,
        order=(1,1,1),
        seasonal_order=(1,1,1,24),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results.forecast(steps=horizon)
