from utils.metrics import mae, rmse

def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred)
    }
