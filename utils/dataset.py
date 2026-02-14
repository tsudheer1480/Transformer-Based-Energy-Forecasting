import numpy as np
from pytorch_forecasting import TimeSeriesDataSet


# ==========================================
# Baseline Sliding Window (Keep This)
# ==========================================
def create_sliding_windows(series, input_len, horizon):
    X, y = [], []
    for i in range(len(series) - input_len - horizon):
        X.append(series[i:i+input_len])
        y.append(series[i+input_len:i+input_len+horizon])
    return np.array(X), np.array(y)


# ==========================================
# TFT Dataset Preparation (NEW)
# ==========================================
def prepare_tft_datasets(df, config):

    max_encoder_length = config["encoder_length"]
    max_prediction_length = config["prediction_length"]

    training_cutoff = df["time_idx"].max() - max_prediction_length * 7

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="load",
        group_ids=["series_id"],

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        time_varying_known_reals=config["known_features"],
        time_varying_unknown_reals=config["unknown_features"],
    )

    validation = TimeSeriesDataSet.from_dataset(
        training,
        df,
        predict=True,
        stop_randomization=True
    )

    return training, validation
