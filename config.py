import torch

# ==============================
# DATA CONFIG
# ==============================

DATA_PATH = r"data/processed/final_energy_forecasting_dataset.csv"

TIME_COL = "time"
TARGET_COL = "load"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ==============================
# 24H MODEL CONFIG
# ==============================

LOOKBACK = 720      # Using full 720 history
HORIZON = 24

BATCH_SIZE = 32
EPOCHS = 20

MODEL_CONFIG = {
    "encoder_length": LOOKBACK,
    "prediction_length": HORIZON,
    "hidden_size": 96,
    "attention_heads": 4,
    "dropout": 0.2,
    "learning_rate": 0.0005,
}

# ==============================
# FEATURE CONFIG
# ==============================

FEATURE_CONFIG = {
    "known_features": [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday"
    ],
    "unknown_features": [
        "load",
        "solar",
        "wind",
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation",
        "load_lag_1",
        "load_lag_24",
        "load_lag_168",
        "rolling_mean_24",
        "rolling_std_24"
    ]
}
