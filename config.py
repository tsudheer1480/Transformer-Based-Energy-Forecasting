import torch

# ==============================
# DATA CONFIG
# ==============================

DATA_PATH = r"data/processed/final_energy_forecasting_dataset.csv"

TIME_COL = "time"
TARGET_COL = "load"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ==============================
# MULTI-TASK CONFIG
# ==============================

LOOKBACK = 720

HORIZON_24 = 24
HORIZON_7D = 168

BATCH_SIZE = 32
EPOCHS = 30

MODEL_CONFIG = {
    "encoder_length": LOOKBACK,
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
