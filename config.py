import torch

DATA_PATH = r"data/processed/final_energy_forecasting_dataset.csv"

TIME_COL = "time"
TARGET_COL = "load"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

LOOKBACK = 720

HORIZON_24 = 24
HORIZON_7D = 168
HORIZON_30D = 720

QUANTILES = [0.1, 0.5, 0.9]
N_QUANTILES = len(QUANTILES)

BATCH_SIZE = 32
EPOCHS = 50

MODEL_CONFIG = {
    "hidden_size": 128,
    "attention_heads": 8,
    "dropout": 0.25,        # increased dropout
    "learning_rate": 0.0002  # reduced learning rate
}

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

MODEL_PATH = "results/models/hybrid_model_final_50epochs.pth"

