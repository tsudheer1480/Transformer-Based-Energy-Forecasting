import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, FEATURE_CONFIG, TARGET_COL
from evaluate_multiscale import evaluate_multiscale
from experiments.rolling_validation import rolling_window_evaluation
from experiments.rolling_validation import rolling_window_evaluation
from forecast_interface import interactive_forecast
import warnings

from train import train_multiscale
warnings.filterwarnings("ignore")

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

feature_cols = (
    FEATURE_CONFIG["known_features"] +
    FEATURE_CONFIG["unknown_features"]
)

print("Number of features:", len(feature_cols))

# ==========================
# NORMALIZATION
# ==========================

feature_scaler = StandardScaler()
target_scaler = StandardScaler()

feature_cols_no_target = [col for col in feature_cols if col != TARGET_COL]

df[feature_cols_no_target] = feature_scaler.fit_transform(df[feature_cols_no_target])
df[TARGET_COL] = target_scaler.fit_transform(df[[TARGET_COL]])

# ==========================
# TRAIN / TEST SPLIT
# ==========================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# ==========================
# TRAIN MODEL 
# ========================== 

# model = train_multiscale(train_df, feature_cols)

# ==========================
# EVALUATE MODEL
# ==========================

evaluate_multiscale(test_df, feature_cols)

print("\nModel evaluation completed successfully.")

# Rolling Window
rolling_window_evaluation(df, feature_cols)


# ==========================
# INTERACTIVE FORECAST SYSTEM
# ==========================

interactive_forecast(test_df, feature_cols, target_scaler)
