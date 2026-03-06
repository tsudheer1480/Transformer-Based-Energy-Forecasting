import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import warnings

import joblib
from config import DATA_PATH, FEATURE_CONFIG, TARGET_COL
from training.train import train_multiscale
from evaluation.evaluate_multiscale import evaluate_multiscale
from experiments.rolling_validation import rolling_window_evaluation
from interface.forecast_interface import generate_forecast_outputs

warnings.filterwarnings("ignore")

print("\n==============================")
print(" ENERGY FORECASTING SYSTEM ")
print("==============================\n")

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(r"D:\Course\python\energy forecasting\preprocessing\evaluation_test_30d_big.csv")

df["time"] = pd.to_datetime(df["time"])
df = df.sort_values("time").reset_index(drop=True)

print("Dataset shape:", df.shape)

feature_cols = (
    FEATURE_CONFIG["known_features"] +
    FEATURE_CONFIG["unknown_features"]
)

print("Number of features:", len(feature_cols))

# ==========================================
# ADD CYCLICAL FEATURES
# ==========================================

if "hour" in df.columns:
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

if "day_of_week" in df.columns:
    df["sin_week"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["cos_week"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# ==========================================
# TRAIN / TEST SPLIT (Time-Based)
# ==========================================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# ==========================================
# NORMALIZATION (Fit ONLY on training data)
# ==========================================

feature_scaler = RobustScaler()
target_scaler = RobustScaler()

feature_cols_no_target = [col for col in feature_cols if col != TARGET_COL]

# Fit on train
train_df[feature_cols_no_target] = feature_scaler.fit_transform(
    train_df[feature_cols_no_target]
)

test_df[feature_cols_no_target] = feature_scaler.transform(
    test_df[feature_cols_no_target]
)


# Log transform target
train_df[TARGET_COL] = np.log1p(train_df[TARGET_COL])
test_df[TARGET_COL] = np.log1p(test_df[TARGET_COL])

# Scale target (fit only on train)
train_df[TARGET_COL] = target_scaler.fit_transform(train_df[[TARGET_COL]])
test_df[TARGET_COL] = target_scaler.transform(test_df[[TARGET_COL]])

# 🔥 NOW SAVE SCALERS
joblib.dump(feature_scaler, "results/models/feature_scaler.pkl")
joblib.dump(target_scaler, "results/models/target_scaler.pkl")
print("Scalers saved successfully.")
# Merge back for rolling + forecasting
df_scaled = pd.concat([train_df, test_df]).reset_index(drop=True)

# ==========================================
# TRAIN MODEL (Optional)
# ==========================================

# Uncomment if retraining required
# print("\n===== TRAINING MODEL =====")
# model = train_multiscale(train_df, feature_cols)

# ==========================================
# FINAL MODEL EVALUATION
# ==========================================

print("\n===== FINAL MODEL EVALUATION =====")

evaluate_multiscale(df_scaled, feature_cols, target_scaler)
print("\nModel evaluation completed successfully.")
print("Training feature columns:")
print(feature_cols)
print("Final feature order used for training:")
print(train_df[feature_cols].columns.tolist())
# ==========================================
# ROLLING WINDOW VALIDATION
# ==========================================

# print("\n===== ROLLING WINDOW VALIDATION =====")

# rolling_window_evaluation(df_scaled, feature_cols, target_scaler)

# print("\nRolling validation completed successfully.")

# ==========================================
# INTERACTIVE FORECAST SYSTEM
# ==========================================

# print("\n===== FUTURE FORECAST SYSTEM =====")

generate_forecast_outputs(df_scaled, feature_cols, target_scaler)

print("\nSystem execution completed successfully.")