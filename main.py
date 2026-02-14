import pandas as pd
from sklearn.preprocessing import StandardScaler
from train import train_multitask
from evaluate_multitask import evaluate_multitask
from config import DATA_PATH, FEATURE_CONFIG, TARGET_COL

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

df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
df[TARGET_COL] = target_scaler.fit_transform(df[[TARGET_COL]])

# ==========================
# TRAIN
# ==========================

model = train_multitask(df, feature_cols)

# ==========================
# EVALUATE
# ==========================

evaluate_multitask(df, feature_cols)
