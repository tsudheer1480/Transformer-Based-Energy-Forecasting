import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import DATA_PATH, FEATURE_CONFIG, TARGET_COL
from train import train_multiscale
from evaluate_multiscale import evaluate_multiscale
from explain_attention import visualize_attention
from explain_features import feature_sensitivity

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

df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
df[TARGET_COL] = target_scaler.fit_transform(df[[TARGET_COL]])

# ==========================
# TRAIN / TEST SPLIT (80/20)
# ==========================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print("Train size:", train_df.shape)
print("Test size:", test_df.shape)

# ==========================
# TRAIN MODEL
# ==========================

model = train_multiscale(train_df, feature_cols)

# ==========================
# EVALUATE MODEL
# ==========================

results = evaluate_multiscale(test_df, feature_cols)

# ==========================
# EXPLAINABILITY
# ==========================

print("\nGenerating Attention Explanation...")
visualize_attention(test_df, feature_cols)

print("\nRunning Feature Sensitivity Analysis...")
feature_sensitivity(test_df, feature_cols)

print("\nPipeline completed successfully.")
