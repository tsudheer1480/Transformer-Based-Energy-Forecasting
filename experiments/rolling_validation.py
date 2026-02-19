import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from models.hybrid_multitask import HybridMultiTask
from config import DEVICE, MODEL_PATH, LOOKBACK, TARGET_COL, BATCH_SIZE


# ==========================================================
# 24H Rolling Dataset (Lightweight)
# ==========================================================

class Dataset24H(Dataset):
    def __init__(self, df, feature_cols):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.lookback = LOOKBACK
        self.horizon = 24

    def __len__(self):
        return max(0, len(self.df) - self.lookback - self.horizon)

    def __getitem__(self, idx):
        x = self.df[self.feature_cols].iloc[idx:idx+self.lookback].values
        y = self.df[TARGET_COL].iloc[idx+self.lookback:
                                     idx+self.lookback+self.horizon].values

        return torch.tensor(x, dtype=torch.float32), \
               torch.tensor(y, dtype=torch.float32)


# ==========================================================
# Rolling Window Evaluation (Fixed Model)
# ==========================================================

def rolling_window_evaluation(df, feature_cols):

    print("\n===== Rolling Window Evaluation (Fixed Model) =====")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])

    window_days = 60   # 60-day evaluation window
    step_days = 30     # Move 30 days each iteration

    start_date = df["time"].min()
    end_date = df["time"].max()

    all_mae = []

    current_start = start_date

    while current_start + pd.Timedelta(days=window_days) <= end_date:

        current_end = current_start + pd.Timedelta(days=window_days)

        window_df = df[
            (df["time"] >= current_start) &
            (df["time"] < current_end)
        ]

        print(f"\nWindow: {current_start.date()} → {current_end.date()}")

        dataset = Dataset24H(window_df, feature_cols)

        if len(dataset) <= 0:
            print("Skipped (insufficient samples).")
            current_start += pd.Timedelta(days=step_days)
            continue

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Load trained model (DO NOT retrain)
        model = HybridMultiTask(len(feature_cols)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        actual = []
        predicted = []

        with torch.no_grad():
            for x, y in loader:

                x = x.to(DEVICE)

                pred24, _, _, _ = model(x)
                pred24 = pred24[:, :, 1]  # median

                actual.append(y.numpy())
                predicted.append(pred24.cpu().numpy())

        actual = np.concatenate(actual)
        predicted = np.concatenate(predicted)

        mae = np.mean(np.abs(actual - predicted))
        print(f"Rolling 24H MAE: {mae:.4f}")

        all_mae.append(mae)

        current_start += pd.Timedelta(days=step_days)

    print("\n===== Rolling Stability Summary =====")

    if len(all_mae) > 0:
        print(f"Average MAE across windows: {np.mean(all_mae):.4f}")
        print(f"MAE Std Deviation: {np.std(all_mae):.4f}")
    else:
        print("No valid rolling windows found.")
