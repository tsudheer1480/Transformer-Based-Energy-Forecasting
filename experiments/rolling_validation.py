import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from preprocessing.dataset_multitask import DatasetMultiTask
from models.hybrid_multitask import HybridMultiTask
from config import DEVICE, MODEL_PATH


def rolling_window_evaluation(df, feature_cols, target_scaler):

    print("\n===== Rolling Window Evaluation (24H Only) =====")

    window_days = 90
    step_days = 30

    start_date = df["time"].min()
    end_date = df["time"].max()

    current_start = start_date

    all_mae = []
    all_pct = []

    while True:

        window_end = current_start + pd.Timedelta(days=window_days)

        if window_end > end_date:
            break

        window_df = df[
            (df["time"] >= current_start) &
            (df["time"] < window_end)
        ]

        print(f"\nWindow: {current_start.date()} → {window_end.date()}")

        dataset = DatasetMultiTask(window_df, feature_cols)

        if len(dataset) < 5:
            print("Skipped (not enough usable sequences).")
            current_start += pd.Timedelta(days=step_days)
            continue

        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        model = HybridMultiTask(len(feature_cols)).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()

        actual = []
        pred = []

        with torch.no_grad():
            for x, y24, _, _ in loader:
                x = x.to(DEVICE)
                p24, _, _, _ = model(x)

                p24 = p24[:, :, 1]  # median (P50)

                actual.append(y24.numpy())
                pred.append(p24.cpu().numpy())

        actual = np.concatenate(actual)
        pred = np.concatenate(pred)

        # ===============================
        # CORRECT INVERSE (log + scaler)
        # ===============================

        def inverse_to_mw(arr):
            scaled = target_scaler.inverse_transform(arr.reshape(-1, 1))
            return np.expm1(scaled).flatten()   # reverse log1p

        actual_real = inverse_to_mw(actual)
        pred_real = inverse_to_mw(pred)

        mae_mw = np.mean(np.abs(actual_real - pred_real))
        error_pct = (mae_mw / np.mean(actual_real)) * 100

        print(f"24H MAE (MW): {mae_mw:,.2f}")
        print(f"24H Error %: {error_pct:.2f}%")

        all_mae.append(mae_mw)
        all_pct.append(error_pct)

        current_start += pd.Timedelta(days=step_days)

    print("\n===== Rolling Stability Summary =====")

    if len(all_mae) > 0:
        print(f"Average MAE (MW): {np.mean(all_mae):,.2f}")
        print(f"Average Error %: {np.mean(all_pct):.2f}%")
        print(f"Std Dev (%): {np.std(all_pct):.2f}%")
    else:
        print("No valid rolling windows.")