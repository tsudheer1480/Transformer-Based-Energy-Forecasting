import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from preprocessing.dataset_multitask import DatasetMultiTask
from models.hybrid_multitask import HybridMultiTask
from config import *

def evaluate_multitask(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load("results/models/hybrid_multitask.pth", weights_only=True))
    model.eval()

    actual_24 = []
    pred_24 = []

    actual_7d = []
    pred_7d = []

    with torch.no_grad():
        for x, y24, y7d in loader:

            x = x.to(DEVICE)

            p24, p7d, _ = model(x)

            actual_24.append(y24.numpy().flatten())
            pred_24.append(p24.cpu().numpy().flatten())

            actual_7d.append(y7d.numpy().flatten())
            pred_7d.append(p7d.cpu().numpy().flatten())

    actual_24 = np.concatenate(actual_24)
    pred_24 = np.concatenate(pred_24)

    actual_7d = np.concatenate(actual_7d)
    pred_7d = np.concatenate(pred_7d)

    # ==============================
    # CREATE FOLDERS
    # ==============================
    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    # ==============================
    # SAVE PREDICTIONS
    # ==============================

    pd.DataFrame({
        "actual_24": actual_24,
        "predicted_24": pred_24
    }).to_csv("results/predictions/predictions_24h_multitask.csv", index=False)

    pd.DataFrame({
        "actual_7d": actual_7d,
        "predicted_7d": pred_7d
    }).to_csv("results/predictions/predictions_7d_multitask.csv", index=False)

    # ==============================
    # METRICS
    # ==============================

    mae_24 = np.mean(np.abs(actual_24 - pred_24))
    rmse_24 = np.sqrt(np.mean((actual_24 - pred_24) ** 2))

    mae_7d = np.mean(np.abs(actual_7d - pred_7d))
    rmse_7d = np.sqrt(np.mean((actual_7d - pred_7d) ** 2))

    with open("results/metrics/metrics_multitask.txt", "w") as f:
        f.write(f"24H MAE: {mae_24}\n")
        f.write(f"24H RMSE: {rmse_24}\n\n")
        f.write(f"7D MAE: {mae_7d}\n")
        f.write(f"7D RMSE: {rmse_7d}\n")

    print("24H MAE:", mae_24)
    print("24H RMSE:", rmse_24)
    print("7D MAE:", mae_7d)
    print("7D RMSE:", rmse_7d)

    # ==============================
    # PLOTS
    # ==============================

    plt.figure(figsize=(10,5))
    plt.plot(actual_24[:500], label="Actual 24H")
    plt.plot(pred_24[:500], label="Predicted 24H")
    plt.legend()
    plt.title("24H Forecast (Multitask)")
    plt.savefig("results/plots/actual_vs_predicted_24h_multitask.png")
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(actual_7d[:500], label="Actual 7D")
    plt.plot(pred_7d[:500], label="Predicted 7D")
    plt.legend()
    plt.title("7D Forecast (Multitask)")
    plt.savefig("results/plots/actual_vs_predicted_7d_multitask.png")
    plt.close()

    print("Evaluation completed successfully.")
