import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from preprocessing.dataset_multitask import DatasetMultiTask
from models.hybrid_multitask import HybridMultiTask
from config import *

def evaluate_multiscale(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)

    model.load_state_dict(
        torch.load("results/models/hybrid_multiscale_quantile.pth", weights_only=True)
    )

    model.eval()

    actual_24, actual_7d, actual_30d = [], [], []
    p10_24, p50_24, p90_24 = [], [], []
    p10_7d, p50_7d, p90_7d = [], [], []
    p10_30d, p50_30d, p90_30d = [], [], []

    with torch.no_grad():
        for x, y24, y7d, y30d in loader:

            x = x.to(DEVICE)

            pred24, pred7d, pred30d, _ = model(x)

            # 24H
            actual_24.append(y24.numpy().flatten())
            p10_24.append(pred24[:, :, 0].cpu().numpy().flatten())
            p50_24.append(pred24[:, :, 1].cpu().numpy().flatten())
            p90_24.append(pred24[:, :, 2].cpu().numpy().flatten())

            # 7D
            actual_7d.append(y7d.numpy().flatten())
            p10_7d.append(pred7d[:, :, 0].cpu().numpy().flatten())
            p50_7d.append(pred7d[:, :, 1].cpu().numpy().flatten())
            p90_7d.append(pred7d[:, :, 2].cpu().numpy().flatten())

            # 30D
            actual_30d.append(y30d.numpy().flatten())
            p10_30d.append(pred30d[:, :, 0].cpu().numpy().flatten())
            p50_30d.append(pred30d[:, :, 1].cpu().numpy().flatten())
            p90_30d.append(pred30d[:, :, 2].cpu().numpy().flatten())

    # Convert to arrays
    actual_24 = np.concatenate(actual_24)
    p10_24 = np.concatenate(p10_24)
    p50_24 = np.concatenate(p50_24)
    p90_24 = np.concatenate(p90_24)

    actual_7d = np.concatenate(actual_7d)
    p10_7d = np.concatenate(p10_7d)
    p50_7d = np.concatenate(p50_7d)
    p90_7d = np.concatenate(p90_7d)

    actual_30d = np.concatenate(actual_30d)
    p10_30d = np.concatenate(p10_30d)
    p50_30d = np.concatenate(p50_30d)
    p90_30d = np.concatenate(p90_30d)

    # ==============================
    # METRICS (Median)
    # ==============================

    def compute_metrics(actual, pred):
        mae = np.mean(np.abs(actual - pred))
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        return mae, rmse

    mae_24, rmse_24 = compute_metrics(actual_24, p50_24)
    mae_7d, rmse_7d = compute_metrics(actual_7d, p50_7d)
    mae_30d, rmse_30d = compute_metrics(actual_30d, p50_30d)

    # ==============================
    # COVERAGE
    # ==============================

    def coverage(actual, lower, upper):
        inside = np.logical_and(actual >= lower, actual <= upper)
        return np.mean(inside)

    cov_24 = coverage(actual_24, p10_24, p90_24)
    cov_7d = coverage(actual_7d, p10_7d, p90_7d)
    cov_30d = coverage(actual_30d, p10_30d, p90_30d)

    # ==============================
    # PRINT RESULTS
    # ==============================

    print("\n===== FINAL TEST RESULTS =====")
    print("24H MAE:", mae_24)
    print("24H RMSE:", rmse_24)
    print("24H Coverage:", cov_24)

    print("\n7D MAE:", mae_7d)
    print("7D RMSE:", rmse_7d)
    print("7D Coverage:", cov_7d)

    print("\n30D MAE:", mae_30d)
    print("30D RMSE:", rmse_30d)
    print("30D Coverage:", cov_30d)

    return {
        "24H": (mae_24, rmse_24, cov_24),
        "7D": (mae_7d, rmse_7d, cov_7d),
        "30D": (mae_30d, rmse_30d, cov_30d)
    }
