import torch
import numpy as np
from torch.utils.data import DataLoader
from preprocessing.dataset_multitask import DatasetMultiTask
from models.hybrid_multitask import HybridMultiTask
from config import DEVICE, MODEL_PATH


def evaluate_multiscale(df, feature_cols, target_scaler):

    dataset = DatasetMultiTask(df, feature_cols)

    if len(dataset) == 0:
        print("Evaluation skipped (insufficient samples).")
        return None

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    actual_24, pred_24 = [], []
    actual_7d, pred_7d = [], []
    actual_30d, pred_30d = [], []

    with torch.no_grad():
        for x, y24, y7d, y30d in loader:

            x = x.to(DEVICE)

            p24, p7d, p30d, _ = model(x)

            # Median prediction (P50)
            p24 = p24[:, :, 1]
            p7d = p7d[:, :, 1]
            p30d = p30d[:, :, 1]

            actual_24.append(y24.numpy())
            pred_24.append(p24.cpu().numpy())

            actual_7d.append(y7d.numpy())
            pred_7d.append(p7d.cpu().numpy())

            actual_30d.append(y30d.numpy())
            pred_30d.append(p30d.cpu().numpy())

    # ==============================
    # CONCATENATE
    # ==============================

    actual_24 = np.concatenate(actual_24)
    pred_24 = np.concatenate(pred_24)

    actual_7d = np.concatenate(actual_7d)
    pred_7d = np.concatenate(pred_7d)

    actual_30d = np.concatenate(actual_30d)
    pred_30d = np.concatenate(pred_30d)

    # ==============================
    # INVERSE SCALE + EXP BACK TO MW
    # ==============================

    def inverse_to_mw(arr):
        scaled = target_scaler.inverse_transform(arr.reshape(-1, 1))
        return np.expm1(scaled).flatten()   # reverse log1p

    actual_24_real = inverse_to_mw(actual_24)
    pred_24_real = inverse_to_mw(pred_24)

    actual_7d_real = inverse_to_mw(actual_7d)
    pred_7d_real = inverse_to_mw(pred_7d)

    actual_30d_real = inverse_to_mw(actual_30d)
    pred_30d_real = inverse_to_mw(pred_30d)

    # ==============================
    # MAE (MW)
    # ==============================

    mae_24_mw = np.mean(np.abs(actual_24_real - pred_24_real))
    mae_7d_mw = np.mean(np.abs(actual_7d_real - pred_7d_real))
    mae_30d_mw = np.mean(np.abs(actual_30d_real - pred_30d_real))

    # ==============================
    # ERROR PERCENTAGE (per horizon)
    # ==============================

    error_pct_24 = (mae_24_mw / np.mean(actual_24_real)) * 100
    error_pct_7d = (mae_7d_mw / np.mean(actual_7d_real)) * 100
    error_pct_30d = (mae_30d_mw / np.mean(actual_30d_real)) * 100

    # ==============================
    # PRINT RESULTS
    # ==============================

    print("\n===== FINAL TEST RESULTS (REAL MW) =====")

    print(f"24H MAE (MW): {mae_24_mw:,.2f}")
    print(f"24H Error Percentage: {error_pct_24:.2f}%")

    print(f"\n7D MAE (MW): {mae_7d_mw:,.2f}")
    print(f"7D Error Percentage: {error_pct_7d:.2f}%")

    print(f"\n30D MAE (MW): {mae_30d_mw:,.2f}")
    print(f"30D Error Percentage: {error_pct_30d:.2f}%")

    return {
        "24H_MAE_MW": mae_24_mw,
        "24H_Error_%": error_pct_24,
        "7D_MAE_MW": mae_7d_mw,
        "7D_Error_%": error_pct_7d,
        "30D_MAE_MW": mae_30d_mw,
        "30D_Error_%": error_pct_30d
    }
