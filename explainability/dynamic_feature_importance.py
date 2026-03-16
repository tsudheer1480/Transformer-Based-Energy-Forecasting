import numpy as np
import torch


def compute_dynamic_feature_importance(
        model,
        df,
        feature_cols,
        device,
        lookback
):

    model.eval()

    # use only the last lookback window
    X = df[feature_cols].values.astype(np.float32)

    X_window = X[-lookback:]

    X_tensor = torch.tensor(X_window).unsqueeze(0).to(device)

    with torch.no_grad():

        pred24, _, _, _ = model(X_tensor)

        baseline_pred = pred24[:, :, 1].cpu().numpy()

    importance = {}

    for i, col in enumerate(feature_cols):

        X_perm = X_window.copy()

        np.random.shuffle(X_perm[:, i])

        X_perm_tensor = torch.tensor(X_perm).unsqueeze(0).to(device)

        with torch.no_grad():

            perm24, _, _, _ = model(X_perm_tensor)

            perm_pred = perm24[:, :, 1].cpu().numpy()

        diff = np.mean(np.abs(baseline_pred - perm_pred))

        importance[col] = float(diff)

    sorted_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_features