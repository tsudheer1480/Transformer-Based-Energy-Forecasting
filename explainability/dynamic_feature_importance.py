import numpy as np
import torch


def create_sequences(data, lookback):

    sequences = []

    for i in range(len(data) - lookback):
        sequences.append(data[i:i + lookback])

    return np.array(sequences)


def compute_dynamic_feature_importance(
        model,
        df,
        feature_cols,
        device,
        lookback
):

    model.eval()

    X = df[feature_cols].values.astype(np.float32)

    X_seq = create_sequences(X, lookback)

    X_tensor = torch.tensor(X_seq).to(device)

    with torch.no_grad():

        # model output
        pred24, pred7d, pred30d, _ = model(X_tensor)

        # use median quantile prediction
        baseline_pred = pred24[:, :, 1]

        baseline_pred = baseline_pred.cpu().numpy()

    importance = {}

    for i, col in enumerate(feature_cols):

        X_perm = X.copy()

        np.random.shuffle(X_perm[:, i])

        X_perm_seq = create_sequences(X_perm, lookback)

        X_perm_tensor = torch.tensor(X_perm_seq).to(device)

        with torch.no_grad():

            perm24, perm7d, perm30d, _ = model(X_perm_tensor)

            perm_pred = perm24[:, :, 1]

            perm_pred = perm_pred.cpu().numpy()

        diff = np.mean(np.abs(baseline_pred - perm_pred))

        importance[col] = float(diff)

    sorted_features = sorted(
        importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_features