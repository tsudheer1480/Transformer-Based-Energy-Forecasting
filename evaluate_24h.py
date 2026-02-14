import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from preprocessing.dataset_24h import Dataset24H
from models.hybrid_24h import Hybrid24H
from config import *

def evaluate_24h(df, feature_cols):

    dataset = Dataset24H(df, feature_cols)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = Hybrid24H(len(feature_cols)).to(DEVICE)
    model.load_state_dict(torch.load("results/models/hybrid_24h.pth"))
    model.eval()

    actual = []
    predicted = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds, _ = model(x)

            actual.append(y.numpy().flatten())
            predicted.append(preds.cpu().numpy().flatten())

    actual = np.concatenate(actual)
    predicted = np.concatenate(predicted)

    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    results_df = pd.DataFrame({
        "actual": actual,
        "predicted": predicted
    })

    results_df.to_csv("results/predictions/predictions_24h.csv", index=False)

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))

    with open("results/metrics/metrics_24h.txt", "w") as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"RMSE: {rmse}\n")

    print("MAE:", mae)
    print("RMSE:", rmse)

    plt.figure(figsize=(10,5))
    plt.plot(actual[:500], label="Actual")
    plt.plot(predicted[:500], label="Predicted")
    plt.legend()
    plt.title("Actual vs Predicted (24h)")
    plt.savefig("results/plots/actual_vs_predicted_24h.png")
    plt.close()

    print("Evaluation complete.")
