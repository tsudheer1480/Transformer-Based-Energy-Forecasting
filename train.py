import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.hybrid_24h import Hybrid24H
from preprocessing.dataset_24h import Dataset24H
from config import *

device = torch.device(DEVICE)

def train_24h(df, feature_cols):

    dataset = Dataset24H(df, feature_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Hybrid24H(len(feature_cols)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG["learning_rate"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0

        for batch_idx, (x, y) in enumerate(loader):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            preds, _ = model(x)

            if epoch == 0 and batch_idx == 0:
                print("Prediction shape:", preds.shape)
                print("Target shape:", y.shape)

            loss = criterion(preds, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        loss_history.append(total_loss)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # ==============================
    # SAVE RESULTS
    # ==============================

    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    torch.save(model.state_dict(), "results/models/hybrid_24h.pth")
    print("Model saved successfully.")

    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss Curve (24h)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/plots/loss_curve_24h.png")
    plt.close()

    print("Loss curve saved.")

    return model
