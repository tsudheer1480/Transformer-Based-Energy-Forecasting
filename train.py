import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.hybrid_multitask import HybridMultiTask
from preprocessing.dataset_multitask import DatasetMultiTask
from config import *

device = torch.device(DEVICE)

def train_multitask(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridMultiTask(len(feature_cols)).to(device)

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

        for batch_idx, (x, y24, y7d) in enumerate(loader):

            x = x.to(device)
            y24 = y24.to(device)
            y7d = y7d.to(device)

            optimizer.zero_grad()

            pred24, pred7d, _ = model(x)

            if epoch == 0 and batch_idx == 0:
                print("24h shape:", pred24.shape)
                print("7d shape:", pred7d.shape)

            loss24 = criterion(pred24, y24)
            loss7d = criterion(pred7d, y7d)

            loss = loss24 + 0.5 * loss7d

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

    torch.save(model.state_dict(), "results/models/hybrid_multitask.pth")

    plt.figure()
    plt.plot(loss_history)
    plt.title("Training Loss (24h + 7d)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("results/plots/loss_multitask.png")
    plt.close()

    print("Multi-task model saved.")

    return model
