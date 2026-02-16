import torch
import os
from torch.utils.data import DataLoader
from models.hybrid_multitask import HybridMultiTask
from preprocessing.dataset_multitask import DatasetMultiTask
from config import *

device = torch.device(DEVICE)

# ==============================
# Quantile Loss
# ==============================

def quantile_loss(preds, target, quantiles):

    losses = []

    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i]
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(torch.mean(loss))

    return sum(losses)


def train_multiscale(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridMultiTask(len(feature_cols)).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MODEL_CONFIG["learning_rate"]
    )

    # 🔥 Cosine scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    loss_history = []

    for epoch in range(EPOCHS):
        total_loss = 0

        for x, y24, y7d, y30d in loader:

            x = x.to(device)
            y24 = y24.to(device)
            y7d = y7d.to(device)
            y30d = y30d.to(device)

            optimizer.zero_grad()

            pred24, pred7d, pred30d, _ = model(x)

            loss24 = quantile_loss(pred24, y24, QUANTILES)
            loss7d = quantile_loss(pred7d, y7d, QUANTILES)
            loss30d = quantile_loss(pred30d, y30d, QUANTILES)

            # 🔥 Adjusted horizon weights
            loss = loss24 + 0.4 * loss7d + 0.15 * loss30d

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        loss_history.append(total_loss)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    os.makedirs("results/models", exist_ok=True)

    torch.save(model.state_dict(), "results/models/hybrid_model_final_positional_v1.pth")

    print("Quantile multi-scale model saved.")

    return model
