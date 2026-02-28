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


# ==============================
# Training Function
# ==============================

def train_multiscale(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = HybridMultiTask(len(feature_cols)).to(device)

    # 🔥 AdamW (better generalization than Adam)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_CONFIG["learning_rate"],
        weight_decay=1e-4
    )

    # 🔥 Cosine LR Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    loss_history = []

    print("\n===== Training Started =====")

    for epoch in range(EPOCHS):

        model.train()
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

            # 🔥 Better horizon weighting (focus more on 24H)
            loss = loss24 + 0.3 * loss7d + 0.1 * loss30d

            loss.backward()

            # ✅ GRADIENT CLIPPING (prevents seasonal instability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.6f}")

    print("\n===== Training Complete =====")

    # ==========================
    # SAVE FINAL MODEL
    # ==========================

    os.makedirs("results/models", exist_ok=True)

    torch.save(
        model.state_dict(),
        MODEL_PATH
    )

    print(f"\nFinal model saved to {MODEL_PATH}")

    return model
