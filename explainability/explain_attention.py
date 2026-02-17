import torch
import numpy as np
import matplotlib.pyplot as plt
from models.hybrid_multitask import HybridMultiTask
from config import *
from preprocessing.dataset_multitask import DatasetMultiTask

def visualize_attention(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)

    # Take one example from test set
    x, y24, y7d, y30d = dataset[0]
    x = x.unsqueeze(0).to(DEVICE)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, weights_only=True)
    )

    model.eval()

    with torch.no_grad():
        pred24, pred7d, pred30d, attn_weights = model(x)

    # Average attention across heads
    attn = attn_weights.mean(dim=1).squeeze().cpu().numpy()

    plt.figure(figsize=(10,5))
    plt.plot(attn[-1])  # last timestep attention
    plt.title("Attention Weights (Influence of Past Timesteps)")
    plt.xlabel("Past Hour Index (0 = oldest, 719 = most recent)")
    plt.ylabel("Attention Importance")
    plt.savefig("results/plots/attention_explanation.png")
    plt.close()

    print("Attention visualization saved.")
