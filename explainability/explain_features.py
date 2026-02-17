import torch
import numpy as np
from models.hybrid_multitask import HybridMultiTask
from config import *
from preprocessing.dataset_multitask import DatasetMultiTask


def feature_sensitivity(df, feature_cols):

    dataset = DatasetMultiTask(df, feature_cols)

    x, y24, y7d, y30d = dataset[0]
    x_original = x.clone().unsqueeze(0).to(DEVICE)

    model = HybridMultiTask(len(feature_cols)).to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )

    model.eval()

    with torch.no_grad():
        base_pred, _, _, _ = model(x_original)

    base_value = base_pred[:, :, 1].mean().item()

    print("\nFeature Sensitivity (Impact on 24H Median Forecast):")

    # ✅ Create dictionary
    feature_impacts = {}

    for i, feature in enumerate(feature_cols):

        x_modified = x.clone()
        x_modified[:, i] += 0.1
        x_modified = x_modified.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            new_pred, _, _, _ = model(x_modified)

        new_value = new_pred[:, :, 1].mean().item()

        impact = new_value - base_value

        # ✅ Store impact
        feature_impacts[feature] = impact

        print(f"{feature}: {impact:.6f}")

    # ✅ Return dictionary
    return feature_impacts
