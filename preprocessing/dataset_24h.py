import torch
from torch.utils.data import Dataset
from config import LOOKBACK, HORIZON, TARGET_COL

class Dataset24H(Dataset):
    def __init__(self, df, feature_cols):
        self.features = df[feature_cols].values
        self.target = df[TARGET_COL].values

    def __len__(self):
        return len(self.features) - LOOKBACK - HORIZON

    def __getitem__(self, idx):
        x = self.features[idx: idx + LOOKBACK]
        y = self.target[idx + LOOKBACK: idx + LOOKBACK + HORIZON]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
