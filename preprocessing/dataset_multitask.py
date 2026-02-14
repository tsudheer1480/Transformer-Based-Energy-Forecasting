import torch
from torch.utils.data import Dataset
from config import LOOKBACK, HORIZON_24, HORIZON_7D, TARGET_COL

class DatasetMultiTask(Dataset):
    def __init__(self, df, feature_cols):
        self.features = df[feature_cols].values
        self.target = df[TARGET_COL].values

    def __len__(self):
        return len(self.features) - LOOKBACK - HORIZON_7D

    def __getitem__(self, idx):
        x = self.features[idx: idx + LOOKBACK]

        y_24 = self.target[idx + LOOKBACK:
                           idx + LOOKBACK + HORIZON_24]

        y_7d = self.target[idx + LOOKBACK:
                           idx + LOOKBACK + HORIZON_7D]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_24, dtype=torch.float32),
            torch.tensor(y_7d, dtype=torch.float32),
        )
