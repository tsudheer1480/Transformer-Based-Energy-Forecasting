import pandas as pd
import numpy as np

def load_series(path, target_col):
    df = pd.read_csv(path)
    df[target_col] = df[target_col].astype(float)
    return df[target_col].values

def split_series(series, train_ratio, val_ratio):
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = series[:train_end]
    val = series[train_end:val_end]
    test = series[val_end:]

    return train, val, test
