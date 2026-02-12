import torch

DATA_PATH = r"energy_env\data\processed\final_energy_forecasting_dataset.csv"

TIME_COL = "timestamp"
TARGET_COL = "load"

INPUT_LEN = 168
HORIZON = 24

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

BATCH_SIZE = 32
EPOCHS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
