import numpy as np
import torch
import pandas as pd

from config import *
from utils.data_loader import load_series, split_series
from utils.dataset import create_sliding_windows
from utils.evaluate import evaluate_model
from sklearn.preprocessing import StandardScaler

# ---------------- BASELINES ----------------
from baselines.naive import naive_forecast
from baselines.sarima import sarima_forecast
from baselines.xgboost import train_xgboost, forecast_xgboost
from baselines.lstm import LSTMModel, train_lstm
from baselines.seq2seq_lstm import Seq2SeqLSTM, train_seq2seq
from baselines.deepar import DeepAR, train_deepar
from baselines.mqrnn import MQRNN, train_mqrnn

results = []

# ================= LOAD DATA =================
series = load_series(DATA_PATH, TARGET_COL)
train, val, test = split_series(series, TRAIN_RATIO, VAL_RATIO)

test_target = test[:HORIZON]   # 🔹 ALWAYS RAW (original scale)

# ================= SCALING (FOR DEEP MODELS ONLY) =================
# Deep learning models REQUIRE normalization
# Classical models (Naive, SARIMA, XGBoost) use RAW data

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()
# ================================================================


# ================= NAIVE =================
print("Running Naive...")
naive_pred = naive_forecast(train, HORIZON)
results.append(evaluate_model("Naive", test_target, naive_pred))


# ================= SARIMA =================
print("Running SARIMA...")
sarima_pred = sarima_forecast(train, HORIZON)
results.append(evaluate_model("SARIMA", test_target, sarima_pred))


# ================= XGBOOST =================
print("Running XGBoost...")
xgb_model = train_xgboost(train)
xgb_pred = forecast_xgboost(xgb_model, train, HORIZON)
results.append(evaluate_model("XGBoost", test_target, xgb_pred))


# ================= LSTM =================
print("Running LSTM...")

# 🔹 SCALING APPLIED (training)
X_train, y_train = create_sliding_windows(train_scaled, INPUT_LEN, 1)
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float()

lstm = LSTMModel().to(DEVICE)
train_lstm(lstm, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE)

# 🔹 SCALING APPLIED (prediction input)
last_input = torch.tensor(
    train_scaled[-INPUT_LEN:]
).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

lstm_preds_scaled = []

for _ in range(HORIZON):
    pred_scaled = lstm(last_input).item()
    lstm_preds_scaled.append(pred_scaled)

    last_input = torch.cat(
        [last_input[:, 1:], torch.tensor([[[pred_scaled]]]).to(DEVICE)],
        dim=1
    )

# 🔹 INVERSE SCALING (back to original units)
lstm_preds = scaler.inverse_transform(
    np.array(lstm_preds_scaled).reshape(-1, 1)
).flatten()

results.append(evaluate_model("LSTM", test_target, lstm_preds))


# ================= SEQ2SEQ =================
print("Running Seq2Seq...")

# 🔹 SCALING APPLIED (training)
X_train, y_train = create_sliding_windows(train_scaled, INPUT_LEN, HORIZON)
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float()

seq2seq = Seq2SeqLSTM(horizon=HORIZON)

train_seq2seq(
    seq2seq,
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

# 🔹 SCALING APPLIED (prediction input)
last_input = torch.tensor(
    train_scaled[-INPUT_LEN:]
).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

seq2seq_pred_scaled = seq2seq(last_input).detach().cpu().numpy().flatten()

# 🔹 INVERSE SCALING
seq2seq_pred = scaler.inverse_transform(
    seq2seq_pred_scaled.reshape(-1, 1)
).flatten()

results.append(evaluate_model("Seq2Seq", test_target, seq2seq_pred))


# ================= DEEPAR =================
print("Running DeepAR...")

# 🔹 SCALING APPLIED (training)
X_train, y_train = create_sliding_windows(train_scaled, INPUT_LEN, 1)
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float()

deepar = DeepAR()

train_deepar(
    deepar,
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

# 🔹 SCALING APPLIED (prediction input)
last_input = torch.tensor(
    train_scaled[-INPUT_LEN:]
).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

mu_scaled, _ = deepar(last_input)

# 🔹 INVERSE SCALING
deepar_pred = scaler.inverse_transform(
    mu_scaled.detach().cpu().numpy().reshape(-1, 1)
).flatten()

results.append(evaluate_model("DeepAR", test_target, deepar_pred))


# ================= MQRNN =================
print("Running MQRNN...")

# 🔹 SCALING APPLIED (training)
X_train, y_train = create_sliding_windows(train_scaled, INPUT_LEN, HORIZON)
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float()

mqrnn = MQRNN(horizon=HORIZON)

train_mqrnn(
    mqrnn,
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

# 🔹 SCALING APPLIED (prediction input)
last_input = torch.tensor(
    train_scaled[-INPUT_LEN:]
).float().unsqueeze(0).unsqueeze(-1).to(DEVICE)

mqrnn_pred_scaled = mqrnn(last_input).detach().cpu().numpy().flatten()

# 🔹 INVERSE SCALING
mqrnn_pred = scaler.inverse_transform(
    mqrnn_pred_scaled.reshape(-1, 1)
).flatten()

results.append(evaluate_model("MQRNN", test_target, mqrnn_pred))

# ================= SAVE PREDICTIONS FOR PLOTTING =================
np.save("plot_actual.npy", test_target)
np.save("plot_naive.npy", naive_pred)
np.save("plot_lstm.npy", lstm_preds)
np.save("plot_deepar.npy", deepar_pred)
# ================================================================

# ================= RESULTS =================
df = pd.DataFrame(results)
print(df)
df.to_csv("baseline_results_scaled.csv", index=False)
