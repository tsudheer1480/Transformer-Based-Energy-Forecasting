import numpy as np
import matplotlib.pyplot as plt

# ================= LOAD DATA =================
actual = np.load("plot_actual.npy")
naive = np.load("plot_naive.npy")
lstm = np.load("plot_lstm.npy")
deepar = np.load("plot_deepar.npy")

horizon = len(actual)
x = range(1, horizon + 1)

# ================= PLOT =================
plt.figure(figsize=(10, 5))

plt.plot(x, actual, label="Actual")
plt.plot(x, naive, label="Naive")
plt.plot(x, lstm, label="LSTM")
plt.plot(x, deepar, label="DeepAR")

plt.xlabel("Forecast Horizon (Hours)")
plt.ylabel("Energy Load")
plt.title("24-Hour Load Forecast Comparison")

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
