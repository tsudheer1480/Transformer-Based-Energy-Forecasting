import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MQRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=24):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        return self.fc(h[-1])


def train_mqrnn(model, X, y, epochs=5, batch_size=32, device="cpu"):
    model.to(device)
    X = X.to(device)
    y = y.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()   # OK for baseline (quantile loss optional)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
