import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DeepAR(nn.Module):
    def __init__(self, input_size=1, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu = nn.Linear(hidden_size, 1)
        self.sigma = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        mu = self.mu(h)
        sigma = torch.exp(self.sigma(h))
        return mu, sigma


def gaussian_nll(y, mu, sigma):
    return torch.mean(torch.log(sigma) + (y - mu) ** 2 / (2 * sigma ** 2))


def train_deepar(model, X, y, epochs=5, batch_size=32, device="cpu"):
    model.to(device)
    X = X.to(device)
    y = y.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = gaussian_nll(yb[:, -1:], mu, sigma)
            loss.backward()
            optimizer.step()
