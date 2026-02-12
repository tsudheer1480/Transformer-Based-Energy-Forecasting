import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, horizon=24):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.horizon = horizon

    def forward(self, x):
        _, (h, c) = self.encoder(x)
        decoder_input = x[:, -1:, :]
        outputs = []

        for _ in range(self.horizon):
            out, (h, c) = self.decoder(decoder_input, (h, c))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred

        return torch.cat(outputs, dim=1)


def train_seq2seq(model, X, y, epochs=5, batch_size=32, device="cpu"):
    model.to(device)
    X = X.to(device)
    y = y.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb.unsqueeze(-1))
            loss.backward()
            optimizer.step()
