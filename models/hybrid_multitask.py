import torch
import torch.nn as nn
import math
from config import (
    MODEL_CONFIG,
    HORIZON_24,
    HORIZON_7D,
    HORIZON_30D,
    N_QUANTILES,
    LOOKBACK
)

# ==============================
# Sinusoidal Positional Encoding
# ==============================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=LOOKBACK):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# ==============================
# Multi-Scale Transformer
# ==============================

class HybridMultiTask(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = MODEL_CONFIG["hidden_size"]
        heads = MODEL_CONFIG["attention_heads"]

        self.weather_dim = 5
        self.main_dim = input_dim - self.weather_dim

        self.main_embedding = nn.Linear(self.main_dim, hidden)
        self.weather_embedding = nn.Linear(self.weather_dim, hidden // 2)

        self.combine = nn.Linear(hidden + hidden // 2, hidden)

        # 🔥 Positional Encoding
        self.positional_encoding = PositionalEncoding(hidden)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden)

        self.fc_24 = nn.Linear(hidden, HORIZON_24 * N_QUANTILES)
        self.fc_7d = nn.Linear(hidden, HORIZON_7D * N_QUANTILES)
        self.fc_30d = nn.Linear(hidden, HORIZON_30D * N_QUANTILES)

    def forward(self, x):

        main_features = x[:, :, :-self.weather_dim]
        weather_features = x[:, :, -self.weather_dim:]

        main_emb = self.main_embedding(main_features)
        weather_emb = self.weather_embedding(weather_features)

        combined = torch.cat([main_emb, weather_emb], dim=-1)
        x = self.combine(combined)

        # 🔥 Add positional encoding here
        x = self.positional_encoding(x)

        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(attn_out + x)

        context = x[:, -1, :]

        out_24 = self.fc_24(context)
        out_7d = self.fc_7d(context)
        out_30d = self.fc_30d(context)

        out_24 = out_24.view(-1, HORIZON_24, N_QUANTILES)
        out_7d = out_7d.view(-1, HORIZON_7D, N_QUANTILES)
        out_30d = out_30d.view(-1, HORIZON_30D, N_QUANTILES)

        return out_24, out_7d, out_30d, attn_weights
