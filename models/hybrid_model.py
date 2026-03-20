import torch
import torch.nn as nn
from config import MODEL_CONFIG, HORIZON_24, HORIZON_7D, HORIZON_30D

class HybridMultiTask(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        hidden = MODEL_CONFIG["hidden_size"]
        heads = MODEL_CONFIG["attention_heads"]

        self.embedding = nn.Linear(input_dim, hidden)

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=heads,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden)

        # Three output heads
        self.fc_24 = nn.Linear(hidden, HORIZON_24)
        self.fc_7d = nn.Linear(hidden, HORIZON_7D)
        self.fc_30d = nn.Linear(hidden, HORIZON_30D)

    def forward(self, x):
        x = self.embedding(x)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(attn_out + x)

        context = x[:, -1, :]

        out_24 = self.fc_24(context)
        out_7d = self.fc_7d(context)
        out_30d = self.fc_30d(context)

        return out_24, out_7d, out_30d, attn_weights
