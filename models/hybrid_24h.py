import torch
import torch.nn as nn
from config import MODEL_CONFIG

class Hybrid24H(nn.Module):
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

        self.fc = nn.Linear(hidden, MODEL_CONFIG["prediction_length"])

    def forward(self, x):
        x = self.embedding(x)
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(attn_out + x)
        context = x[:, -1, :]
        output = self.fc(context)
        return output, attn_weights
