import numpy as np
import torch

def attention_explanation(attention, horizon=""):

    if attention is None:
        return f"Attention weights unavailable for {horizon} forecast."

    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    avg_attention = np.mean(attention)

    explanation = f"""
Attention Insight ({horizon}):

The attention mechanism indicates average temporal 
importance weight of {avg_attention:.4f}, suggesting 
recent historical patterns significantly influence 
current forecast generation.

Higher attention weights imply stronger short-term 
temporal dependency within the encoder representation.
"""

    return explanation