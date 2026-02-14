import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


def build_tft_model(training_dataset, config):
    model = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        attention_head_size=config["attention_heads"],
        dropout=config["dropout"],
        loss=QuantileLoss(),
    )
    return model
