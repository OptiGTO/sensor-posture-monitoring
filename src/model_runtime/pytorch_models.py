"""Deliberately small allow-list of deployable PyTorch factories."""

from __future__ import annotations


def build_model(factory_key: str, **parameters):
    import torch.nn as nn

    if factory_key == "mlp":
        return nn.Sequential(nn.Flatten(), nn.LazyLinear(int(parameters.get("hidden_size", 128))), nn.ReLU(), nn.Linear(int(parameters.get("hidden_size", 128)), int(parameters["class_count"])))
    raise ValueError(f"승인되지 않은 PyTorch 모델 factory: {factory_key}")
