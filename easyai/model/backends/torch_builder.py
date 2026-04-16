"""Torch model builder for the unified layer DSL."""

from __future__ import annotations

import torch.nn as nn

from easyai.config.schema import ModelConfig, TaskConfig
from easyai.model.errors import ModelTranslationError


SUPPORTED_TORCH_LAYERS = {
    "Conv1d",
    "Conv2d",
    "DepthwiseConv2d",
    "Linear",
    "BatchNorm1d",
    "BatchNorm2d",
    "MaxPool1d",
    "MaxPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "GlobalAvgPool1d",
    "GlobalAvgPool2d",
    "Flatten",
    "Dropout",
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
}


def _layer_kwargs(layer):
    return layer.model_dump(exclude={"type"})


def _build_torch_layer(layer):
    kwargs = _layer_kwargs(layer)

    if layer.type == "Conv1d":
        return nn.Conv1d(**kwargs)
    if layer.type == "Conv2d":
        return nn.Conv2d(**kwargs)
    if layer.type == "DepthwiseConv2d":
        in_channels = kwargs.pop("in_channels")
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            **kwargs,
        )
    if layer.type == "Linear":
        return nn.Linear(**kwargs)
    if layer.type == "BatchNorm1d":
        return nn.BatchNorm1d(**kwargs)
    if layer.type == "BatchNorm2d":
        return nn.BatchNorm2d(**kwargs)
    if layer.type == "MaxPool1d":
        return nn.MaxPool1d(**kwargs)
    if layer.type == "MaxPool2d":
        return nn.MaxPool2d(**kwargs)
    if layer.type == "AvgPool1d":
        return nn.AvgPool1d(**kwargs)
    if layer.type == "AvgPool2d":
        return nn.AvgPool2d(**kwargs)
    if layer.type == "GlobalAvgPool1d":
        return nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
    if layer.type == "GlobalAvgPool2d":
        return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
    if layer.type == "Flatten":
        return nn.Flatten()
    if layer.type == "Dropout":
        return nn.Dropout(**kwargs)
    if layer.type == "ReLU":
        return nn.ReLU()
    if layer.type == "ReLU6":
        return nn.ReLU6()
    if layer.type == "LeakyReLU":
        return nn.LeakyReLU(**kwargs)
    if layer.type == "Sigmoid":
        return nn.Sigmoid()
    if layer.type == "Tanh":
        return nn.Tanh()
    if layer.type == "Softmax":
        return nn.Softmax(**kwargs)

    raise ModelTranslationError(f"Unsupported torch layer type: {layer.type}")


def build_torch_model(model_cfg: ModelConfig, task_cfg: TaskConfig):
    """Translate unified layers into a torch model."""
    unsupported = [
        layer.type
        for layer in model_cfg.layers
        if layer.type not in SUPPORTED_TORCH_LAYERS
    ]
    if unsupported:
        raise ModelTranslationError(
            f"Unsupported torch layer types: {sorted(set(unsupported))}"
        )

    modules = [_build_torch_layer(layer) for layer in model_cfg.layers]
    return nn.Sequential(*modules)
