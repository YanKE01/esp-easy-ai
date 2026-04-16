"""PC-side model definition validation."""

from __future__ import annotations

import torch

from easyai.config.schema import Backend
from easyai.config.schema import EasyAIConfig
from easyai.model.errors import ModelDefinitionError
from easyai.model.translate import build_model


def _keras_input_shape(cfg: EasyAIConfig):
    shape = cfg.dataset.input_shape
    if cfg.task.input_dim == 1:
        channels, length = shape
        return (1, length, channels)
    channels, height, width = shape
    return (1, height, width, channels)


def _torch_input_shape(cfg: EasyAIConfig):
    return (1, *cfg.dataset.input_shape)


def validate_model_definition(cfg: EasyAIConfig):
    """Validate that a backend-native model can be defined and run on PC."""
    try:
        model = build_model(cfg)

        if cfg.backend == Backend.ESP_DL:
            model.eval()
            dummy = torch.zeros(_torch_input_shape(cfg), dtype=torch.float32)
            with torch.no_grad():
                model(dummy)
            return model

        if cfg.backend == Backend.TFLITE_MICRO:
            import tensorflow as tf

            dummy = tf.zeros(_keras_input_shape(cfg), dtype=tf.float32)
            model(dummy, training=False)
            return model
    except Exception as e:
        raise ModelDefinitionError(
            f"Model definition validation failed for backend={cfg.backend.value}: {e}"
        ) from e

    raise ModelDefinitionError(
        f"Unsupported backend for model validation: {cfg.backend}"
    )
