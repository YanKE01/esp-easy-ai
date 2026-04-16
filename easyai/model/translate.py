"""Unified model translation entrypoint."""

from easyai.config.schema import Backend, EasyAIConfig
from easyai.model.backends.keras_builder import build_keras_model
from easyai.model.backends.torch_builder import build_torch_model


def build_model(cfg: EasyAIConfig):
    """Build a backend-native model from the unified config."""
    if cfg.model is None:
        raise ValueError("Model config is required to build a model.")

    if cfg.backend == Backend.ESP_DL:
        return build_torch_model(cfg.model, cfg.task)
    if cfg.backend == Backend.TFLITE_MICRO:
        return build_keras_model(cfg.model, cfg.task)

    raise ValueError(f"Unsupported backend: {cfg.backend}")
