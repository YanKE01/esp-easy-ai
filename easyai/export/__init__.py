"""Backend-specific export entrypoints."""

from __future__ import annotations

from easyai.config.schema import Backend, EasyAIConfig

from . import esp_dl, tflite_micro


def run_export(cfg: EasyAIConfig):
    if cfg.backend == Backend.ESP_DL:
        return esp_dl.run_export(cfg)
    if cfg.backend == Backend.TFLITE_MICRO:
        return tflite_micro.run_export(cfg)
    raise NotImplementedError(
        f"Export is not implemented for backend={cfg.backend.value}."
    )


__all__ = ["run_export"]
