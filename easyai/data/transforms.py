"""Runtime dataset transforms for the current 1D pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch


def _as_channel_tensor(value, channels: int, rank: int) -> torch.Tensor:
    if isinstance(value, list):
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        tensor = torch.tensor([value], dtype=torch.float32)
    if tensor.numel() == 1:
        tensor = tensor.repeat(channels)
    if tensor.numel() != channels:
        raise ValueError(
            f"Normalize expects 1 value or {channels} channel values, got {tensor.numel()}."
        )
    if rank < 1:
        raise ValueError(f"Normalize expects tensor rank >= 1, got {rank}.")
    return tensor.view(channels, *([1] * (rank - 1)))


@dataclass
class RuntimeTransformPipeline:
    transforms: list
    normalize_stats: dict | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for item in self.transforms:
            t = item.type
            if t == "Identity":
                continue
            if t == "Rescale":
                out = out * float(item.model_dump().get("scale", 1.0))
                continue
            if t == "Normalize":
                cfg = item.model_dump()
                channels = out.shape[0]
                rank = out.ndim
                mean_value = cfg.get("mean")
                std_value = cfg.get("std")
                if mean_value is None or std_value is None:
                    if self.normalize_stats is None:
                        raise ValueError(
                            "Normalize requires train-set statistics, but none were provided."
                        )
                    mean_value = self.normalize_stats["mean"]
                    std_value = self.normalize_stats["std"]
                mean = _as_channel_tensor(mean_value, channels, rank).to(out.dtype)
                std = _as_channel_tensor(std_value, channels, rank).to(out.dtype)
                out = (out - mean) / std
                continue
            raise ValueError(f"Unsupported runtime transform: {t}")
        return out


def compile_transforms(
    items: Iterable, normalize_stats: dict | None = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    return RuntimeTransformPipeline(
        transforms=list(items or []),
        normalize_stats=normalize_stats,
    )
