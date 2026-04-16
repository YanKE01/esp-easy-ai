"""Runtime dataset transforms for the minimal 1D pipeline."""

from __future__ import annotations

from typing import Callable, Iterable

import torch


def _as_channel_tensor(value, channels: int) -> torch.Tensor:
    if isinstance(value, list):
        tensor = torch.tensor(value, dtype=torch.float32)
    else:
        tensor = torch.tensor([value], dtype=torch.float32)
    if tensor.numel() == 1:
        tensor = tensor.repeat(channels)
    return tensor.view(channels, *([1]))


def compile_transforms(
    items: Iterable, normalize_stats: dict | None = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    transforms = list(items or [])

    def apply(x: torch.Tensor) -> torch.Tensor:
        out = x
        for item in transforms:
            t = item.type
            if t == "Identity":
                continue
            if t == "Rescale":
                out = out * float(item.model_dump().get("scale", 1.0))
                continue
            if t == "Normalize":
                cfg = item.model_dump()
                channels = out.shape[0]
                mean_value = cfg.get("mean")
                std_value = cfg.get("std")
                if mean_value is None or std_value is None:
                    if normalize_stats is None:
                        raise ValueError(
                            "Normalize requires train-set statistics, but none were provided."
                        )
                    mean_value = normalize_stats["mean"]
                    std_value = normalize_stats["std"]
                mean = _as_channel_tensor(mean_value, channels).to(out.dtype)
                std = _as_channel_tensor(std_value, channels).to(out.dtype)
                out = (out - mean) / std
                continue
            raise ValueError(f"Unsupported runtime transform: {t}")
        return out

    return apply
