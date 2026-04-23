"""Reference sample loading for export."""

from __future__ import annotations

from pathlib import Path

from easyai.config.schema import (
    ContinuousPerClassDataset,
    EasyAIConfig,
    WindowedPerClassDataset,
    WindowedRegressionDataset,
)
from easyai.data.loaders import (
    _build_calibration_indices,
    _compute_normalize_stats,
    _load_continuous_per_class_split,
    _load_windowed_per_class_split,
    _load_windowed_regression_split,
    _select_calibration_dataset,
)


def load_raw_split(cfg: EasyAIConfig, split_name: str):
    ds = cfg.dataset
    root = Path(ds.root)
    split_dir = root / (ds.train_dir if split_name == "train" else ds.val_dir)
    if isinstance(ds, WindowedPerClassDataset):
        return _load_windowed_per_class_split(ds, split_dir)
    if isinstance(ds, WindowedRegressionDataset):
        return _load_windowed_regression_split(ds, split_dir)
    if isinstance(ds, ContinuousPerClassDataset):
        return _load_continuous_per_class_split(ds, split_dir)
    raise NotImplementedError(
        f"Export is not implemented for dataset format: {ds.format}"
    )


def flatten_raw_input(feature) -> list[float]:
    tensor = feature.detach().cpu()
    return tensor.reshape(-1).tolist()


def to_jsonable_target(target):
    if hasattr(target, "ndim") and getattr(target, "ndim", 0) == 0:
        return target.item()
    if hasattr(target, "tolist"):
        return target.tolist()
    return target


def extract_preprocess(cfg: EasyAIConfig, source: str, raw_train_features) -> dict:
    ds = cfg.dataset
    items = ds.transforms.train if source == "reuse_train" else ds.transforms.val
    rescale = 1.0
    for item in items or []:
        if item.type == "Identity":
            continue
        if item.type == "Rescale":
            rescale *= float(item.model_dump().get("scale", 1.0))
            continue
        if item.type == "Normalize":
            cfg_item = item.model_dump()
            mean_value = cfg_item.get("mean")
            std_value = cfg_item.get("std")
            if mean_value is not None and std_value is not None:
                return {
                    "rescale": rescale,
                    "mean": mean_value,
                    "std": std_value,
                }
            stats = _compute_normalize_stats(raw_train_features)
            return {
                "rescale": rescale,
                "mean": stats["mean"],
                "std": stats["std"],
            }
        raise ValueError(f"Unsupported export preprocess transform: {item.type}")

    channels = cfg.dataset.input_shape[0]
    return {
        "rescale": rescale,
        "mean": [0.0] * channels,
        "std": [1.0] * channels,
    }


def resolve_reference_sample(
    cfg: EasyAIConfig, calibration_source: str, subset_index: int
):
    split_name = "val" if calibration_source == "reuse_val" else "train"
    raw_features, raw_targets = load_raw_split(cfg, split_name)
    dataset, num_samples = _select_calibration_dataset(cfg)
    indices = _build_calibration_indices(dataset, num_samples, cfg.task.type, cfg.seed)
    if not indices:
        raw_index = 0
    else:
        subset_index = max(0, min(subset_index, len(indices) - 1))
        raw_index = indices[subset_index]
    return split_name, raw_index, raw_features[raw_index], raw_targets[raw_index]
