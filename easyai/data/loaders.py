"""Minimal dataset loaders for 1D training."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from easyai.config.schema import (
    CsvPerClassDataset,
    CsvRegressionDataset,
    DatasetFormat,
    EasyAIConfig,
    FeatureLayout,
    TaskType,
)
from easyai.data.transforms import compile_transforms


def _detect_delimiter(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return "," if "," in line else "whitespace"
    return ","


def _split_row(line: str, delimiter: str) -> List[str]:
    line = line.strip()
    if delimiter in ("whitespace", " "):
        return line.split()
    return line.split(delimiter)


def _read_rows(
    path: Path, delimiter: str | None, has_header: bool
) -> List[List[float]]:
    effective_delimiter = delimiter or _detect_delimiter(path)
    rows: List[List[float]] = []
    with open(path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0 and has_header:
                continue
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in _split_row(line, effective_delimiter)])
    return rows


def _reshape_features(
    flat: Sequence[float], num_features: int, length: int, feature_layout: FeatureLayout
) -> torch.Tensor:
    tensor = torch.tensor(flat, dtype=torch.float32)
    if feature_layout == FeatureLayout.INTERLEAVED:
        return tensor.view(length, num_features).transpose(0, 1).contiguous()
    return tensor.view(num_features, length)


class _TensorDataset(Dataset):
    def __init__(
        self,
        features: List[torch.Tensor],
        targets: List[torch.Tensor],
        transform: Callable[[torch.Tensor], torch.Tensor],
    ):
        self.features = features
        self.targets = targets
        self.transform = transform

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.transform(self.features[idx].clone())
        y = self.targets[idx]
        return x, y


def _uses_normalize(items) -> bool:
    return any(getattr(item, "type", None) == "Normalize" for item in (items or []))


def _compute_normalize_stats(features: List[torch.Tensor]) -> dict:
    stacked = torch.stack(features)
    dims = tuple(i for i in range(stacked.ndim) if i != 1)
    mean = stacked.mean(dim=dims)
    std = stacked.std(dim=dims, unbiased=False).clamp_min(1e-6)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def _build_split_datasets(cfg: EasyAIConfig):
    ds = cfg.dataset
    transforms_cfg = ds.transforms
    root = Path(ds.root)

    if ds.format == DatasetFormat.CSV_PER_CLASS:
        train_features, train_targets = _load_csv_per_class_split(
            ds, root / ds.train_dir
        )
        val_features, val_targets = _load_csv_per_class_split(ds, root / ds.val_dir)
    elif ds.format == DatasetFormat.CSV_REGRESSION:
        train_features, train_targets = _load_csv_regression_split(
            ds, root / ds.train_dir
        )
        val_features, val_targets = _load_csv_regression_split(ds, root / ds.val_dir)
    else:
        raise NotImplementedError(
            f"Training data loader not implemented for dataset format: {ds.format}"
        )

    train_items = getattr(transforms_cfg, "train", [])
    val_items = getattr(transforms_cfg, "val", [])
    normalize_stats = None
    if _uses_normalize(train_items) or _uses_normalize(val_items):
        normalize_stats = _compute_normalize_stats(train_features)

    train_transform = compile_transforms(train_items, normalize_stats=normalize_stats)
    val_transform = compile_transforms(val_items, normalize_stats=normalize_stats)
    train_dataset = _TensorDataset(train_features, train_targets, train_transform)
    val_dataset = _TensorDataset(val_features, val_targets, val_transform)
    return train_dataset, val_dataset


def _select_calibration_dataset(cfg: EasyAIConfig):
    train_dataset, val_dataset = _build_split_datasets(cfg)
    calibration_cfg = cfg.dataset.calibration
    source = getattr(calibration_cfg, "source", "reuse_val")
    dataset = val_dataset if source == "reuse_val" else train_dataset
    num_samples = min(
        getattr(calibration_cfg, "num_samples", len(dataset)), len(dataset)
    )
    return dataset, num_samples


def _load_csv_per_class_split(ds: CsvPerClassDataset, split_dir: Path):
    length = ds.input_shape[1]
    class_names = ds.class_names or sorted(p.stem for p in split_dir.glob("*.csv"))
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for csv_path in sorted(split_dir.glob("*.csv")):
        class_id = class_to_id[csv_path.stem]
        for row in _read_rows(csv_path, ds.delimiter, ds.has_header):
            feature = _reshape_features(row, ds.num_features, length, ds.feature_layout)
            features.append(feature)
            targets.append(torch.tensor(class_id, dtype=torch.long))

    return features, targets


def _resolve_label_slice(row_len: int, start: int, end: int) -> Tuple[int, int]:
    start_idx = start if start >= 0 else row_len + start
    end_idx = end if end >= 0 else row_len + end
    return start_idx, end_idx


def _load_csv_regression_split(ds: CsvRegressionDataset, split_dir: Path):
    length = ds.input_shape[1]
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for csv_path in sorted(split_dir.glob("*.csv")):
        for row in _read_rows(csv_path, ds.delimiter, ds.has_header):
            start_idx, end_idx = _resolve_label_slice(
                len(row), ds.label_start_col, ds.label_end_col
            )
            label_values = row[start_idx : end_idx + 1]
            feature_values = row[:start_idx] + row[end_idx + 1 :]
            feature = _reshape_features(
                feature_values, ds.num_features, length, ds.feature_layout
            )
            target = torch.tensor(label_values, dtype=torch.float32)
            if target.numel() == 1:
                target = target.squeeze(0)
            features.append(feature)
            targets.append(target)

    return features, targets


def create_dataloaders(cfg: EasyAIConfig) -> Tuple[DataLoader, DataLoader]:
    ds = cfg.dataset
    train_dataset, val_dataset = _build_split_datasets(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=ds.batch_size,
        shuffle=True,
        num_workers=ds.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=ds.batch_size,
        shuffle=False,
        num_workers=ds.num_workers,
    )
    return train_loader, val_loader


def create_calibration_dataloader(cfg: EasyAIConfig) -> DataLoader:
    dataset, num_samples = _select_calibration_dataset(cfg)
    subset = torch.utils.data.Subset(dataset, range(num_samples))
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
    )


def _dataset_to_numpy(dataset: Dataset, task_type: TaskType, input_dim: int):
    features = []
    targets = []
    for idx in range(len(dataset)):
        x, y = dataset[idx]
        if input_dim == 1:
            x = x.transpose(0, 1).contiguous()
        else:
            x = x.permute(1, 2, 0).contiguous()
        features.append(x.numpy())
        if isinstance(y, torch.Tensor):
            targets.append(y.numpy())
        else:
            targets.append(y)

    x_array = np.stack(features)
    y_array = (
        np.stack(targets) if task_type == TaskType.REGRESSION else np.array(targets)
    )
    return x_array, y_array


def create_array_datasets(cfg: EasyAIConfig):
    train_dataset, val_dataset = _build_split_datasets(cfg)
    train_x, train_y = _dataset_to_numpy(
        train_dataset, cfg.task.type, cfg.task.input_dim
    )
    val_x, val_y = _dataset_to_numpy(val_dataset, cfg.task.type, cfg.task.input_dim)
    return train_x, train_y, val_x, val_y


def create_calibration_arrays(cfg: EasyAIConfig):
    dataset, num_samples = _select_calibration_dataset(cfg)
    features = []
    for idx in range(num_samples):
        x, _ = dataset[idx]
        if cfg.task.input_dim == 1:
            x = x.transpose(0, 1).contiguous()
        else:
            x = x.permute(1, 2, 0).contiguous()
        features.append(x.numpy())
    return np.stack(features)
