"""Filesystem-level dataset validation for esp-easy-ai."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from easyai.config.schema import (
    ContinuousPerClassDataset,
    EasyAIConfig,
    WindowedPerClassDataset,
    WindowedRegressionDataset,
)


class DatasetCheckError(Exception):
    pass


def _resolve(base: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _detect_delimiter(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return "," if "," in line else "whitespace"
    return ","


def _read_first_row(path: Path, delimiter: str | None, has_header: bool):
    effective_delimiter = delimiter or _detect_delimiter(path)
    with open(path, encoding="utf-8") as f:
        if effective_delimiter in ("whitespace", " "):
            rows = (line.strip().split() for line in f)
        else:
            rows = csv.reader(f, delimiter=effective_delimiter)

        if has_header:
            next(rows, None)
        return next(rows, None)


def _check_windowed_per_class(
    ds: WindowedPerClassDataset, root: Path, errors: List[str]
):
    train_path = root / ds.train_dir
    val_path = root / ds.val_dir

    if not train_path.exists():
        errors.append(f"train_dir not found: {train_path}")
    if not val_path.exists():
        errors.append(f"val_dir not found: {val_path}")
    if errors:
        return

    # Check class_names match actual subdirectory/file names
    actual_classes = sorted(
        p.stem if p.is_file() else p.name
        for p in train_path.iterdir()
        if p.suffix in (".csv", "") or p.is_dir()
    )
    # Only CSV files count
    actual_classes = sorted(p.stem for p in train_path.iterdir() if p.suffix == ".csv")
    if not actual_classes:
        errors.append(f"No CSV files found in train_dir: {train_path}")
        return

    if ds.class_names:
        missing = [c for c in ds.class_names if c not in actual_classes]
        extra = [c for c in actual_classes if c not in ds.class_names]
        if missing:
            errors.append(
                f"class_names {missing} not found as CSV files in {train_path}. "
                f"Available: {actual_classes}"
            )
        if extra:
            errors.append(
                f"CSV files {extra} in {train_path} are not listed in class_names"
            )

    # Check input_shape matches actual data
    sample_csv = next(train_path.glob("*.csv"), None)
    if sample_csv:
        first_row = _read_first_row(sample_csv, ds.delimiter, ds.has_header)
        if first_row is not None:
            n_cols = len(first_row)
            expected_cols = ds.num_features * ds.input_shape[1]
            if n_cols != expected_cols:
                errors.append(
                    f"CSV row has {n_cols} columns but "
                    f"num_features={ds.num_features} × input_shape[1]={ds.input_shape[1]} "
                    f"= {expected_cols}"
                )


def _check_windowed_regression(
    ds: WindowedRegressionDataset, root: Path, errors: List[str]
):
    train_path = root / ds.train_dir
    val_path = root / ds.val_dir

    if not train_path.exists():
        errors.append(f"train_dir not found: {train_path}")
    if not val_path.exists():
        errors.append(f"val_dir not found: {val_path}")
    if errors:
        return

    sample_csv = next(train_path.glob("*.csv"), None)
    if sample_csv:
        first_row = _read_first_row(sample_csv, ds.delimiter, ds.has_header)
        if first_row is not None:
            n_cols = len(first_row)
            expected_cols = ds.num_features * ds.input_shape[1]
            if n_cols <= expected_cols:
                errors.append(
                    f"CSV row has {n_cols} columns; expected at least "
                    f"num_features × L + label columns = {expected_cols + 1}"
                )


def _check_continuous_per_class(
    ds: ContinuousPerClassDataset, root: Path, errors: List[str]
):
    train_path = root / ds.train_dir
    val_path = root / ds.val_dir

    if not train_path.exists():
        errors.append(f"train_dir not found: {train_path}")
    if not val_path.exists():
        errors.append(f"val_dir not found: {val_path}")
    if errors:
        return

    actual_classes = sorted(p.name for p in train_path.iterdir() if p.is_dir())
    if not actual_classes:
        errors.append(f"No class directories found in train_dir: {train_path}")
        return

    if ds.class_names:
        missing = [c for c in ds.class_names if c not in actual_classes]
        extra = [c for c in actual_classes if c not in ds.class_names]
        if missing:
            errors.append(
                f"class_names {missing} not found as subdirs in {train_path}. "
                f"Available: {actual_classes}"
            )
        if extra:
            errors.append(
                f"Subdirs {extra} in {train_path} are not listed in class_names"
            )

    sample_csv = next(train_path.glob("*/*.csv"), None)
    if sample_csv is None:
        errors.append(f"No CSV files found under class directories in {train_path}")
        return

    rows = []
    effective_delimiter = ds.delimiter or _detect_delimiter(sample_csv)
    with open(sample_csv, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if idx == 0 and ds.has_header:
                continue
            rows.append(
                line.split()
                if effective_delimiter in ("whitespace", " ")
                else line.split(effective_delimiter)
            )
    if len(rows) < ds.window_size:
        errors.append(
            f"Sample CSV {sample_csv} has {len(rows)} rows but window_size={ds.window_size}"
        )
        return
    header = None
    if ds.has_header:
        effective_delimiter = ds.delimiter or _detect_delimiter(sample_csv)
        with open(sample_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                header = (
                    line.split()
                    if effective_delimiter in ("whitespace", " ")
                    else line.split(effective_delimiter)
                )
                break
    row_width = len(rows[0])
    if ds.feature_columns:
        if not header:
            errors.append("feature_columns requires has_header=true")
            return
        missing = [name for name in ds.feature_columns if name not in header]
        if missing:
            errors.append(f"feature_columns {missing} not found in header {header}")
        if len(ds.feature_columns) != ds.num_features:
            errors.append(
                f"feature_columns has {len(ds.feature_columns)} entries but num_features={ds.num_features}"
            )
    else:
        expected = ds.num_features + (1 if ds.time_column is not None else 0)
        if row_width != expected:
            errors.append(
                f"Continuous CSV row has {row_width} columns but expected {expected} "
                f"(num_features={ds.num_features}, time_column={ds.time_column})"
            )


def validate_dataset_files(cfg: EasyAIConfig, config_path: str) -> List[str]:
    """Check that dataset files/directories exist and match the config.

    Returns a list of error strings (empty = all good).
    All paths in config are resolved relative to CWD.
    """
    root = Path(cfg.dataset.root).resolve()

    if not root.exists():
        return [f"dataset root not found: {root}"]

    errors: List[str] = []
    ds = cfg.dataset

    if isinstance(ds, WindowedPerClassDataset):
        _check_windowed_per_class(ds, root, errors)
    elif isinstance(ds, WindowedRegressionDataset):
        _check_windowed_regression(ds, root, errors)
    elif isinstance(ds, ContinuousPerClassDataset):
        _check_continuous_per_class(ds, root, errors)

    return errors
