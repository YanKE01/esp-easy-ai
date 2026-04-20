"""Small white-user config completion helpers."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def _detect_delimiter(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return "," if "," in line else "whitespace"
    return ","


def _split_row(line: str, delimiter: str) -> list[str]:
    return line.split(",") if delimiter == "," else line.split()


def _first_non_empty_row(path: Path) -> list[str]:
    delimiter = _detect_delimiter(path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return _split_row(line, delimiter)
    raise ValueError(f"No data rows found in {path}")


def _infer_project_name(raw: dict) -> str:
    dataset_root = Path(raw["dataset"]["root"])
    return f"{dataset_root.name}_{raw['task']['type']}"


def _default_training(task_type: str, epochs: int) -> dict:
    if task_type == "classification":
        return {
            "epochs": epochs,
            "optimizer": {"type": "Adam", "lr": 1e-3, "weight_decay": 1e-4},
            "lr_scheduler": {"type": "CosineAnnealingLR", "T_max": epochs},
            "loss": "cross_entropy",
            "metrics": ["accuracy"],
            "checkpoint": {"save_best": True, "monitor": "val_accuracy"},
            "device": "auto",
        }

    return {
        "epochs": epochs,
        "optimizer": {"type": "Adam", "lr": 1e-3, "weight_decay": 1e-4},
        "lr_scheduler": {"type": "CosineAnnealingLR", "T_max": epochs},
        "loss": "mse",
        "metrics": ["mae", "rmse"],
        "checkpoint": {"save_best": True, "monitor": "val_mae"},
        "device": "auto",
    }


def _merge_training_defaults(training: dict, task_type: str) -> dict:
    epochs = training.get("epochs")
    if epochs is None:
        raise ValueError("training.epochs is required in small white-user mode")

    merged = _default_training(task_type, epochs)
    merged.update(training)

    if "optimizer" in training:
        merged["optimizer"] = {
            **_default_training(task_type, epochs)["optimizer"],
            **training["optimizer"],
        }
    if "lr_scheduler" in training and training["lr_scheduler"] is not None:
        merged["lr_scheduler"] = {
            **_default_training(task_type, epochs)["lr_scheduler"],
            **training["lr_scheduler"],
        }
    if "checkpoint" in training and training["checkpoint"] is not None:
        merged["checkpoint"] = {
            **_default_training(task_type, epochs)["checkpoint"],
            **training["checkpoint"],
        }

    return merged


def _default_quantization(_backend: str) -> dict:
    return {
        "enable": True,
        "method": "ptq",
        "dtype": "int8",
        "target": "esp32s3",
        "checkpoint": "best",
    }


def _merge_quantization_defaults(raw: dict) -> None:
    backend = raw["backend"]
    defaults = _default_quantization(backend)
    quantization = raw.get("quantization")
    if quantization is None:
        raw["quantization"] = defaults
        return

    merged = defaults.copy()
    merged.update(quantization)
    raw["quantization"] = merged


def _infer_windowed_per_class(dataset: dict) -> dict:
    root = Path(dataset["root"])
    train_dir = dataset.get("train_dir", "train")
    val_dir = dataset.get("val_dir", "test")
    train_path = root / train_dir
    sample_csv = next(iter(sorted(train_path.glob("*.csv"))))
    row = _first_non_empty_row(sample_csv)
    num_features = dataset["num_features"]
    if len(row) % num_features != 0:
        raise ValueError(
            f"Cannot infer input_shape: row has {len(row)} values but num_features={num_features}"
        )
    length = len(row) // num_features
    class_names = dataset.get("class_names") or sorted(
        p.stem for p in train_path.glob("*.csv")
    )
    return {
        "format": "windowed_per_class_csv",
        "train_dir": train_dir,
        "val_dir": val_dir,
        "feature_layout": dataset.get("feature_layout", "interleaved"),
        "has_header": dataset.get("has_header", False),
        "input_shape": dataset.get("input_shape", [num_features, length]),
        "num_classes": dataset.get("num_classes", len(class_names)),
        "class_names": class_names,
        "transforms": dataset.get("transforms", {"train": "none", "val": "none"}),
        "batch_size": dataset.get("batch_size", 64),
        "num_workers": dataset.get("num_workers", 4),
        "calibration": dataset.get(
            "calibration", {"source": "reuse_val", "num_samples": 256}
        ),
    }


def _infer_windowed_regression(dataset: dict) -> dict:
    root = Path(dataset["root"])
    train_dir = dataset.get("train_dir", "train")
    val_dir = dataset.get("val_dir", "test")
    train_path = root / train_dir
    sample_csv = next(iter(sorted(train_path.glob("*.csv"))))
    row = _first_non_empty_row(sample_csv)
    num_features = dataset["num_features"]
    feature_values = len(row) - 1
    if feature_values % num_features != 0:
        raise ValueError(
            f"Cannot infer input_shape: feature count {feature_values} is not divisible by num_features={num_features}"
        )
    length = feature_values // num_features
    return {
        "format": "windowed_regression_csv",
        "train_dir": train_dir,
        "val_dir": val_dir,
        "feature_layout": dataset.get("feature_layout", "interleaved"),
        "has_header": dataset.get("has_header", False),
        "label_start_col": dataset.get("label_start_col", 0),
        "label_end_col": dataset.get("label_end_col", 0),
        "input_shape": dataset.get("input_shape", [num_features, length]),
        "output_dim": dataset.get("output_dim", 1),
        "transforms": dataset.get("transforms", {"train": "none", "val": "none"}),
        "batch_size": dataset.get("batch_size", 64),
        "num_workers": dataset.get("num_workers", 4),
        "calibration": dataset.get(
            "calibration", {"source": "reuse_val", "num_samples": 256}
        ),
    }


def _infer_continuous_per_class(dataset: dict) -> dict:
    root = Path(dataset["root"])
    train_dir = dataset.get("train_dir", "train")
    val_dir = dataset.get("val_dir", "test")
    train_path = root / train_dir
    class_names = dataset.get("class_names") or sorted(
        p.name for p in train_path.iterdir() if p.is_dir()
    )
    window_size = dataset.get("window_size")
    window_stride = dataset.get("window_stride")
    if window_size is None or window_stride is None:
        raise ValueError(
            "dataset.window_size and dataset.window_stride are required for continuous_per_class_csv"
        )
    return {
        "format": "continuous_per_class_csv",
        "train_dir": train_dir,
        "val_dir": val_dir,
        "has_header": dataset.get("has_header", False),
        "time_column": dataset.get("time_column"),
        "feature_columns": dataset.get("feature_columns"),
        "window_size": window_size,
        "window_stride": window_stride,
        "input_shape": dataset.get(
            "input_shape", [dataset["num_features"], window_size]
        ),
        "num_classes": dataset.get("num_classes", len(class_names)),
        "class_names": class_names,
        "transforms": dataset.get("transforms", {"train": "none", "val": "none"}),
        "batch_size": dataset.get("batch_size", 64),
        "num_workers": dataset.get("num_workers", 4),
        "calibration": dataset.get(
            "calibration", {"source": "reuse_val", "num_samples": 256}
        ),
    }


def _apply_dataset_defaults(raw: dict) -> None:
    dataset = raw["dataset"]
    task_type = raw["task"]["type"]
    input_dim = raw["task"]["input_dim"]

    if input_dim != 1:
        return

    if "root" not in dataset or "num_features" not in dataset:
        raise ValueError(
            "dataset.root and dataset.num_features are required in small white-user mode"
        )

    fmt = dataset.get("format")
    if fmt == "continuous_per_class_csv":
        defaults = _infer_continuous_per_class(dataset)
    elif fmt == "windowed_per_class_csv":
        defaults = _infer_windowed_per_class(dataset)
    elif fmt == "windowed_regression_csv":
        defaults = _infer_windowed_regression(dataset)
    elif fmt is not None:
        return
    elif task_type == "classification":
        defaults = _infer_windowed_per_class(dataset)
    elif task_type == "regression":
        defaults = _infer_windowed_regression(dataset)
    else:
        return

    for key, value in defaults.items():
        dataset.setdefault(key, value)


def _tiny_conv_1d_layers(
    task_type: str, input_channels: int, output_dim: int
) -> list[dict]:
    if task_type == "classification":
        hidden = 16
        out_channels = 32
    else:
        hidden = 8
        out_channels = 16

    return [
        {
            "type": "Conv1d",
            "in_channels": input_channels,
            "out_channels": hidden,
            "kernel_size": 5,
            "padding": 2,
        },
        {"type": "ReLU"},
        {"type": "MaxPool1d", "kernel_size": 2},
        {
            "type": "Conv1d",
            "in_channels": hidden,
            "out_channels": out_channels,
            "kernel_size": 5,
            "padding": 2,
        },
        {"type": "ReLU"},
        {"type": "GlobalAvgPool1d"},
        {"type": "Linear", "in_features": out_channels, "out_features": output_dim},
    ]


def _tiny_mlp_1d_layers(input_shape: list[int], output_dim: int) -> list[dict]:
    flattened = input_shape[0] * input_shape[1]
    return [
        {"type": "Flatten"},
        {"type": "Linear", "in_features": flattened, "out_features": 32},
        {"type": "ReLU"},
        {"type": "Linear", "in_features": 32, "out_features": output_dim},
    ]


def _apply_model_defaults(raw: dict) -> None:
    model = raw.get("model")
    if not model or "layers" in model or "preset" not in model:
        return

    preset = model["preset"]
    task_type = raw["task"]["type"]
    ds = raw["dataset"]
    input_shape = ds["input_shape"]
    input_channels = input_shape[0]
    output_dim = (
        ds.get("num_classes", 1)
        if task_type == "classification"
        else ds.get("output_dim", 1)
    )

    if preset == "tiny_conv_1d":
        raw["model"] = {
            "layers": _tiny_conv_1d_layers(task_type, input_channels, output_dim)
        }
        return
    if preset == "tiny_mlp_1d":
        raw["model"] = {"layers": _tiny_mlp_1d_layers(input_shape, output_dim)}
        return

    raise ValueError(
        f"Unknown model preset '{preset}'. Supported presets: ['tiny_conv_1d', 'tiny_mlp_1d']"
    )


def complete_small_white_user_config(raw: dict) -> dict:
    completed = deepcopy(raw)
    completed.setdefault("project_name", _infer_project_name(completed))
    completed.setdefault("output_dir", f"./outputs/{completed['project_name']}")
    completed.setdefault("seed", 42)

    _apply_dataset_defaults(completed)

    if "training" in completed:
        completed["training"] = _merge_training_defaults(
            completed["training"], completed["task"]["type"]
        )

    _merge_quantization_defaults(completed)
    _apply_model_defaults(completed)
    return completed
