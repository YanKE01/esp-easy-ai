"""Minimal training entrypoint for the 1D pipeline."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from easyai.config.schema import Backend, EasyAIConfig, TaskType
from easyai.data import create_array_datasets, create_dataloaders
from easyai.model import build_model


def _select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    params = cfg.model_dump(exclude={"type"})
    if cfg.type == "Adam":
        return torch.optim.Adam(model.parameters(), **params)
    if cfg.type == "SGD":
        return torch.optim.SGD(model.parameters(), **params)
    raise NotImplementedError(f"Unsupported optimizer type: {cfg.type}")


def _build_scheduler(optimizer: torch.optim.Optimizer, cfg):
    if cfg is None:
        return None
    params = cfg.model_dump(exclude={"type"})
    if cfg.type == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    if cfg.type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    raise NotImplementedError(f"Unsupported lr_scheduler type: {cfg.type}")


def _build_loss(cfg):
    name = cfg.training.loss
    if name == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    if name == "mse":
        return torch.nn.MSELoss()
    if name == "l1":
        return torch.nn.L1Loss()
    if name == "huber":
        return torch.nn.HuberLoss()
    raise NotImplementedError(f"Unsupported loss type: {name}")


def _build_keras_optimizer(cfg):
    import tensorflow as tf

    params = cfg.model_dump(exclude={"type"})
    if "lr" in params:
        params["learning_rate"] = params.pop("lr")
    if cfg.type == "Adam":
        return tf.keras.optimizers.Adam(**params)
    if cfg.type == "SGD":
        return tf.keras.optimizers.SGD(**params)
    raise NotImplementedError(f"Unsupported optimizer type: {cfg.type}")


def _build_keras_loss(cfg):
    import tensorflow as tf

    name = cfg.training.loss
    if name == "cross_entropy":
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    if name == "mse":
        return tf.keras.losses.MeanSquaredError()
    if name == "l1":
        return tf.keras.losses.MeanAbsoluteError()
    if name == "huber":
        return tf.keras.losses.Huber()
    raise NotImplementedError(f"Unsupported loss type: {name}")


def _build_keras_metrics(cfg):
    import tensorflow as tf

    if cfg.task.type == TaskType.CLASSIFICATION:
        return [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    metrics = []
    for name in cfg.training.metrics:
        if name == "mae":
            metrics.append(tf.keras.metrics.MeanAbsoluteError(name="mae"))
        elif name == "rmse":
            metrics.append(tf.keras.metrics.RootMeanSquaredError(name="rmse"))
    return metrics


def _compute_metrics(
    task_type: TaskType, outputs: torch.Tensor, targets: torch.Tensor
) -> Dict[str, float]:
    if task_type == TaskType.CLASSIFICATION:
        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()
        return {"accuracy": accuracy}

    preds = outputs
    if preds.ndim > 1 and preds.shape[-1] == 1:
        preds = preds.squeeze(-1)
    if targets.ndim > 1 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    mae = F.l1_loss(preds, targets).item()
    rmse = math.sqrt(F.mse_loss(preds, targets).item())
    return {"mae": mae, "rmse": rmse}


def _run_epoch(model, loader, loss_fn, device, task_type, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    metric_totals: Dict[str, float] = {}
    batches = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        outputs = model(features)
        if task_type == TaskType.REGRESSION:
            outputs = outputs.squeeze(-1)
            targets = targets.float()

        loss = loss_fn(outputs, targets)

        if training:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        metrics = _compute_metrics(task_type, outputs.detach(), targets.detach())
        for name, value in metrics.items():
            metric_totals[name] = metric_totals.get(name, 0.0) + value
        batches += 1

    result = {"loss": total_loss / max(batches, 1)}
    for name, total in metric_totals.items():
        result[name] = total / max(batches, 1)
    return result


def _monitor_mode(name: str) -> str:
    lowered = name.lower()
    if any(key in lowered for key in ("loss", "mae", "rmse")):
        return "min"
    return "max"


def _is_improved(candidate: float, best: float | None, mode: str) -> bool:
    if best is None:
        return True
    return candidate < best if mode == "min" else candidate > best


class _EpochPrinter:
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        metrics_preview = ", ".join(f"{k}={float(v):.4f}" for k, v in logs.items())
        print(f"epoch {epoch + 1}/{self.total_epochs}: {metrics_preview}")


def _run_torch_training(cfg: EasyAIConfig):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device(cfg.training.device)
    train_loader, val_loader = create_dataloaders(cfg)
    model = build_model(cfg).to(device)
    optimizer = _build_optimizer(model, cfg.training.optimizer)
    scheduler = _build_scheduler(optimizer, cfg.training.lr_scheduler)
    loss_fn = _build_loss(cfg)

    monitor = cfg.training.checkpoint.monitor if cfg.training.checkpoint else "val_loss"
    mode = _monitor_mode(monitor)
    best_value = None
    best_epoch = 0
    early_stopping = cfg.training.early_stopping
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = _run_epoch(
            model, train_loader, loss_fn, device, cfg.task.type, optimizer
        )
        val_metrics = _run_epoch(model, val_loader, loss_fn, device, cfg.task.type)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        for name, value in train_metrics.items():
            if name != "loss":
                epoch_metrics[f"train_{name}"] = value
        for name, value in val_metrics.items():
            if name != "loss":
                epoch_metrics[f"val_{name}"] = value
        history.append(epoch_metrics)
        torch.save(model.state_dict(), output_dir / "last_model.pt")

        monitor_value = epoch_metrics[monitor]
        improved = _is_improved(monitor_value, best_value, mode)
        if improved:
            best_value = monitor_value
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()

        metrics_preview = ", ".join(
            f"{k}={v:.4f}" for k, v in epoch_metrics.items() if k != "epoch"
        )
        print(f"epoch {epoch}/{cfg.training.epochs}: {metrics_preview}")

        if early_stopping and patience_counter >= early_stopping.patience:
            print(f"early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

    report = {
        "project_name": cfg.project_name,
        "backend": cfg.backend.value,
        "task": cfg.task.type.value,
        "device": str(device),
        "best_epoch": best_epoch,
        "monitor": monitor,
        "best_value": best_value,
        "history": history,
        "best_checkpoint": str(output_dir / "best_model.pt"),
        "last_checkpoint": str(output_dir / "last_model.pt"),
        "checkpoint": str(output_dir / "best_model.pt"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _run_keras_training(cfg: EasyAIConfig):
    import tensorflow as tf

    if cfg.task.input_dim != 1:
        raise NotImplementedError(
            "tflite-micro training is currently implemented only for 1D tasks."
        )

    if cfg.seed is not None:
        tf.keras.utils.set_random_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_x, train_y, val_x, val_y = create_array_datasets(cfg)
    model = build_model(cfg)
    optimizer = _build_keras_optimizer(cfg.training.optimizer)
    loss = _build_keras_loss(cfg)
    metrics = _build_keras_metrics(cfg)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    monitor = cfg.training.checkpoint.monitor if cfg.training.checkpoint else "val_loss"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor=monitor,
            save_best_only=True,
            mode=_monitor_mode(monitor),
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "last_model.keras"),
            save_best_only=False,
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=_EpochPrinter(cfg.training.epochs).on_epoch_end
        ),
    ]
    if cfg.training.early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=_monitor_mode(cfg.training.early_stopping.monitor),
                restore_best_weights=True,
            )
        )

    fit_history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=cfg.training.epochs,
        batch_size=cfg.dataset.batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    history = []
    epochs_ran = len(fit_history.history["loss"])
    monitor_values = fit_history.history.get(monitor, [])
    mode = _monitor_mode(monitor)
    best_epoch = 0
    best_value = None
    for epoch_idx in range(epochs_ran):
        epoch_metrics = {"epoch": epoch_idx + 1}
        for name, values in fit_history.history.items():
            epoch_metrics[name] = float(values[epoch_idx])
        history.append(epoch_metrics)
        if monitor_values:
            candidate = float(monitor_values[epoch_idx])
            if _is_improved(candidate, best_value, mode):
                best_value = candidate
                best_epoch = epoch_idx + 1

    if best_value is None and history:
        best_epoch = len(history)
        best_value = history[-1].get(monitor)

    report = {
        "project_name": cfg.project_name,
        "backend": cfg.backend.value,
        "task": cfg.task.type.value,
        "device": "tensorflow",
        "best_epoch": best_epoch,
        "monitor": monitor,
        "best_value": best_value,
        "history": history,
        "best_checkpoint": str(output_dir / "best_model.keras"),
        "last_checkpoint": str(output_dir / "last_model.keras"),
        "checkpoint": str(output_dir / "best_model.keras"),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def run_training(cfg: EasyAIConfig):
    if cfg.model is None or cfg.training is None:
        raise ValueError("Training requires model and training config.")
    if cfg.backend == Backend.ESP_DL:
        return _run_torch_training(cfg)
    if cfg.backend == Backend.TFLITE_MICRO:
        return _run_keras_training(cfg)
    raise NotImplementedError(
        f"Training is not implemented for backend={cfg.backend.value}."
    )
