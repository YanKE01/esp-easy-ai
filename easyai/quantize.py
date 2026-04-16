"""Minimal PTQ INT8 quantization entrypoint."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from easyai.config.schema import Backend, EasyAIConfig, TaskType
from easyai.data import (
    create_array_datasets,
    create_calibration_arrays,
    create_calibration_dataloader,
)
from easyai.model import build_model


def _resolve_checkpoint_source(cfg: EasyAIConfig) -> str:
    quant_cfg = cfg.quantization
    source = getattr(quant_cfg, "checkpoint", None) if quant_cfg else None
    return source or "best"


def _load_checkpoint_path(cfg: EasyAIConfig) -> Path:
    output_dir = Path(cfg.output_dir)
    checkpoint_source = _resolve_checkpoint_source(cfg)
    if cfg.backend == Backend.TFLITE_MICRO:
        filename = (
            "best_model.keras" if checkpoint_source == "best" else "last_model.keras"
        )
    else:
        filename = "best_model.pt" if checkpoint_source == "best" else "last_model.pt"
    path = output_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for quantization: {path}. Run training first."
        )
    return path


def _quantize_array(
    array: np.ndarray, scale: float, zero_point: int, dtype
) -> np.ndarray:
    if scale == 0:
        return array.astype(dtype)
    info = np.iinfo(dtype)
    quantized = np.round(array / scale + zero_point)
    quantized = np.clip(quantized, info.min, info.max)
    return quantized.astype(dtype)


def _dequantize_array(array: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale == 0:
        return array.astype(np.float32)
    return (array.astype(np.float32) - zero_point) * scale


def _evaluate_tflite_model(cfg: EasyAIConfig, model_path: Path) -> dict:
    import tensorflow as tf

    _, _, val_x, val_y = create_array_datasets(cfg)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    predictions = []
    for sample in val_x:
        batched = sample[None, ...].astype(np.float32)
        if input_details["dtype"] != np.float32:
            batched = _quantize_array(
                batched,
                input_scale,
                input_zero_point,
                input_details["dtype"],
            )
        interpreter.set_tensor(input_details["index"], batched)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] != np.float32:
            output = _dequantize_array(output, output_scale, output_zero_point)
        predictions.append(output[0])

    predictions = np.asarray(predictions)
    report = {
        "eval_source": "val",
        "eval_num_samples": int(len(val_y)),
    }

    sample_count = min(10, len(val_y))
    rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 42)
    sample_indices = (
        rng.choice(len(val_y), size=sample_count, replace=False).tolist()
        if sample_count > 0
        else []
    )

    if cfg.task.type == TaskType.CLASSIFICATION:
        pred_labels = predictions.argmax(axis=1)
        accuracy = float((pred_labels == val_y).mean())
        report["eval_accuracy"] = accuracy
        report["eval_examples"] = [
            {
                "index": int(idx),
                "pred": int(pred_labels[idx]),
                "target": int(val_y[idx]),
            }
            for idx in sample_indices
        ]
        return report

    preds = predictions.reshape(len(predictions), -1)
    targets = np.asarray(val_y).reshape(len(val_y), -1)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(math.sqrt(np.mean(np.square(preds - targets))))
    report["eval_mae"] = mae
    report["eval_rmse"] = rmse
    report["eval_examples"] = [
        {
            "index": int(idx),
            "pred": float(preds[idx].reshape(-1)[0]),
            "target": float(targets[idx].reshape(-1)[0]),
        }
        for idx in sample_indices
    ]
    return report


def _run_tflite_ptq_int8(cfg: EasyAIConfig):
    import tensorflow as tf

    checkpoint_path = _load_checkpoint_path(cfg)
    model = build_model(cfg)
    input_shape = (1, cfg.dataset.input_shape[1], cfg.dataset.input_shape[0])
    dummy = tf.zeros(input_shape, dtype=tf.float32)
    model(dummy, training=False)
    model.load_weights(checkpoint_path)
    calibration_x = create_calibration_arrays(cfg)

    @tf.function
    def serve(x):
        return model(x, training=False)

    concrete = serve.get_concrete_function(tf.TensorSpec(input_shape, tf.float32))

    def representative_dataset():
        for sample in calibration_x:
            yield [sample[None, ...].astype("float32")]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_int8.tflite"
    model_path.write_bytes(tflite_model)
    eval_report = _evaluate_tflite_model(cfg, model_path)

    report = {
        "backend": cfg.backend.value,
        "method": "ptq",
        "dtype": "int8",
        "target": getattr(cfg.quantization, "target", None)
        if cfg.quantization
        else None,
        "checkpoint_source": _resolve_checkpoint_source(cfg),
        "checkpoint": str(checkpoint_path),
        "calibration_source": cfg.dataset.calibration.source
        if cfg.dataset.calibration
        else "reuse_val",
        "calibration_num_samples": len(calibration_x),
        "calibration_shuffle": False,
        "output": str(model_path),
    }
    report.update(eval_report)
    (output_dir / "quantization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def _resolve_quant_target(cfg: EasyAIConfig) -> str:
    quant_cfg = cfg.quantization
    target = getattr(quant_cfg, "target", None) if quant_cfg else None
    return target or "esp32s3"


def _to_jsonable_target(target):
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None and isinstance(target, torch.Tensor):
        if target.ndim == 0:
            return target.item()
        return target.detach().cpu().tolist()
    if hasattr(target, "tolist"):
        return target.tolist()
    return target


def _run_espdl_ptq_int8(cfg: EasyAIConfig):
    import torch
    from esp_ppq.api import espdl_quantize_torch

    checkpoint_path = _load_checkpoint_path(cfg)
    model = build_model(cfg)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    calibration_loader = create_calibration_dataloader(cfg)
    calibration_size = len(calibration_loader.dataset)
    target = _resolve_quant_target(cfg)
    calibration_steps = calibration_size
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model_int8.espdl"
    sample_input, sample_target = calibration_loader.dataset[0]
    quant_input = sample_input.unsqueeze(0).to("cpu")
    quant_inputs = [quant_input]

    def collate_fn(batch):
        return batch[0].to("cpu")

    espdl_quantize_torch(
        model=model,
        espdl_export_file=str(model_path),
        calib_dataloader=calibration_loader,
        calib_steps=calibration_steps,
        input_shape=[1] + cfg.dataset.input_shape,
        inputs=quant_inputs,
        target=target,
        num_of_bits=8,
        collate_fn=collate_fn,
        device="cpu",
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,
    )

    report = {
        "backend": cfg.backend.value,
        "method": "ptq",
        "dtype": "int8",
        "target": target,
        "checkpoint_source": _resolve_checkpoint_source(cfg),
        "checkpoint": str(checkpoint_path),
        "calibration_source": cfg.dataset.calibration.source
        if cfg.dataset.calibration
        else "reuse_val",
        "calibration_num_samples": calibration_size,
        "calibration_steps": calibration_steps,
        "calibration_shuffle": False,
        "test_input_shape": list(quant_input.shape),
        "test_input_target": _to_jsonable_target(sample_target),
        "output": str(model_path),
    }
    (output_dir / "quantization_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def run_quantization(cfg: EasyAIConfig):
    if cfg.backend == Backend.TFLITE_MICRO:
        return _run_tflite_ptq_int8(cfg)
    if cfg.backend == Backend.ESP_DL:
        return _run_espdl_ptq_int8(cfg)
    raise NotImplementedError(
        f"Quantization is not implemented for backend={cfg.backend.value}."
    )
