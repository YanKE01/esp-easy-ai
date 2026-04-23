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
    create_dataloaders,
)
from easyai.model import build_model


def _choose_reference_index(targets, task_type: TaskType) -> int:
    sample_count = len(targets)
    if sample_count == 0:
        return 0

    rng = np.random.default_rng()
    if task_type == TaskType.CLASSIFICATION:
        label_to_indices = {}
        for idx, target in enumerate(targets):
            label = int(_to_jsonable_target(target))
            label_to_indices.setdefault(label, []).append(idx)
        chosen_label = int(rng.choice(sorted(label_to_indices)))
        return int(rng.choice(label_to_indices[chosen_label]))
    return int(rng.integers(0, sample_count))


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


def _to_jsonable_output(array: np.ndarray):
    flat = np.asarray(array, dtype=np.float32).reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    return [float(v) for v in flat.tolist()]


def _build_accuracy_summary(targets, pred_labels):
    targets = [int(_to_jsonable_target(v)) for v in targets]
    pred_labels = [int(v) for v in pred_labels]
    total = len(targets)
    correct = sum(int(p == t) for p, t in zip(pred_labels, targets))
    per_class = []
    class_ids = sorted(set(targets))
    for class_id in class_ids:
        indices = [i for i, target in enumerate(targets) if target == class_id]
        class_total = len(indices)
        class_correct = sum(int(pred_labels[i] == targets[i]) for i in indices)
        per_class.append(
            {
                "label": class_id,
                "correct": class_correct,
                "total": class_total,
                "accuracy": float(class_correct / max(class_total, 1)),
            }
        )
    return {
        "accuracy": float(correct / max(total, 1)),
        "per_class_accuracy": per_class,
    }


def _evaluate_tflite_model(cfg: EasyAIConfig, model_path: Path) -> dict:
    import tensorflow as tf

    _, _, val_x, val_y = create_array_datasets(cfg)
    checkpoint_path = _load_checkpoint_path(cfg)
    float_model = build_model(cfg)
    if cfg.task.input_dim == 1:
        input_shape = (1, cfg.dataset.input_shape[1], cfg.dataset.input_shape[0])
    else:
        input_shape = (
            1,
            cfg.dataset.input_shape[1],
            cfg.dataset.input_shape[2],
            cfg.dataset.input_shape[0],
        )
    float_model(tf.zeros(input_shape, dtype=tf.float32), training=False)
    float_model.load_weights(checkpoint_path)
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
    if len(val_y) > 0:
        test_input_index = _choose_reference_index(val_y, cfg.task.type)
        fp32_output = float_model.predict(
            val_x[test_input_index][None, ...].astype(np.float32),
            verbose=0,
        )[0]
        test_output = predictions[test_input_index]
        report["test_input_index"] = test_input_index
        report["test_input_shape"] = list(np.asarray(val_x[test_input_index]).shape)
        report["test_input_target"] = _to_jsonable_target(val_y[test_input_index])
        report["test_fp32_output"] = _to_jsonable_output(fp32_output)
        report["test_output"] = _to_jsonable_output(test_output)
        if cfg.task.type == TaskType.CLASSIFICATION:
            report["test_predicted_label"] = int(np.argmax(test_output))

    if cfg.task.type == TaskType.CLASSIFICATION:
        fp32_predictions = float_model.predict(val_x.astype(np.float32), verbose=0)
        fp32_pred_labels = fp32_predictions.argmax(axis=1)
        pred_labels = predictions.argmax(axis=1)
        accuracy = float((pred_labels == val_y).mean())
        report["eval_accuracy"] = accuracy
        report["fp32_eval_accuracy"] = float((fp32_pred_labels == val_y).mean())
        report["accuracy_delta"] = (
            report["eval_accuracy"] - report["fp32_eval_accuracy"]
        )
        report["fp32_per_class_accuracy"] = _build_accuracy_summary(
            val_y, fp32_pred_labels
        )["per_class_accuracy"]
        report["quantized_per_class_accuracy"] = _build_accuracy_summary(
            val_y, pred_labels
        )["per_class_accuracy"]
        return report

    preds = predictions.reshape(len(predictions), -1)
    targets = np.asarray(val_y).reshape(len(val_y), -1)
    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(math.sqrt(np.mean(np.square(preds - targets))))
    report["eval_mae"] = mae
    report["eval_rmse"] = rmse
    return report


def _run_tflite_ptq_int8(cfg: EasyAIConfig):
    import tensorflow as tf

    checkpoint_path = _load_checkpoint_path(cfg)
    model = build_model(cfg)
    if cfg.task.input_dim == 1:
        input_shape = (1, cfg.dataset.input_shape[1], cfg.dataset.input_shape[0])
    else:
        input_shape = (
            1,
            cfg.dataset.input_shape[1],
            cfg.dataset.input_shape[2],
            cfg.dataset.input_shape[0],
        )
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
    test_report = {
        "backend": cfg.backend.value,
        "method": "ptq",
        "dtype": "int8",
        "checkpoint_source": report["checkpoint_source"],
        "checkpoint": report["checkpoint"],
        "calibration_source": report["calibration_source"],
        "calibration_num_samples": report["calibration_num_samples"],
        "calibration_shuffle": report["calibration_shuffle"],
        "model": str(model_path),
        "test_input_index": report.get("test_input_index"),
        "test_input_shape": report.get("test_input_shape"),
        "test_input_target": report.get("test_input_target"),
        "test_output": report.get("test_output"),
    }
    if "test_predicted_label" in report:
        test_report["test_predicted_label"] = report["test_predicted_label"]
    test_report_path = output_dir / "tflite_test_report.json"
    test_report_path.write_text(
        json.dumps(test_report, indent=2),
        encoding="utf-8",
    )
    report["test_report"] = str(test_report_path)
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
    from esp_ppq.executor.torch import TorchExecutor

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
    dataset_targets = [
        calibration_loader.dataset[idx][1] for idx in range(calibration_size)
    ]
    sample_index = _choose_reference_index(dataset_targets, cfg.task.type)
    sample_input, sample_target = calibration_loader.dataset[sample_index]
    raw_index = (
        int(calibration_loader.dataset.indices[sample_index])
        if hasattr(calibration_loader.dataset, "indices")
        else sample_index
    )
    quant_input = sample_input.unsqueeze(0).to("cpu")
    quant_inputs = [quant_input]

    def collate_fn(batch):
        return batch[0].to("cpu")

    print("input shape:", [1] + cfg.dataset.input_shape)
    quant_graph = espdl_quantize_torch(
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

    with torch.no_grad():
        fp32_output = model(quant_input).detach().cpu().numpy()[0]
    quant_executor = TorchExecutor(graph=quant_graph, device="cpu")
    quant_output = quant_executor(quant_input)[0]
    if isinstance(quant_output, torch.Tensor):
        quant_output = quant_output.detach().cpu().numpy()
    quant_output = np.asarray(quant_output)[0]

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
        "test_input_index": sample_index,
        "test_input_raw_index": raw_index,
        "test_input_shape": list(quant_input.shape),
        "test_input_target": _to_jsonable_target(sample_target),
        "test_fp32_output": _to_jsonable_output(fp32_output),
        "test_output": _to_jsonable_output(quant_output),
        "output": str(model_path),
    }
    if cfg.task.type == TaskType.CLASSIFICATION:
        report["test_predicted_label"] = int(np.argmax(quant_output))
        _, val_loader = create_dataloaders(cfg)
        val_dataset = val_loader.dataset
        val_targets = [val_dataset[idx][1] for idx in range(len(val_dataset))]
        fp32_pred_labels = []
        quant_pred_labels = []
        for idx in range(len(val_dataset)):
            sample_x, _ = val_dataset[idx]
            batched = sample_x.unsqueeze(0).to("cpu")
            with torch.no_grad():
                fp32_pred = model(batched).detach().cpu().numpy()[0]
            quantized_pred = quant_executor(batched)[0]
            if isinstance(quantized_pred, torch.Tensor):
                quantized_pred = quantized_pred.detach().cpu().numpy()
            quantized_pred = np.asarray(quantized_pred)[0]
            fp32_pred_labels.append(int(np.argmax(fp32_pred)))
            quant_pred_labels.append(int(np.argmax(quantized_pred)))
        report["fp32_eval_accuracy"] = _build_accuracy_summary(
            val_targets, fp32_pred_labels
        )["accuracy"]
        report["eval_accuracy"] = _build_accuracy_summary(
            val_targets, quant_pred_labels
        )["accuracy"]
        report["accuracy_delta"] = (
            report["eval_accuracy"] - report["fp32_eval_accuracy"]
        )
        report["fp32_per_class_accuracy"] = _build_accuracy_summary(
            val_targets, fp32_pred_labels
        )["per_class_accuracy"]
        report["quantized_per_class_accuracy"] = _build_accuracy_summary(
            val_targets, quant_pred_labels
        )["per_class_accuracy"]
    test_report = {
        "backend": cfg.backend.value,
        "method": "ptq",
        "dtype": "int8",
        "checkpoint_source": report["checkpoint_source"],
        "checkpoint": report["checkpoint"],
        "target": report["target"],
        "calibration_source": report["calibration_source"],
        "calibration_num_samples": report["calibration_num_samples"],
        "calibration_steps": report["calibration_steps"],
        "calibration_shuffle": report["calibration_shuffle"],
        "model": str(model_path),
        "test_input_index": report["test_input_index"],
        "test_input_raw_index": report["test_input_raw_index"],
        "test_input_shape": report["test_input_shape"],
        "test_input_target": report["test_input_target"],
        "test_output": report["test_output"],
    }
    if "test_predicted_label" in report:
        test_report["test_predicted_label"] = report["test_predicted_label"]
    test_report_path = output_dir / "espdl_test_report.json"
    test_report_path.write_text(
        json.dumps(test_report, indent=2),
        encoding="utf-8",
    )
    report["test_report"] = str(test_report_path)
    return report


def run_quantization(cfg: EasyAIConfig):
    if cfg.task.input_dim != 1:
        raise NotImplementedError("Quantization currently supports only 1D tasks.")
    if cfg.backend == Backend.TFLITE_MICRO:
        return _run_tflite_ptq_int8(cfg)
    if cfg.backend == Backend.ESP_DL:
        return _run_espdl_ptq_int8(cfg)
    raise NotImplementedError(
        f"Quantization is not implemented for backend={cfg.backend.value}."
    )
