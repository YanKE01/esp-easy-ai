"""TFLite Micro export entrypoint."""

from __future__ import annotations

import json
import math
from pathlib import Path

import tensorflow as tf

from easyai.config.schema import Backend, EasyAIConfig, TaskType

from .common import (
    format_byte_array,
    format_float_array,
    format_float_literal,
    sanitize_identifier,
    write_text,
)
from .sample import (
    extract_preprocess,
    flatten_raw_input,
    load_raw_split,
    to_jsonable_target,
)
from .templates import render_template


def _load_quantization_assets(cfg: EasyAIConfig) -> dict:
    output_dir = Path(cfg.output_dir)
    quant_report_path = output_dir / "quantization_report.json"
    if not quant_report_path.exists():
        raise FileNotFoundError(
            f"Quantization report not found: {quant_report_path}. Run quantize first."
        )
    quant_report = json.loads(quant_report_path.read_text(encoding="utf-8"))
    test_report_path = output_dir / "tflite_test_report.json"
    if not test_report_path.exists():
        raise FileNotFoundError(
            f"TFLite test report not found: {test_report_path}. Run quantize first."
        )
    test_report = json.loads(test_report_path.read_text(encoding="utf-8"))
    model_path = output_dir / "model_int8.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Quantized model not found: {model_path}")
    return {
        "quant_report": quant_report,
        "test_report": test_report,
        "model_path": model_path,
    }


def _load_tflite_metadata(model_path: Path) -> dict:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    op_details = interpreter._get_ops_details()  # pylint: disable=protected-access
    return {
        "input_shape": list(input_details["shape"]),
        "input_dtype": np_dtype_name(input_details["dtype"]),
        "input_scale": float(input_details["quantization"][0]),
        "input_zero_point": int(input_details["quantization"][1]),
        "output_shape": list(output_details["shape"]),
        "output_dtype": np_dtype_name(output_details["dtype"]),
        "output_scale": float(output_details["quantization"][0]),
        "output_zero_point": int(output_details["quantization"][1]),
        "ops": [item["op_name"] for item in op_details],
    }


def np_dtype_name(dtype) -> str:
    return getattr(dtype, "__name__", str(dtype))


def _resolver_method(op_name: str) -> str:
    mapping = {
        "ADD": "Add",
        "AVERAGE_POOL_2D": "AveragePool2D",
        "CONCATENATION": "Concatenation",
        "CONV_2D": "Conv2D",
        "DEPTHWISE_CONV_2D": "DepthwiseConv2D",
        "EXPAND_DIMS": "ExpandDims",
        "FULLY_CONNECTED": "FullyConnected",
        "LOGISTIC": "Logistic",
        "MAX_POOL_2D": "MaxPool2D",
        "MEAN": "Mean",
        "MUL": "Mul",
        "PAD": "Pad",
        "QUANTIZE": "Quantize",
        "RESHAPE": "Reshape",
        "SOFTMAX": "Softmax",
        "SQUEEZE": "Squeeze",
        "STRIDED_SLICE": "StridedSlice",
        "SUB": "Sub",
        "TRANSPOSE": "Transpose",
    }
    if op_name not in mapping:
        raise NotImplementedError(
            f"TFLite Micro export does not support resolver mapping for op '{op_name}'."
        )
    return mapping[op_name]


def _resolver_code(ops: list[str]) -> tuple[int, str]:
    unique_ops: list[str] = []
    for op in ops:
        if op == "DELEGATE":
            continue
        if op not in unique_ops:
            unique_ops.append(op)
    lines = []
    for op in unique_ops:
        method = _resolver_method(op)
        lines.append(
            f"    if (resolver.Add{method}() != kTfLiteOk) {{\n"
            f'        MicroPrintf("Failed to add {op} operator.");\n'
            f"        return false;\n"
            f"    }}"
        )
    return len(unique_ops), "\n\n".join(lines)


def _target_decl(symbol: str, task_type: TaskType, output_count: int) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"extern const int {symbol}_test_target_label;"
    return f"extern const float {symbol}_test_target[{output_count}];"


def _target_def(symbol: str, task_type: TaskType, target) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"const int {symbol}_test_target_label = {int(target)};"
    if not isinstance(target, list):
        target = [float(target)]
    return (
        f"const float {symbol}_test_target[{len(target)}] = {{\n"
        f"{format_float_array(target)}\n"
        f"}};"
    )


def _target_comment(task_type: TaskType, target) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"// Reference label for this sample: {int(target)}"
    if not isinstance(target, list):
        target = [float(target)]
    return "// Reference target for this sample: " + ", ".join(
        f"{float(v):.6g}" for v in target
    )


def _label_decl(symbol: str, task_type: TaskType) -> str:
    if task_type != TaskType.CLASSIFICATION:
        return ""
    return f"int {symbol}_model_predict_label(const float* input, size_t input_len);"


def _label_impl(symbol: str, task_type: TaskType, output_count: int) -> str:
    if task_type != TaskType.CLASSIFICATION:
        return ""
    return f"""
int {symbol}_model_predict_label(const float* input, size_t input_len) {{
    float output[{output_count}] = {{0}};
    if (!{symbol}_model_predict(input, input_len, output, {output_count})) {{
        return -1;
    }}
    int best_index = 0;
    for (size_t i = 1; i < {output_count}; ++i) {{
        if (output[i] > output[best_index]) {{
            best_index = static_cast<int>(i);
        }}
    }}
    return best_index;
}}
"""


def _main_body(symbol: str, task_type: TaskType, output_count: int) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"""    float output[{output_count}] = {{0}};
    if (!{symbol}_model_predict(
            {symbol}_test_input,
            sizeof({symbol}_test_input) / sizeof(float),
            output,
            sizeof(output) / sizeof(float))) {{
        printf("Model prediction failed.\\n");
        return;
    }}

    printf("Model prediction output:\\n");
    for (size_t i = 0; i < sizeof(output) / sizeof(float); ++i) {{
        printf("  output[%zu]: %f\\n", i, output[i]);
    }}

    int predicted = {symbol}_model_predict_label(
        {symbol}_test_input,
        sizeof({symbol}_test_input) / sizeof(float));
    printf("Predicted label: %d\\n", predicted);
    printf("Target label   : %d\\n", {symbol}_test_target_label);"""

    return f"""    float output[{output_count}] = {{0}};
    if (!{symbol}_model_predict(
            {symbol}_test_input,
            sizeof({symbol}_test_input) / sizeof(float),
            output,
            sizeof(output) / sizeof(float))) {{
        printf("Model prediction failed.\\n");
        return;
    }}

    printf("Model prediction output:\\n");
    for (size_t i = 0; i < sizeof(output) / sizeof(float); ++i) {{
        printf("  output[%zu]: %f\\n", i, output[i]);
    }}

    printf("Target:\\n");
    for (size_t i = 0; i < {output_count}; ++i) {{
        printf("  target[%zu]: %f\\n", i, {symbol}_test_target[i]);
    }}"""


def run_export(cfg: EasyAIConfig):
    if cfg.backend != Backend.TFLITE_MICRO:
        raise NotImplementedError(
            f"TFLite Micro exporter received backend={cfg.backend.value}."
        )

    assets = _load_quantization_assets(cfg)
    quant_report = assets["quant_report"]
    test_report = assets["test_report"]
    model_path = assets["model_path"]
    model_bytes = model_path.read_bytes()
    metadata = _load_tflite_metadata(model_path)
    symbol = sanitize_identifier(cfg.project_name)

    export_dir = Path(cfg.output_dir) / "export"
    project_dir = export_dir / cfg.project_name
    component_dir = project_dir / "components" / cfg.project_name
    main_dir = project_dir / "main"
    if project_dir.exists():
        import shutil

        shutil.rmtree(project_dir)
    stale_root_main = export_dir / "main.cpp"
    if stale_root_main.exists():
        stale_root_main.unlink()
    component_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)
    main_cpp_path = main_dir / "main.cpp"
    stale_component_main = component_dir / "main.cpp"
    if stale_component_main.exists():
        stale_component_main.unlink()

    write_text(
        project_dir / "CMakeLists.txt",
        render_template(
            "tflite_micro/1d/project/CMakeLists.txt.tpl",
            {"project_name": cfg.project_name},
        ),
    )
    write_text(
        project_dir / "partitions.csv",
        render_template("tflite_micro/1d/project/partitions.csv.tpl", {}),
    )
    write_text(
        project_dir / "sdkconfig.defaults",
        render_template("tflite_micro/1d/project/sdkconfig.defaults.tpl", {}),
    )
    write_text(
        main_dir / "CMakeLists.txt",
        render_template("tflite_micro/1d/project/main/CMakeLists.txt.tpl", {}),
    )
    write_text(
        main_dir / "idf_component.yml",
        render_template(
            "tflite_micro/1d/project/main/idf_component.yml.tpl",
            {"project_name": cfg.project_name},
        ),
    )

    calibration_source = quant_report.get("calibration_source", "reuse_val")
    split_name = "val" if calibration_source == "reuse_val" else "train"
    raw_train_features, _ = load_raw_split(cfg, "train")
    raw_split_features, raw_split_targets = load_raw_split(cfg, split_name)
    sample_index = int(
        test_report.get("test_input_index", quant_report.get("test_input_index", 0))
    )
    target = to_jsonable_target(raw_split_targets[sample_index])
    raw_feature = raw_split_features[sample_index]
    preprocess = extract_preprocess(cfg, calibration_source, raw_train_features)

    raw_input_values = flatten_raw_input(raw_feature)
    raw_input_shape = list(cfg.dataset.input_shape)
    input_count = len(raw_input_values)
    output_count = math.prod(metadata["output_shape"])
    resolver_size, resolver_code = _resolver_code(metadata["ops"])
    reorder_to_lc = metadata["input_shape"] == [
        1,
        raw_input_shape[1],
        raw_input_shape[0],
    ]

    write_text(
        component_dir / "CMakeLists.txt",
        render_template("tflite_micro/1d/CMakeLists.txt.tpl", {}),
    )
    write_text(
        component_dir / "idf_component.yml",
        render_template("tflite_micro/1d/idf_component.yml.tpl", {}),
    )
    write_text(
        component_dir / "model.h",
        render_template(
            "tflite_micro/1d/model.h.tpl",
            {
                "symbol": symbol,
                "byte_count": len(model_bytes),
            },
        ),
    )
    write_text(
        component_dir / "model.cpp",
        render_template(
            "tflite_micro/1d/model.cpp.tpl",
            {
                "symbol": symbol,
                "byte_count": len(model_bytes),
                "model_bytes": format_byte_array(model_bytes),
            },
        ),
    )
    write_text(
        component_dir / "predict.h",
        render_template(
            "tflite_micro/1d/predict.h.tpl",
            {
                "symbol": symbol,
                "label_decl": _label_decl(symbol, cfg.task.type),
            },
        ),
    )
    write_text(
        component_dir / "predict.cpp",
        render_template(
            "tflite_micro/1d/predict.cpp.tpl",
            {
                "symbol": symbol,
                "tensor_arena_size": 64 * 1024,
                "input_count": input_count,
                "input_channels": raw_input_shape[0],
                "input_length": raw_input_shape[1],
                "raw_input_shape": raw_input_shape,
                "model_input_shape": metadata["input_shape"],
                "output_count": output_count,
                "resolver_size": resolver_size,
                "resolver_code": resolver_code,
                "rescale_literal": format_float_literal(preprocess["rescale"]),
                "normalize_mean_values": format_float_array(preprocess["mean"]),
                "normalize_std_values": format_float_array(preprocess["std"]),
                "input_scale": format_float_literal(metadata["input_scale"]),
                "input_zero_point": metadata["input_zero_point"],
                "output_scale": format_float_literal(metadata["output_scale"]),
                "output_zero_point": metadata["output_zero_point"],
                "input_is_int8": "true"
                if metadata["input_dtype"] == "int8"
                else "false",
                "output_is_int8": "true"
                if metadata["output_dtype"] == "int8"
                else "false",
                "reorder_to_lc": "true" if reorder_to_lc else "false",
                "label_impl": _label_impl(symbol, cfg.task.type, output_count),
                "sample_index": sample_index,
                "target_comment": _target_comment(cfg.task.type, target),
            },
        ),
    )
    write_text(
        component_dir / "test_data.h",
        render_template(
            "tflite_micro/1d/test_data.h.tpl",
            {
                "symbol": symbol,
                "input_count": input_count,
                "output_count": output_count,
                "target_decl": _target_decl(symbol, cfg.task.type, output_count),
            },
        ),
    )
    write_text(
        component_dir / "test_data.cpp",
        render_template(
            "tflite_micro/1d/test_data.cpp.tpl",
            {
                "symbol": symbol,
                "input_count": input_count,
                "test_input_values": format_float_array(raw_input_values),
                "output_count": output_count,
                "reference_output_values": format_float_array(
                    test_report["test_output"]
                    if isinstance(test_report["test_output"], list)
                    else [test_report["test_output"]]
                ),
                "sample_index": sample_index,
                "target_def": _target_def(symbol, cfg.task.type, target),
            },
        ),
    )
    write_text(
        main_cpp_path,
        render_template(
            "tflite_micro/1d/main.cpp.tpl",
            {
                "symbol": symbol,
                "main_body": _main_body(symbol, cfg.task.type, output_count),
            },
        ),
    )

    report = {
        "backend": cfg.backend.value,
        "export_dir": str(export_dir),
        "project_dir": str(project_dir),
        "component_dir": str(component_dir),
        "main_cpp": str(main_cpp_path),
        "model": str(component_dir / "model.cpp"),
        "test_input_index": sample_index,
        "test_input_shape": raw_input_shape,
        "test_input_target": target,
        "reference_output": test_report["test_output"],
    }
    export_report_path = project_dir / "export_report.json"
    export_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["export_report"] = str(export_report_path)
    return report
