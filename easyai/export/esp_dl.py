"""ESP-DL export entrypoint."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from easyai.config.schema import Backend, EasyAIConfig, TaskType

from .common import (
    format_float_array,
    format_float_literal,
    format_int_array,
    parse_info_tensor_section,
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
    model_path = output_dir / "model_int8.espdl"
    info_path = output_dir / "model_int8.info"
    if not model_path.exists():
        raise FileNotFoundError(f"Quantized model not found: {model_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Quantized model info not found: {info_path}")
    info_text = info_path.read_text(encoding="utf-8")
    return {
        "report": quant_report,
        "model_path": model_path,
        "input_tensor": parse_info_tensor_section(info_text, "test inputs value"),
        "output_tensor": parse_info_tensor_section(info_text, "test outputs value"),
    }


def _label_decl(symbol: str, task_type: TaskType) -> str:
    if task_type != TaskType.CLASSIFICATION:
        return ""
    return f"int {symbol}_model_predict_label(const float* input, size_t input_len);"


def _class_names_decl(symbol: str, task_type: TaskType, num_classes: int) -> str:
    if task_type != TaskType.CLASSIFICATION:
        return ""
    return f"extern const char* const {symbol}_class_names[{num_classes}];"


def _target_decl(symbol: str, task_type: TaskType, output_count: int) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"extern const int {symbol}_test_target_label;"
    return f"extern const float {symbol}_test_target[{output_count}];"


def _class_names_def(symbol: str, task_type: TaskType, class_names) -> str:
    if task_type != TaskType.CLASSIFICATION:
        return ""
    return (
        f"const char* const {symbol}_class_names[{len(class_names)}] = {{\n"
        + "    "
        + ", ".join(f'"{name}"' for name in class_names)
        + "\n};"
    )


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


def _label_impl(symbol: str, task_type: TaskType, output_count: int) -> str:
    if task_type != TaskType.CLASSIFICATION:
        return ""
    return f"""
int {symbol}_model_predict_label(const float* input, size_t input_len) {{
    int8_t output[{output_count}] = {{0}};
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


def _target_comment(task_type: TaskType, target) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"// Reference label for this sample: {int(target)}"
    if not isinstance(target, list):
        target = [float(target)]
    return "// Reference target for this sample: " + ", ".join(
        f"{float(v):.6g}" for v in target
    )


def _main_body(symbol: str, task_type: TaskType, output_count: int) -> str:
    if task_type == TaskType.CLASSIFICATION:
        return f"""    int8_t output[{output_count}] = {{0}};
    if (!{symbol}_model_predict(
            {symbol}_test_input,
            sizeof({symbol}_test_input) / sizeof(float),
            output,
            sizeof(output) / sizeof(int8_t))) {{
        printf("Model prediction failed.\\n");
        return;
    }}

    printf("Model prediction output:\\n");
    for (size_t i = 0; i < sizeof(output) / sizeof(int8_t); ++i) {{
        printf("  output[%zu]: %d\\n", i, (int)output[i]);
    }}

    int predicted = {symbol}_model_predict_label(
        {symbol}_test_input,
        sizeof({symbol}_test_input) / sizeof(float));
    printf("Predicted label: %d\\n", predicted);
    printf("Target label   : %d\\n", {symbol}_test_target_label);"""

    return f"""    int8_t output[{output_count}] = {{0}};
    if (!{symbol}_model_predict(
            {symbol}_test_input,
            sizeof({symbol}_test_input) / sizeof(float),
            output,
            sizeof(output) / sizeof(int8_t))) {{
        printf("Model prediction failed.\\n");
        return;
    }}

    printf("Model prediction output:\\n");
    for (size_t i = 0; i < sizeof(output) / sizeof(int8_t); ++i) {{
        printf("  output[%zu]: %d\\n", i, (int)output[i]);
    }}

    printf("Target:\\n");
    for (size_t i = 0; i < {output_count}; ++i) {{
        printf("  target[%zu]: %f\\n", i, {symbol}_test_target[i]);
    }}"""


def _quantize_loop(raw_input_shape, reorder_to_lc: bool) -> str:
    channels, length = raw_input_shape
    if reorder_to_lc:
        return f"""for (size_t t = 0; t < {length}; ++t) {{
        for (size_t c = 0; c < {channels}; ++c) {{
            const float normalized = ApplyPreprocess(input[SourceFlatIndex(c, t)], c);
            input_ptr[dst++] = dl::quantize<int8_t>(normalized, input_inv_scale);
        }}
    }}"""
    return f"""for (size_t c = 0; c < {channels}; ++c) {{
        for (size_t t = 0; t < {length}; ++t) {{
            const float normalized = ApplyPreprocess(input[SourceFlatIndex(c, t)], c);
            input_ptr[dst++] = dl::quantize<int8_t>(normalized, input_inv_scale);
        }}
    }}"""


def _float_input_loop(raw_input_shape, reorder_to_lc: bool) -> str:
    channels, length = raw_input_shape
    if reorder_to_lc:
        return f"""for (size_t t = 0; t < {length}; ++t) {{
        for (size_t c = 0; c < {channels}; ++c) {{
            input_ptr[dst++] = ApplyPreprocess(input[SourceFlatIndex(c, t)], c);
        }}
    }}"""
    return f"""for (size_t c = 0; c < {channels}; ++c) {{
        for (size_t t = 0; t < {length}; ++t) {{
            input_ptr[dst++] = ApplyPreprocess(input[SourceFlatIndex(c, t)], c);
        }}
    }}"""


def run_export(cfg: EasyAIConfig):
    if cfg.backend != Backend.ESP_DL:
        raise NotImplementedError(
            f"ESP-DL exporter received backend={cfg.backend.value}."
        )

    assets = _load_quantization_assets(cfg)
    quant_report = assets["report"]
    input_tensor = assets["input_tensor"]
    output_tensor = assets["output_tensor"]
    symbol = sanitize_identifier(cfg.project_name)

    export_dir = Path(cfg.output_dir) / "export"
    project_dir = export_dir / cfg.project_name
    component_dir = project_dir / "components" / cfg.project_name
    main_dir = project_dir / "main"
    if project_dir.exists():
        shutil.rmtree(project_dir)
    stale_root_main = export_dir / "main.cpp"
    if stale_root_main.exists():
        stale_root_main.unlink()
    component_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(assets["model_path"], component_dir / "model.espdl")
    main_cpp_path = main_dir / "main.cpp"
    stale_component_main = component_dir / "main.cpp"
    if stale_component_main.exists():
        stale_component_main.unlink()

    write_text(
        project_dir / "CMakeLists.txt",
        render_template(
            "esp_dl/1d/project/CMakeLists.txt.tpl",
            {"project_name": cfg.project_name},
        ),
    )
    write_text(
        project_dir / "partitions.csv",
        render_template("esp_dl/1d/project/partitions.csv.tpl", {}),
    )
    write_text(
        project_dir / "sdkconfig.defaults",
        render_template("esp_dl/1d/project/sdkconfig.defaults.tpl", {}),
    )
    write_text(
        main_dir / "CMakeLists.txt",
        render_template("esp_dl/1d/project/main/CMakeLists.txt.tpl", {}),
    )
    write_text(
        main_dir / "idf_component.yml",
        render_template(
            "esp_dl/1d/project/main/idf_component.yml.tpl",
            {"project_name": cfg.project_name},
        ),
    )

    calibration_source = quant_report.get("calibration_source", "reuse_val")
    split_name = "val" if calibration_source == "reuse_val" else "train"
    raw_train_features, _ = load_raw_split(cfg, "train")
    raw_split_features, raw_split_targets = load_raw_split(cfg, split_name)
    sample_index = int(quant_report.get("test_input_index", 0))
    target = to_jsonable_target(raw_split_targets[sample_index])
    raw_feature = raw_split_features[sample_index]
    preprocess = extract_preprocess(cfg, calibration_source, raw_train_features)

    raw_input_values = flatten_raw_input(raw_feature)
    raw_input_shape = list(cfg.dataset.input_shape)
    model_input_shape = input_tensor["shape"]
    input_count = len(raw_input_values)
    output_count = math.prod(output_tensor["shape"])
    reorder_to_lc = model_input_shape == [1, raw_input_shape[1], raw_input_shape[0]]
    class_names = getattr(cfg.dataset, "class_names", []) or []

    write_text(
        component_dir / "CMakeLists.txt",
        render_template("esp_dl/1d/CMakeLists.txt.tpl", {}),
    )
    write_text(
        component_dir / "idf_component.yml",
        render_template("esp_dl/1d/idf_component.yml.tpl", {}),
    )
    write_text(
        component_dir / "predict.h",
        render_template(
            "esp_dl/1d/predict.h.tpl",
            {
                "symbol": symbol,
                "label_decl": _label_decl(symbol, cfg.task.type),
            },
        ),
    )
    write_text(
        component_dir / "predict.cpp",
        render_template(
            "esp_dl/1d/predict.cpp.tpl",
            {
                "symbol": symbol,
                "raw_input_length": raw_input_shape[1],
                "raw_input_count": input_count,
                "output_count": output_count,
                "rescale_literal": format_float_literal(preprocess["rescale"]),
                "quantize_loop": _quantize_loop(raw_input_shape, reorder_to_lc),
                "float_input_loop": _float_input_loop(raw_input_shape, reorder_to_lc),
                "label_impl": _label_impl(symbol, cfg.task.type, output_count),
                "raw_input_shape": raw_input_shape,
                "model_input_shape": model_input_shape,
                "input_tensor_name": input_tensor["name"],
                "input_exponent": input_tensor["exponent"],
                "output_tensor_name": output_tensor["name"],
                "output_exponent": output_tensor["exponent"],
                "sample_index": sample_index,
                "target_comment": _target_comment(cfg.task.type, target),
            },
        ),
    )
    write_text(
        component_dir / "test_data.h",
        render_template(
            "esp_dl/1d/test_data.h.tpl",
            {
                "symbol": symbol,
                "input_count": input_count,
                "output_count": output_count,
                "channels": raw_input_shape[0],
                "class_names_decl": _class_names_decl(
                    symbol, cfg.task.type, getattr(cfg.dataset, "num_classes", 0)
                ),
                "target_decl": _target_decl(symbol, cfg.task.type, output_count),
            },
        ),
    )
    write_text(
        component_dir / "test_data.cpp",
        render_template(
            "esp_dl/1d/test_data.cpp.tpl",
            {
                "symbol": symbol,
                "input_count": input_count,
                "test_input_values": format_float_array(raw_input_values),
                "output_count": output_count,
                "reference_output_values": format_int_array(output_tensor["values"]),
                "channels": raw_input_shape[0],
                "normalize_mean_values": format_float_array(preprocess["mean"]),
                "normalize_std_values": format_float_array(preprocess["std"]),
                "sample_index": sample_index,
                "class_names_def": _class_names_def(symbol, cfg.task.type, class_names),
                "target_def": _target_def(symbol, cfg.task.type, target),
            },
        ),
    )
    write_text(
        main_cpp_path,
        render_template(
            "esp_dl/1d/main.cpp.tpl",
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
        "model": str(component_dir / "model.espdl"),
        "test_input_index": sample_index,
        "test_input_shape": raw_input_shape,
        "test_input_target": target,
        "reference_output": output_tensor["values"],
    }
    export_report_path = project_dir / "export_report.json"
    export_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["export_report"] = str(export_report_path)
    return report
