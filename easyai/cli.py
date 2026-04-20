"""Command-line interface for esp-easy-ai."""

import argparse
import io
import sys

from easyai.config.schema import Backend
from easyai.config import validate_config_file
from easyai.export import run_export
from easyai.model import validate_model_definition
from easyai.quantize import run_quantization
from easyai.train import run_training


def _format_model_definition(cfg):
    if cfg.model is None:
        return None

    model = validate_model_definition(cfg)
    if cfg.backend == Backend.ESP_DL:
        return str(model)

    if cfg.backend == Backend.TFLITE_MICRO:
        buf = io.StringIO()
        model.summary(print_fn=lambda line: buf.write(line + "\n"))
        return buf.getvalue().rstrip()

    return str(model)


def cmd_validate(args):
    try:
        cfg = validate_config_file(args.config)
    except Exception as e:
        print(f"[FAIL] Validation: {e}", file=sys.stderr)
        sys.exit(1)
    print("[OK]  Config file")

    print(f"\n  project : {cfg.project_name}")
    print(f"  backend : {cfg.backend.value}")
    print(f"  task    : {cfg.task.type.value} (input_dim={cfg.task.input_dim})")
    print(f"  dataset : {cfg.dataset.format}  root={cfg.dataset.root}")
    if cfg.model:
        print(f"  layers  : {len(cfg.model.layers)}")
        model_definition = _format_model_definition(cfg)
        if model_definition:
            print("\n  model:")
            for line in model_definition.splitlines():
                print(f"    {line}")


def cmd_train(args):
    try:
        cfg = validate_config_file(args.config)
    except Exception as e:
        print(f"[FAIL] Validation: {e}", file=sys.stderr)
        sys.exit(1)

    print("[OK]  Validation passed")
    print(f"  project : {cfg.project_name}")
    print(f"  backend : {cfg.backend.value}")
    print(f"  output  : {cfg.output_dir}")
    print("\n  training:")

    try:
        report = run_training(cfg)
    except Exception as e:
        print(f"[FAIL] Training: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK]  Training finished")
    print(f"  best_epoch : {report['best_epoch']}")
    print(f"  monitor    : {report['monitor']}")
    print(f"  best_value : {report['best_value']}")
    print(f"  report     : {cfg.output_dir}/report.json")
    print(f"  best_ckpt  : {report['best_checkpoint']}")
    print(f"  last_ckpt  : {report['last_checkpoint']}")


def cmd_quantize(args):
    try:
        cfg = validate_config_file(args.config)
    except Exception as e:
        print(f"[FAIL] Validation: {e}", file=sys.stderr)
        sys.exit(1)

    print("[OK]  Validation passed")
    print(f"  project : {cfg.project_name}")
    print(f"  backend : {cfg.backend.value}")
    print(f"  output  : {cfg.output_dir}")
    print("\n  quantization:")

    try:
        report = run_quantization(cfg)
    except Exception as e:
        print(f"[FAIL] Quantization: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK]  Quantization finished")
    print(f"  method              : {report['method']}")
    print(f"  dtype               : {report['dtype']}")
    if report.get("checkpoint_source") is not None:
        print(f"  checkpoint_source   : {report['checkpoint_source']}")
    if report.get("checkpoint") is not None:
        print(f"  checkpoint          : {report['checkpoint']}")
    if report.get("target") is not None:
        print(f"  target              : {report['target']}")
    print(f"  calibration_source  : {report['calibration_source']}")
    print(f"  calibration_samples : {report['calibration_num_samples']}")
    print(f"  calibration_shuffle : {report['calibration_shuffle']}")
    if "test_input_index" in report:
        print(f"  test_input_index    : {report['test_input_index']}")
    if "test_input_shape" in report:
        print(f"  test_input_shape    : {report['test_input_shape']}")
    if "test_input_target" in report:
        print(f"  test_input_target   : {report['test_input_target']}")
    if "test_output" in report:
        print(f"  test_output         : {report['test_output']}")
    if "test_predicted_label" in report:
        print(f"  test_predicted_label: {report['test_predicted_label']}")
    if "test_report" in report:
        print(f"  test_report         : {report['test_report']}")
    if "eval_source" in report:
        print(f"  eval_source         : {report['eval_source']}")
    if "eval_num_samples" in report:
        print(f"  eval_samples        : {report['eval_num_samples']}")
    if "eval_accuracy" in report:
        print(f"  eval_accuracy       : {report['eval_accuracy']:.4f}")
    if "eval_mae" in report:
        print(f"  eval_mae            : {report['eval_mae']:.4f}")
    if "eval_rmse" in report:
        print(f"  eval_rmse           : {report['eval_rmse']:.4f}")
    print(f"  output              : {report['output']}")
    if report.get("eval_examples"):
        print("  eval_examples:")
        for example in report["eval_examples"]:
            print(
                f"    idx={example['index']} pred={example['pred']} target={example['target']}"
            )


def cmd_export(args):
    try:
        cfg = validate_config_file(args.config)
    except Exception as e:
        print(f"[FAIL] Validation: {e}", file=sys.stderr)
        sys.exit(1)

    print("[OK]  Validation passed")
    print(f"  project : {cfg.project_name}")
    print(f"  backend : {cfg.backend.value}")
    print(f"  output  : {cfg.output_dir}")
    print("\n  export:")

    try:
        report = run_export(cfg)
    except Exception as e:
        print(f"[FAIL] Export: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n[OK]  Export finished")
    if "project_dir" in report:
        print(f"  project_dir         : {report['project_dir']}")
    print(f"  component_dir       : {report['component_dir']}")
    print(f"  main_cpp            : {report['main_cpp']}")
    print(f"  model               : {report['model']}")
    print(f"  test_input_index    : {report['test_input_index']}")
    print(f"  test_input_shape    : {report['test_input_shape']}")
    print(f"  test_input_target   : {report['test_input_target']}")
    print(f"  reference_output    : {report['reference_output']}")
    print(f"  export_report       : {report['export_report']}")


def main():
    parser = argparse.ArgumentParser(prog="easyai")
    sub = parser.add_subparsers(dest="command")

    p_validate = sub.add_parser("validate", help="Validate a JSON config file")
    p_validate.add_argument("config", help="Path to config JSON")

    p_train = sub.add_parser("train", help="Validate config and run training")
    p_train.add_argument("config", help="Path to config JSON")

    p_quantize = sub.add_parser(
        "quantize", help="Validate config and run PTQ INT8 quantization"
    )
    p_quantize.add_argument("config", help="Path to config JSON")

    p_export = sub.add_parser(
        "export", help="Validate config and export ESP component reference files"
    )
    p_export.add_argument("config", help="Path to config JSON")

    args = parser.parse_args()
    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "quantize":
        cmd_quantize(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
