---
name: esp-easy-ai
description: Use when authoring, validating, or reasoning about esp-easy-ai JSON configs, or running its validate/train/quantize/export pipeline. Explains simple mode vs advanced mode, per-format dataset rules, auto-filled defaults, and the current 1D-only public surface for both esp-dl and tflite-micro backends.
---

# esp-easy-ai

Use this skill for tasks about `esp-easy-ai` configuration, validation, training, quantization, and export.

## Current Scope

The active public pipeline is **1D only**.

Supported dataset formats:

- `windowed_per_class_csv` (1D classification)
- `continuous_per_class_csv` (1D classification)
- `windowed_regression_csv` (1D regression)

Supported backends:

- `esp-dl` (PyTorch + ESP-PPQ)
- `tflite-micro` (Keras + TFLite converter)

Supported CLI:

- `uv run esp-easy-ai validate <config.json>`
- `uv run esp-easy-ai train <config.json>`
- `uv run esp-easy-ai quantize <config.json>`
- `uv run esp-easy-ai export <config.json>`

`train` / `quantize` / `export` all run a full `validate` first. Quantize / export do **not** retrain; they read existing checkpoints / quantized artifacts.

Do **not** assume 2D classification or YOLO detection is part of the current public surface.

## Two Authoring Styles, Same Keys

Both styles share the same top-level structure:

- `backend`
- `task`
- `project_name` *(auto-filled if omitted)*
- `output_dir` *(auto-filled if omitted)*
- `seed` *(defaults to 42)*
- `dataset`
- `model`
- `training`
- `quantization`

The difference is how much you write. `validate_config` always runs `complete_small_white_user_config` first, so any field with a documented default can be omitted.

### Simple mode

- `model.preset` picks a preset architecture.
- `dataset` may contain only `format`, `root`, `num_features`; the rest is inferred by reading the CSVs on disk.
- `training.epochs` is the only training field you need.
- `quantization` block can be omitted entirely.

### Advanced mode

- `model.layers` defines the full layer stack.
- `training` supplies the full optimizer / loss / scheduler / checkpoint block.
- `dataset` is written explicitly (no inference needed).

Simple and advanced differ only in how `model` is specified and in whether you spell out training/dataset defaults. You can mix: e.g. advanced model with simple dataset.

## Task × Dataset Format Compatibility

Enforced by the schema:

| `task.input_dim` | `task.type`      | Valid `dataset.format`                                  |
|------------------|------------------|---------------------------------------------------------|
| 1                | `classification` | `windowed_per_class_csv`, `continuous_per_class_csv`    |
| 1                | `regression`     | `windowed_regression_csv`                               |

Any other combination is rejected.

## Simple Mode — Minimal Examples

### Classification (esp-dl)

```json
{
  "backend": "esp-dl",
  "task": { "input_dim": 1, "type": "classification" },
  "dataset": {
    "format": "windowed_per_class_csv",
    "root": "/abs/path/to/dataset",
    "num_features": 3
  },
  "model": { "preset": "tiny_conv_1d" },
  "training": { "epochs": 200 },
  "quantization": { "enable": true, "checkpoint": "best" }
}
```

What gets auto-filled:

- `project_name` ← `<root_folder>_<task.type>`
- `output_dir` ← `./outputs/<project_name>`
- `seed` ← `42`
- `dataset.train_dir` / `val_dir` ← `"train"` / `"test"`
- `dataset.input_shape` ← `[num_features, row_len / num_features]` (from the first row of the first CSV in `train_dir`)
- `dataset.class_names` ← sorted CSV stems under `train_dir`
- `dataset.num_classes` ← `len(class_names)`
- `dataset.feature_layout` ← `"interleaved"`
- `dataset.has_header` ← `false`
- `dataset.batch_size` ← `64`, `num_workers` ← `4`
- `dataset.transforms` ← `{"train": "none", "val": "none"}`
- `dataset.calibration` ← `{"source": "reuse_val", "num_samples": 256}`
- Full `training` block (Adam 1e-3, CosineAnnealingLR, `cross_entropy`, `accuracy`, `val_accuracy` monitor, device `auto`)
- Full `quantization` block (`method=ptq`, `dtype=int8`, `target=esp32s3`, `checkpoint=best`)
  For `esp-dl`, `target` is chip-limited to `esp32s3` / `esp32p4`. For `tflite-micro`, the field is accepted but not chip-limited.

### Regression

```json
{
  "backend": "tflite-micro",
  "task": { "input_dim": 1, "type": "regression" },
  "dataset": {
    "format": "windowed_regression_csv",
    "root": "/abs/path/to/dataset",
    "num_features": 1
  },
  "model": { "preset": "tiny_mlp_1d" },
  "training": { "epochs": 100 }
}
```

Regression-specific defaults:

- `dataset.label_start_col` / `label_end_col` ← `0` (single scalar label in column 0)
- `dataset.output_dim` ← `1`
- `training.loss` ← `"mse"`, `metrics` ← `["mae", "rmse"]`, checkpoint monitor ← `"val_mae"`

If your labels are not a single scalar in column 0, write `label_start_col` / `label_end_col` / `output_dim` explicitly.

## Advanced Mode — Full Example

```json
{
  "backend": "esp-dl",
  "task": { "input_dim": 1, "type": "classification" },
  "project_name": "my_1d_classifier",
  "output_dir": "./outputs/my_1d_classifier",
  "seed": 42,
  "dataset": {
    "format": "windowed_per_class_csv",
    "root": "/abs/path/to/dataset",
    "train_dir": "train",
    "val_dir": "test",
    "num_features": 3,
    "feature_layout": "interleaved",
    "has_header": false,
    "input_shape": [3, 200],
    "num_classes": 3,
    "class_names": ["class_a", "class_b", "class_c"],
    "transforms": { "train": "none", "val": "none" },
    "batch_size": 32,
    "num_workers": 4,
    "calibration": { "source": "reuse_val", "num_samples": 100 }
  },
  "model": {
    "layers": [
      { "type": "Conv1d", "in_channels": 3, "out_channels": 16, "kernel_size": 5, "padding": 2 },
      { "type": "ReLU" },
      { "type": "MaxPool1d", "kernel_size": 2 },
      { "type": "Conv1d", "in_channels": 16, "out_channels": 32, "kernel_size": 5, "padding": 2 },
      { "type": "ReLU" },
      { "type": "GlobalAvgPool1d" },
      { "type": "Linear", "in_features": 32, "out_features": 3 }
    ]
  },
  "training": {
    "epochs": 200,
    "optimizer": { "type": "Adam", "lr": 0.001, "weight_decay": 0.0001 },
    "lr_scheduler": { "type": "CosineAnnealingLR", "T_max": 30 },
    "loss": "cross_entropy",
    "metrics": ["accuracy"],
    "checkpoint": { "save_best": true, "monitor": "val_accuracy" },
    "device": "auto"
  },
  "quantization": {
    "enable": true,
    "method": "ptq",
    "dtype": "int8",
    "target": "esp32s3",
    "checkpoint": "best"
  }
}
```

## Dataset Rules Per Format

1D input semantics are always `[C, L]`:

- `C` = `num_features`
- `L` = sequence length

Tensor layout on the training side is `[N, C, L]`. See the ESP-DL layout note for deployment-side reshapes.

Delimiter is auto-detected per file (comma or whitespace) unless `delimiter` is set.

### `windowed_per_class_csv`

Directory layout: **one CSV per class**, file stem = class name.

```
<root>/<train_dir>/<class_a>.csv
<root>/<train_dir>/<class_b>.csv
<root>/<val_dir>/<class_a>.csv
...
```

Each row = one full sample with `num_features × L` columns.

Required:

- `format`, `root`, `num_features`

Advanced mode also required:

- `input_shape` = `[num_features, L]`

Auto-inferred in simple mode; checked against the first CSV row count in advanced mode.

### `windowed_regression_csv`

Directory layout: any `*.csv` under `train_dir` / `val_dir`. There is no per-class split. Each row contains feature columns + label columns.

Required:

- `format`, `root`, `num_features`

Advanced mode also required:

- `input_shape` = `[num_features, L]`
- `label_start_col`, `label_end_col` (inclusive range identifying the label columns within each row)
- `output_dim` (must equal `label_end_col - label_start_col + 1` when both cols are non-negative)

Optional:

- `target_normalize`: `{ "mean": ..., "std": ... }` — float or list matching `output_dim`

### `continuous_per_class_csv`

Directory layout: **one folder per class**, each folder holds one or more continuous-sequence CSVs.

```
<root>/<train_dir>/<class_a>/seq_0.csv
<root>/<train_dir>/<class_a>/seq_1.csv
<root>/<train_dir>/<class_b>/seq_0.csv
...
```

Each row in a file = one timestamped sample point. Training windows are cut with `window_size` / `window_stride`.

Required in **all** modes (simple mode does **not** default these):

- `window_size`, `window_stride`

Required:

- `format`, `root`, `num_features`, `window_size`, `window_stride`

Advanced mode also required:

- `input_shape` = `[num_features, window_size]` (the two must match)

Optional:

- `time_column` (int index or column name)
- `feature_columns` (list of column names when `has_header=true`)

## Common Dataset Fields

Shared across the three 1D formats:

- `train_dir`, `val_dir` — default `"train"` / `"test"`
- `has_header` — default `false`
- `delimiter` — default auto-detect (`,` or whitespace)
- `feature_layout` — `"interleaved"` (default) or `"grouped"`; only applies when multiple features share a window row
- `batch_size` — default `64`
- `num_workers` — default `4`
- `transforms` — see below
- `calibration` — see below

## Transforms

Goes inside `dataset.transforms`:

```json
"transforms": {
  "train": [{ "type": "Normalize" }],
  "val":   [{ "type": "Normalize" }]
}
```

Supported `type` values:

- `Normalize`
- `Rescale`
- `Identity`

Shorthand to disable per split:

```json
"transforms": { "train": "none", "val": "none" }
```

(Same effect as an empty list. `null` also works.)

`Normalize` without `mean` / `std`:

- Stats are computed from the train split.
- The same stats are reused for **train, val, calibration, and quantization**.
- Manual `{"mean": ..., "std": ...}` is accepted when you need to override.

## Calibration

```json
"calibration": { "source": "reuse_val", "num_samples": 256 }
```

- `source`: `"reuse_val"` (default) or `"reuse_train"`
- `num_samples`: positive int, default `256`
- The calibration dataloader always runs with `shuffle=false`.
- For `esp-dl`, one calibration sample is picked as the reference sample; its index is recorded as `test_input_index` and reused by `export`.

## Model

### Simple-mode presets (1D)

- `tiny_conv_1d` — Conv1d → ReLU → MaxPool1d → Conv1d → ReLU → GlobalAvgPool1d → Linear. Hidden / output channel widths depend on task type.
- `tiny_mlp_1d` — Flatten → Linear(32) → ReLU → Linear(output_dim).

Output dim comes from `dataset.num_classes` (classification) or `dataset.output_dim` (regression).

### Advanced-mode layer types (1D)

Unified DSL names:

- `Conv1d`, `Linear`
- `BatchNorm1d`
- `MaxPool1d`, `AvgPool1d`, `GlobalAvgPool1d`
- `Flatten`, `Dropout`
- `ReLU`, `ReLU6`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`

2D layers are rejected when `task.input_dim=1`.

## Training

### Advanced-mode required fields

- `epochs` (> 0)
- `optimizer` — e.g. `{ "type": "Adam", "lr": 0.001, "weight_decay": 0.0001 }`
- `loss` — unified name: `cross_entropy`, `mse`, `l1`, `huber`, etc.

### Advanced-mode optional fields

- `lr_scheduler` — e.g. `{ "type": "CosineAnnealingLR", "T_max": 30 }`
- `metrics` — default `["accuracy"]`
- `early_stopping` — `{ "patience": int, "monitor": "val_accuracy" }`
- `checkpoint` — `{ "save_best": true, "monitor": "val_accuracy" }`
- `device` — `"auto"` (default), `"cuda"`, `"cpu"`

### Checkpointing

Training always writes two checkpoints:

- `best_model.*` — chosen by `training.checkpoint.monitor`
- `last_model.*` — always the final epoch

## Quantization

Current public path: **PTQ, INT8**.

```json
"quantization": {
  "enable": true,
  "method": "ptq",
  "dtype": "int8",
  "target": "esp32s3",
  "checkpoint": "best"
}
```

Defaults (all fields optional in simple mode; whole block omittable):

- `enable`: `true`
- `method`: `"ptq"` (only accepted value)
- `dtype`: `"int8"` (only accepted value)
- `target`: for `esp-dl`, `"esp32s3"` (default) or `"esp32p4"`. For `tflite-micro`, this field is accepted but not chip-limited.
- `checkpoint`: `"best"`. Other accepted value: `"last"`.

The same checkpoint source is used both as the FP32 baseline and as the input to quantization. `quantization.dtype = "int8"` describes the quantization target, not the runtime tensor dtype of the deployed model — always read the actual input dtype from `.info` / runtime tensors on the device side.

For `tflite-micro`, quantization also runs a full validation-set eval on the `.tflite` model and emits a single random test sample (`test_input_index`, `test_output`, etc.).

## Export

`easyai export` reads existing quantized artifacts and writes a reference component under `output_dir/export/<component_name>/`:

Common outputs:

- `predict.h`, `predict.cpp`
- `test_data.h`, `test_data.cpp`
- `CMakeLists.txt`, `idf_component.yml`
- `export_report.json`
- `output_dir/export/main.cpp` (top-level helper main)

Backend-specific:

- `esp-dl`: `model.espdl` (embedded via component CMake)
- `tflite-micro`: `model.h`, `model.cpp` (C array)

The exported test sample is the **same index** as the quantize reference sample, with the raw float input, its ground-truth target, and the reference output.

This is a reference component, not a full ESP-IDF project generator.

### ESP-DL layout note (NCL → NLC)

Training / config semantics stay `[N, C, L]`, but `esp-dl` quantized/deployed input tensors may be `[N, L, C]`. The generated `predict.cpp` handles this:

- Normalize
- INT8 quantize if the deployed input tensor is INT8; otherwise pass FLOAT
- Reorder NCL → NLC when required by the deployed tensor

The layout decision must come from `.info` / the real runtime input shape, not from `input_shape` in the JSON.

### TFLite Micro layout note

Training / config semantics stay `[C, L]`; the deployed TFLite Micro model is fed `[L, C]`. `predict.cpp` handles the transpose internally; the exported `test_input` stays in `[C, L]`.

## Required-Fields Quick Reference

Always required (any mode):

- `backend`
- `task.input_dim`
- `task.type`
- `dataset.format`
- `dataset.root`
- `dataset.num_features`
- `training.epochs`

Required for classification:

- `dataset.num_classes` *(auto-inferred in simple mode from CSV file names / folder names)*
- `dataset.class_names` *(auto-inferred in simple mode)*

Required for regression:

- `dataset.output_dim` *(defaults to 1 in simple mode)*
- `dataset.label_start_col`, `dataset.label_end_col` *(default 0 in simple mode; write them explicitly if your label isn't a single scalar in column 0)*

Required for `continuous_per_class_csv` (no simple-mode default):

- `dataset.window_size`, `dataset.window_stride`

Required for simple mode:

- `model.preset`

Required for advanced mode:

- `model.layers`
- `training.optimizer`, `training.loss`
- `dataset.input_shape`

## Testing Guidance

When asked to smoke-test a full config:

1. `validate` first.
2. For broad sweeps, make a temporary config with:
   - `training.epochs = 1`
   - `dataset.num_workers = 0`
   - a temporary `output_dir`
3. Run `train` → `quantize` → `export`.

## Working Rules

- Prefer public, path-agnostic explanations; do not bake a developer's local paths into examples.
- Describe config semantics, not one developer's filesystem.
- Do not reintroduce 2D classification or YOLO detection. `schema.py` enforces `task.input_dim = 1` and only exposes `classification` / `regression` task types — any attempt to re-add them must be an explicit product decision, not inferred from older docs.
- Use `rg` for config and code search.
- The repository itself does not ship dataset or reference config files (`datasets/` is gitignored). The minimal JSON examples above are the source of truth for config shape.
