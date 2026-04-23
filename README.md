# esp-easy-ai

**English** | [中文](./README_ZH.md)

`esp-easy-ai` is a JSON-driven workflow for building 1D AI models that run on ESP devices:

- dataset validation
- training
- PTQ INT8 quantization
- export as an ESP-IDF component + project template

The current release targets **1D only**. 2D classification and YOLO detection are not in scope.

## Supported Matrix

| Task                 | Dataset formats                                       | Backends                    |
|----------------------|-------------------------------------------------------|-----------------------------|
| 1D classification    | `windowed_per_class_csv`, `continuous_per_class_csv`  | `esp-dl`, `tflite-micro`    |
| 1D regression        | `windowed_regression_csv`                             | `esp-dl`, `tflite-micro`    |

- `esp-dl`: PyTorch + ESP-PPQ
- `tflite-micro`: Keras + TFLite converter
- Quantization is currently **PTQ + INT8** only
- For `esp-dl`, quantization `target` can be `esp32s3` (default) or `esp32p4`
- For `tflite-micro`, `quantization.target` is accepted but not chip-limited

## Dataset Formats

- **windowed_per_class_csv** — one row = one full sample; one CSV per class, the file stem is the class name.
  ```
  <root>/<train_dir>/<class>.csv
  <root>/<val_dir>/<class>.csv
  ```
- **windowed_regression_csv** — one row = one full sample; feature columns and label columns share the row. Any CSV under `train_dir` / `val_dir` is picked up.
- **continuous_per_class_csv** — one row = one timestamped point; one folder per class, each folder may hold multiple continuous-sequence CSVs. Training windows are cut with `window_size` / `window_stride`.
  ```
  <root>/<train_dir>/<class>/seq_*.csv
  ```

The 1D input semantic is always `[C, L]`: `C = num_features`, `L = sequence length`.

## Two Authoring Styles

Same top-level JSON keys, two ways of filling them in.

### Simple mode

A preset model plus a three-field `dataset` block; everything else is auto-filled:

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

Auto-filled in simple mode: `project_name`, `output_dir`, `seed`, `input_shape`, `class_names`, `num_classes`, default transforms / calibration / `batch_size` / `num_workers`, a full `training` block (Adam + CosineAnnealingLR + `cross_entropy` / `mse`), and a full `quantization` block.

Current presets: `tiny_conv_1d`, `tiny_mlp_1d`.

### Advanced mode

Write the full layer stack under `model.layers` and a complete `training` block (optimizer / scheduler / loss / checkpoint).

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

#### Field reference

Values accepted by each `type` / string field in the example above.

**dataset**

- `train_dir` / `val_dir` — subdirectory names under `root`. Defaults: `"train"` / `"test"`.
- `feature_layout` — `"interleaved"` (default) or `"grouped"`. Describes how multi-feature columns are arranged inside one windowed row.
- `has_header` — whether the first CSV row is a header. Default `false`.
- `delimiter` — `","` / `"whitespace"`. Auto-detected per file when omitted.
- `batch_size`, `num_workers` — dataloader knobs. Defaults `64` / `4`.
- `transforms` — see below.
- `calibration.source` — `"reuse_val"` (default) or `"reuse_train"`.

**transforms**

`dataset.transforms` takes per-split lists of transform items:

```json
"transforms": {
  "train": [{ "type": "Normalize" }],
  "val":   [{ "type": "Normalize" }]
}
```

Shorthand to disable a split: `"none"`, `null`, or `[]`.

Supported `type` values:

- `Normalize` — if `mean` / `std` are omitted, stats are computed from the train split and reused for train / val / calibration / quantization. Manual `{ "mean": ..., "std": ... }` is also accepted.
- `Rescale`
- `Identity`

**training**

- `optimizer.type` — `Adam` or `SGD`. Any extra keys (e.g. `lr`, `weight_decay`, `momentum`) are forwarded to the backend's optimizer constructor.
- `lr_scheduler.type` — `CosineAnnealingLR` or `StepLR`. Extra keys (e.g. `T_max`, `step_size`, `gamma`) are forwarded.
- `loss` — `cross_entropy`, `mse`, `l1`, or `huber`.
- `metrics` — any combination of `accuracy`, `mae`, `rmse`.
- `checkpoint.monitor` — a metric key prefixed with `val_`, e.g. `val_accuracy`, `val_mae`.
- `device` — `auto` (default), `cuda`, or `cpu`.

**quantization**

- `method` — `ptq` (only supported value).
- `dtype` — `int8` (only supported value).
- `target` — when `backend="esp-dl"`, must be `esp32s3` or `esp32p4`. For `tflite-micro`, this field is accepted but not chip-limited.
- `checkpoint` — `best` (default) or `last`. Selects which training checkpoint is fed into quantization.

**model.layers**

Supported `type` values:
`Conv1d`, `Linear`, `BatchNorm1d`, `MaxPool1d`, `AvgPool1d`, `GlobalAvgPool1d`, `Flatten`, `Dropout`, `ReLU`, `ReLU6`, `LeakyReLU`, `Sigmoid`, `Tanh`, `Softmax`.

Extra keys on each layer are forwarded as constructor kwargs (e.g. `in_channels`, `out_channels`, `kernel_size`, `padding`, `stride`).

## Commands

```bash
uv run esp-easy-ai validate <config.json>
uv run esp-easy-ai train    <config.json>
uv run esp-easy-ai quantize <config.json>
uv run esp-easy-ai export   <config.json>
```

- `train` / `quantize` / `export` all run a full `validate` first.
- `quantize` does not retrain; it reads existing checkpoints.
- `export` neither retrains nor requantizes; it reads existing quantization artifacts.

## Outputs

Training artifacts (`<output_dir>/`):

- `best_model.*` (selected by `training.checkpoint.monitor`)
- `last_model.*` (final epoch)
- `report.json`

Quantization artifacts:

- `esp-dl`: `model_int8.espdl`, `espdl_test_report.json`
- `tflite-micro`: `model_int8.tflite`, `tflite_test_report.json`

Calibration uses the sample source configured in `dataset.calibration` with `shuffle=false`. For `esp-dl`, one calibration sample is randomly picked as the reference sample and recorded as `test_input_index`; the later `export` step reuses the same index.

## Export

`esp-easy-ai export` generates a reference **ESP-IDF component** plus a companion **ESP-IDF project template** under `<output_dir>/export/`:

Component directory (`<output_dir>/export/<component_name>/`):

- `predict.h` / `predict.cpp`
- `test_data.h` / `test_data.cpp`
- `CMakeLists.txt`, `idf_component.yml`
- `export_report.json`
- `esp-dl`: `model.espdl` (embedded by the component's CMakeLists)
- `tflite-micro`: `model.h` / `model.cpp` (C arrays)

Project template (`<output_dir>/export/`):

- `CMakeLists.txt`, `sdkconfig.defaults`
- `main/CMakeLists.txt`, `main/idf_component.yml`, `main/main.cpp`
- `esp-dl` only: `partitions.csv` (8 MB factory partition; assumes 16 MB flash + PSRAM, matching typical S3 / P4 modules)
- `tflite-micro`: no `partitions.csv`, no flash-size / PSRAM defaults — IDF picks its own 2 MB-safe defaults so the template builds on C3 / C6 / H2 / S2 / S3 / P4 alike

`predict.cpp` runs on-device: Normalize → INT8-quantize or pass-through FLOAT based on the deployed tensor's real dtype → reorder layout when needed.

## Input Layout Notes

### ESP-DL (`NCL → NLC`)

Training / config semantics stay `[N, C, L]`; the `esp-dl` quantized deployed tensor may be `[N, L, C]`. The current 1D export template treats the deployed layout as `NLC` by default.

The JSON config does not change because of this. Layout decisions should follow the runtime tensor in the quantization artifacts, not `input_shape` from the JSON.

In addition, `quantization.dtype = "int8"` describes the quantization target, not the runtime input dtype of the deployed model. Always read the actual input dtype from `.info` / runtime tensors on the device side — the template branches automatically: INT8 input → Normalize + quantize; FLOAT input → Normalize + write FLOAT.

### TFLite Micro (`[C, L] → [L, C]`)

Training / config semantics stay `[C, L]`; the deployed TFLite Micro model is fed `[L, C]`. The exported `test_input` keeps the raw `[C, L]` semantics and the transpose happens inside `predict.cpp`.
