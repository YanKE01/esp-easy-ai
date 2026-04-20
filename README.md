# esp-easy-ai

`esp-easy-ai` provides a JSON-driven workflow for training, quantizing, and exporting AI models for ESP devices.

## Current Scope

The current release focuses on 1D tasks and supports:

- `esp-dl`
- `tflite-micro`
- training
- PTQ INT8 quantization
- component-style export

## Dataset Types

The current release supports two 1D dataset families:

- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`

In this project:

- **windowed** means each row already represents one complete sample
- **continuous** means each row is one timestamped sample point, and training windows are cut from longer sequences

## ESP-DL Input Layout Note

For model configuration and training, the project uses the usual framework-side layouts:

- 1D: `NCL`
- 2D: `NCHW`

For the current ESP-DL quantization and export pipeline, the deployed tensor layout may become:

- 1D: `NLC`
- 2D: `NHWC`

For the current 1D ESP-DL export template, the deployed input layout is treated as `NLC`.

This means an exported ESP-DL runtime component may need to reorder input data before inference. In other words:

- 1D inputs may require `NCL -> NLC`
- 2D inputs may require `NCHW -> NHWC`

The JSON config does not change because of this. The conversion is a deployment-side adaptation step handled by the ESP-DL runtime/export path.

## TFLite Micro Input Layout Note

For the current 1D TFLite Micro path, training still uses the project-side semantic layout `[C, L]`, while the deployed runtime path feeds the model as `[L, C]`.

The exported TFLite Micro runtime component therefore keeps `test_input` in raw `[C, L]` semantics and performs preprocess plus layout adaptation inside `predict.cpp`.

## Commands

```bash
uv run easyai validate <config.json>
uv run easyai train <config.json>
uv run easyai quantize <config.json>
uv run easyai export <config.json>
```
