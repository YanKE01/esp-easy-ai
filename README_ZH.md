# esp-easy-ai

[English](./README.md) | **中文**

`esp-easy-ai` 提供一套以 JSON 配置为核心的工作流，面向 ESP 设备端的 1D 模型：

- 数据集检查
- 训练
- PTQ INT8 量化
- 端侧组件 / ESP-IDF 工程模板导出

当前只支持 **1D** 任务。不包含 2D 分类、YOLO 检测。

## Supported Matrix

| Task                 | Dataset formats                                       | Backends                    |
|----------------------|-------------------------------------------------------|-----------------------------|
| 1D classification    | `windowed_per_class_csv`, `continuous_per_class_csv`  | `esp-dl`, `tflite-micro`    |
| 1D regression        | `windowed_regression_csv`                             | `esp-dl`, `tflite-micro`    |

- `esp-dl`：PyTorch + ESP-PPQ
- `tflite-micro`：Keras + TFLite converter
- 量化当前仅支持 **PTQ + INT8**
- `esp-dl` 的量化 target 可选 `esp32s3`（默认）、`esp32p4`
- `tflite-micro` 的 `quantization.target` 允许填写，但不做芯片值限制

## Dataset Formats

- **windowed_per_class_csv** — 每行一个完整样本；每个类一份 CSV，文件名即类名。布局：
  ```
  <root>/<train_dir>/<class>.csv
  <root>/<val_dir>/<class>.csv
  ```
- **windowed_regression_csv** — 每行一个完整样本，特征列 + 标签列共行。`train_dir` / `val_dir` 下的任意 CSV 都会被读入。
- **continuous_per_class_csv** — 每行是一个时间点；每个类一个目录，目录内可有多段连续序列。训练时按 `window_size` / `window_stride` 切窗。布局：
  ```
  <root>/<train_dir>/<class>/seq_*.csv
  ```

1D 输入语义统一为 `[C, L]`：`C = num_features`，`L = 序列长度`。

## Two Authoring Styles

同一套 JSON key，两种写法：

### Simple mode

用预设模型 + 一个 dataset 三元组，其它全部自动补齐：

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

系统会自动补：`project_name` / `output_dir` / `seed` / `input_shape` / `class_names` / `num_classes` / 默认 transforms、calibration、batch_size、num_workers、完整 training（Adam + CosineAnnealingLR + cross_entropy/mse）、完整 quantization。

当前预设：`tiny_conv_1d`、`tiny_mlp_1d`。

### Advanced mode

用 `model.layers` 写明完整层栈，`training` 写完整 optimizer/scheduler/loss/checkpoint：

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

#### 字段说明

以下列出示例里各 `type` / 字符串字段的实际可选值。

**dataset**

- `train_dir` / `val_dir` —— 相对 `root` 的子目录名，默认 `"train"` / `"test"`
- `feature_layout` —— `"interleaved"`（默认）或 `"grouped"`，描述多特征列在一行窗口内的排布方式
- `has_header` —— 首行是否为表头，默认 `false`
- `delimiter` —— `","` 或 `"whitespace"`，未写则按文件内容自动检测
- `batch_size`、`num_workers` —— DataLoader 参数，默认 `64` / `4`
- `transforms` —— 见下方
- `calibration.source` —— `"reuse_val"`（默认）或 `"reuse_train"`

**transforms**

`dataset.transforms` 按 split 分别给一个变换列表：

```json
"transforms": {
  "train": [{ "type": "Normalize" }],
  "val":   [{ "type": "Normalize" }]
}
```

关闭某个 split 的快捷写法：`"none"` / `null` / `[]`。

支持的 `type`：

- `Normalize` —— 不写 `mean` / `std` 时，自动从 train 集统计，同一组统计量会被 train / val / calibration / quantize 共用；也接受手写 `{ "mean": ..., "std": ... }`
- `Rescale`
- `Identity`

**training**

- `optimizer.type` —— `Adam` 或 `SGD`，额外字段（`lr`、`weight_decay`、`momentum` 等）会透传给后端优化器构造函数
- `lr_scheduler.type` —— `CosineAnnealingLR` 或 `StepLR`，额外字段（`T_max`、`step_size`、`gamma` 等）会透传
- `loss` —— `cross_entropy`、`mse`、`l1`、`huber`
- `metrics` —— `accuracy`、`mae`、`rmse` 的任意组合
- `checkpoint.monitor` —— 带 `val_` 前缀的指标名，例如 `val_accuracy`、`val_mae`
- `device` —— `auto`（默认）、`cuda`、`cpu`

**quantization**

- `method` —— 仅支持 `ptq`
- `dtype` —— 仅支持 `int8`
- `target` —— 当 `backend="esp-dl"` 时，必须是 `esp32s3` 或 `esp32p4`；对 `tflite-micro` 不做芯片值限制
- `checkpoint` —— `best`（默认）或 `last`，选定哪个训练 checkpoint 进入量化

**model.layers**

支持的 `type`：
`Conv1d`、`Linear`、`BatchNorm1d`、`MaxPool1d`、`AvgPool1d`、`GlobalAvgPool1d`、`Flatten`、`Dropout`、`ReLU`、`ReLU6`、`LeakyReLU`、`Sigmoid`、`Tanh`、`Softmax`。

每一层上除 `type` 外的键会作为构造参数透传，例如 `in_channels` / `out_channels` / `kernel_size` / `padding` / `stride`。

## Commands

```bash
uv run esp-easy-ai validate <config.json>
uv run esp-easy-ai train    <config.json>
uv run esp-easy-ai quantize <config.json>
uv run esp-easy-ai export   <config.json>
```

- `train` / `quantize` / `export` 都会先跑一次完整 `validate`
- `quantize` 不会重新训练，只读已有 checkpoint
- `export` 不会重新训练也不会重新量化，只读已有量化产物

## Outputs

训练产物（`<output_dir>/`）：

- `best_model.*`（由 `training.checkpoint.monitor` 选出）
- `last_model.*`（最后一个 epoch）
- `report.json`

量化产物：

- `esp-dl`：`model_int8.espdl`、`espdl_test_report.json`
- `tflite-micro`：`model_int8.tflite`、`tflite_test_report.json`

校准使用 `dataset.calibration` 指定的样本源，`shuffle=false`。`esp-dl` 会随机选一条作为 reference sample，并记下 `test_input_index`，后续 `export` 导出的测试样本会使用同一条。

## Export

`esp-easy-ai export` 生成一份参考 **ESP-IDF 组件** + 一份配套 **ESP-IDF 工程模板**，位于 `<output_dir>/export/`：

组件目录（`<output_dir>/export/<component_name>/`）：

- `predict.h` / `predict.cpp`
- `test_data.h` / `test_data.cpp`
- `CMakeLists.txt`、`idf_component.yml`
- `export_report.json`
- `esp-dl` 下附：`model.espdl`（组件 CMake 直接嵌入）
- `tflite-micro` 下附：`model.h` / `model.cpp`（C 数组）

工程模板（`<output_dir>/export/`）：

- `CMakeLists.txt`、`sdkconfig.defaults`
- `main/CMakeLists.txt`、`main/idf_component.yml`、`main/main.cpp`
- `esp-dl` 专属：`partitions.csv`（8 MB factory 分区；默认假设 16 MB flash + PSRAM，对应 S3 / P4 常见模组）
- `tflite-micro`：不生成 `partitions.csv`，也不锁定 flash 大小 / PSRAM，IDF 会走 2 MB 默认——C3 / C6 / H2 / S2 / S3 / P4 都能直接 build

`predict.cpp` 会在端侧执行：Normalize → 按部署 tensor 的真实 dtype 做 INT8 量化或保留 FLOAT → 按需做布局重排。

## Input Layout Notes

### ESP-DL (`NCL → NLC`)

训练 / 配置侧统一按 `[N, C, L]` 理解输入；`esp-dl` 量化后的部署 tensor 可能是 `[N, L, C]`。当前 1D 导出模板默认按 `NCL → NLC` 处理。

JSON 配置不需要因此改变，布局决策以量化产物里的 runtime tensor 为准，而不是 JSON 里的 `input_shape`。

同时，`quantization.dtype = "int8"` 描述的是量化目标，不等于部署 tensor 的输入 dtype。端侧应以 `.info` 和 runtime tensor dtype 为准——模板会在输入为 INT8 时自动做 Normalize→量化，为 FLOAT 时直接写 FLOAT。

### TFLite Micro (`[C, L] → [L, C]`)

训练 / 配置语义保持 `[C, L]`；部署时模型接收 `[L, C]`。导出的 `test_input` 仍保持原始 `[C, L]`，布局转换发生在 `predict.cpp` 内部。
