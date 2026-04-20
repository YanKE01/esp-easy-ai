# Workflows

## Validate

命令：

```bash
uv run easyai validate <config.json>
```

当前会做：

- JSON 读取
- schema 校验
- 数据集文件校验
- backend 模型定义检查
- dummy forward

当前支持的 dataset format：

- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`

## Train

命令：

```bash
uv run easyai train <config.json>
```

当前行为：

- 先执行完整 `validate`
- 再开始训练
- 训练阶段始终保存两份 checkpoint
  - `best_model.*`
  - `last_model.*`
- 输出 `report.json`

`best_model.*`：

- 由 `training.checkpoint.monitor` 决定

`last_model.*`：

- 始终表示最后一个 epoch 的模型

CLI 当前会打印：

- `best_epoch`
- `monitor`
- `best_value`
- `best_ckpt`
- `last_ckpt`

## Quantize

命令：

```bash
uv run easyai quantize <config.json>
```

当前行为：

- 不会重新训练
- 先执行完整 `validate`
- 再读取已有 checkpoint 做量化
- 输出 `quantization_report.json`
- `tflite-micro` 量化完成后会加载 `.tflite` 模型并在验证集上跑一遍评测

量化 checkpoint 选择：

```json
"quantization": {
  "checkpoint": "best"
}
```

可选值：

- `best`
- `last`

默认是 `best`

CLI 当前会打印：

- `checkpoint_source`
- 实际使用的 `checkpoint`
- calibration 信息
- `esp-dl` 下的 test input shape / target
- `tflite-micro` 下的验证集评测指标
  - 分类：`eval_accuracy`
  - 回归：`eval_mae` / `eval_rmse`
- `tflite-micro` 下的单条随机测试样本信息
  - `test_input_index`
  - `test_input_target`
  - `test_output`
  - `test_predicted_label`（classification）

## Calibration Rules

- 校准数据来自 `dataset.calibration`
- calibration dataloader 必须 `shuffle=false`
- `esp-dl` 量化时：
  - 会随机取一条真实校准样本作为 `inputs`
  - 开启 `export_test_values=true`
  - 导出的 `.info` 文件中可以查看 test input / test output
  - 端侧接入时，输入 tensor dtype 需要以 `.info` / runtime model input dtype 为准
  - `quantization_report.json` 会记录这条样本的 `test_input_index`
  - 后续 `esp-dl export` 会复用同一个 `test_input_index`
  - 因此 export 的测试数据和 quantize 的参考样本是同一条数据

## ESP-DL Layout Note

对 `esp-dl`，训练侧与配置侧仍按常用布局理解输入：

- 1D: `NCL`
- 2D: `NCHW`

但当前量化/导出后的部署 tensor 可能表现为：

- 1D: `NLC`
- 2D: `NHWC`

因此 `esp-dl` 导出的 `predict.cpp` 需要按部署模型真实输入 shape 做重排。判断依据应以量化产物和 `.info` 中的输入 shape 为准，而不是仅看训练侧 `input_shape`。

对当前 1D `esp-dl` 导出模板，默认按 `NCL -> NLC` 处理，并会在导出的 `predict.cpp` 注释中明确写出这一点。

另外，`esp-dl` 导出模板不能假定所有量化模型都接收 `INT8` 输入。当前模板会按模型真实输入 dtype 分支：

- 输入 tensor 为 `INT8`：端侧先 `Normalize`，再量化成 `INT8`
- 输入 tensor 为 `FLOAT`：端侧先 `Normalize`，再直接写入 `FLOAT`

## Normalize

当前 `Normalize` 支持自动统计，不需要用户填写 `mean/std`：

```json
"transforms": {
  "train": [{"type": "Normalize"}],
  "val": [{"type": "Normalize"}]
}
```

规则：

- 统计量只从 `train` 集计算
- 同一组统计量同时用于 `train`、`val`、`calibration`、`quantize`
- 如果配置里显式写了 `mean/std`，运行时也兼容，但默认推荐不写

## Quantization Config

当前高级模式统一使用同一套量化字段：

```json
"quantization": {
  "enable": true,
  "method": "ptq",
  "dtype": "int8",
  "target": "esp32s3",
  "checkpoint": "best"
}
```

说明：

- 当前只支持 `PTQ + INT8`
- `target` 对两个 backend 都保留
- calibration 参数统一放在 `dataset.calibration`

## Small Config Defaults

小白模式下，如果不写完整量化字段，系统会自动补默认值：

- `enable=true`
- `method=ptq`
- `dtype=int8`
- `target=esp32s3`
- `checkpoint=best`

## Output Files

训练后常见产物：

- `report.json`
- `best_model.pt` / `best_model.keras`
- `last_model.pt` / `last_model.keras`

量化后常见产物：

- `quantization_report.json`
- `model_int8.espdl`
- `model_int8.tflite`
- `tflite_test_report.json`（仅 `tflite-micro`）

## Export

命令：

```bash
uv run easyai export <config.json>
```

当前行为：

- 不会重新训练
- 不会重新量化
- 先执行完整 `validate`
- 读取已有量化产物

当前会基于量化产物生成：

- `output_dir/export/<component_name>/CMakeLists.txt`
- `output_dir/export/<component_name>/idf_component.yml`
- `output_dir/export/<component_name>/model.h`（仅 `tflite-micro`）
- `output_dir/export/<component_name>/model.cpp`（仅 `tflite-micro`）
- `output_dir/export/<component_name>/predict.h`
- `output_dir/export/<component_name>/predict.cpp`
- `output_dir/export/<component_name>/test_data.h`
- `output_dir/export/<component_name>/test_data.cpp`
- `output_dir/export/<component_name>/model.espdl`（仅 `esp-dl`）
- `output_dir/export/<component_name>/export_report.json`
- `output_dir/export/main.cpp`

说明：

- 以 ESP-IDF 组件目录形式导出
- 当前支持 `esp-dl` 和 `tflite-micro` export
- export 实现按 backend 分文件
- 组件源码通过 `.tpl` 模板渲染生成
- `esp-dl` 下组件会直接嵌入 `model.espdl`
- `tflite-micro` 下组件会直接生成 `model.cpp` / `model.h`
- 会导出一条和量化参考样本同 index 的测试样本
  - `test_input`
  - `test_target`
  - `reference_output`
- `predict.cpp` 会在端侧执行：
  - `Normalize`
  - INT8 quantize
  - 必要时按部署 tensor layout 做重排
- 当前是参考模板，不是完整工程生成器

## TFLite Micro Layout Note

对当前 1D `tflite-micro` 路径：

- 训练/配置语义仍然按 `[C, L]`
- 导出组件内部按 `[L, C]` 喂给 TFLite Micro 模型

因此导出的 `test_input` 仍保持原始 `[C, L]` 语义，布局适配发生在 `predict.cpp` 内部。

## Current Test Priority

当前下一步优先级不是扩新功能，而是完整测试现有链路：

1. `validate`
2. `train`
3. `quantize`
4. `export`

需要覆盖：

- `esp-dl`
- `tflite-micro`
- advanced config
- simple config
- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`
