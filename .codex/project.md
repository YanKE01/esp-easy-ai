# Project

## Position

`esp-easy-ai` 面向小白用户，目标是用一份 JSON 配置完成：

- 数据集检查
- 模型训练
- 模型量化
- 导出部署产物

当前项目不把“自由度最高的深度学习实验平台”作为目标，而是优先做：

- 简化配置
- 预设模型
- 一维数据主线
- `esp-dl` / `tflite-micro` 双后端统一入口

## Current Scope

当前主线仍是一维任务：

- 1D classification
- 1D regression
- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`

当前不支持：

- 2D classification
- detection 训练/量化/导出

当前已支持：

- `esp-easy-ai validate <config>`
- `esp-easy-ai train <config>`
- `esp-easy-ai quantize <config>`
- `esp-easy-ai export <config>`
- 小白模式自动补全
- backend 模型定义检查
- `esp-dl` 1D 训练
- `tflite-micro` 1D 训练
- `esp-dl` PTQ INT8 量化
- `tflite-micro` PTQ INT8 量化
- `esp-dl` 参考导出模板
- `tflite-micro` 参考导出模板

当前不作为主线：

- 严格部署侧算子白名单校验
- MCU 内存评估
- 量化精度阈值保护

## User Modes

同一套 JSON key，分两种使用方式：

### 小白模式

- `model.preset`
- `training.epochs`
- `dataset.root`
- `dataset.num_features`

系统自动补：

- dataset 格式与默认目录
- `input_shape`
- `num_classes` / `class_names` / `output_dim`
- `transforms`
- 默认 `training`
- 默认 `quantization`

### 高级模式

- `model.layers`
- 完整 `training`
- 完整 `quantization`

## Current Config Rules

### Dataset

1D 输入统一语义：

- shape: `[C, L]`
- `C = num_features`
- `L = sequence length`

训练与配置侧张量语义保持框架常用布局：

- 1D: `[N, C, L]`

ESP-DL 当前量化/导出链路下，部署侧输入 tensor 可能转换为：

- 1D: `[N, L, C]`

因此在 `esp-dl` 导出的端侧组件里：

- 1D 输入可能需要做 `NCL -> NLC`

当前 1D `esp-dl` 导出模板按 `NCL -> NLC` 处理。

这个转换属于部署侧适配，不改变训练侧与 JSON 配置里的输入语义。

支持的数据集：

- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`

其中：

- `windowed_*` 表示每一行已经是一个完整样本
- `continuous_per_class_csv` 表示每一行是一个时间点、每个文件是一段连续序列，训练时再按 `window_size` / `window_stride` 切窗

分隔符：

- 默认自动检测
- 逗号或空白分隔

transforms：

- `Normalize` 默认从 train 集自动统计
- 不要求用户手写 `mean/std`
- 同一组统计量同时用于 `train`、`val`、`calibration`、`quantize`

### Model

当前统一 DSL 已支持的主线层类型：

- `Conv1d`
- `Linear`
- `BatchNorm1d`
- `MaxPool1d`
- `AvgPool1d`
- `GlobalAvgPool1d`
- `Flatten`
- `Dropout`
- `ReLU`
- `ReLU6`
- `LeakyReLU`
- `Sigmoid`
- `Tanh`
- `Softmax`

当前小白预设：

- `tiny_conv_1d`
- `tiny_mlp_1d`

### Training

训练前总是先做完整 `validate`。

训练阶段始终保存两份 checkpoint：

- `best_model.*`
- `last_model.*`

其中：

- `best_model.*` 由 `training.checkpoint.monitor` 决定
- `last_model.*` 始终表示最后一个 epoch 的模型

### Quantization

当前只支持：

- `PTQ`
- `INT8`

当前高级模式统一量化字段：

```json
"quantization": {
  "enable": true,
  "method": "ptq",
  "dtype": "int8",
  "target": "esp32s3",
  "checkpoint": "best"
}
```

当前量化支持：

- `esp-dl`
- `tflite-micro`

checkpoint 选择：

- `quantization.checkpoint = "best" | "last"`

校准规则：

- 使用 `dataset.calibration`
- calibration dataloader 必须 `shuffle=false`
- `esp-dl` 量化会使用真实校准样本作为 `inputs`
- `esp-dl` 量化会随机选择一条 calibration 样本作为 reference sample
- `esp-dl` 量化会导出 test input / test output
- `esp-dl` 量化模型的实际输入 tensor dtype 不应只根据 `quantization.dtype` 推断
- `quantization.dtype = "int8"` 表示量化目标是 `INT8`，不等于模型输入接口一定是 `INT8`
- 对端侧接入与 export，应以 `.info` 和 runtime tensor dtype 为准
- `esp-dl export` 会复用测试报告里的 `test_input_index`
- 因此 export 的测试数据和 quantize 的 reference sample 是同一条样本，只是 export 导出为端侧更易使用的表示形式
- `tflite-micro` 量化后会自动用验证集评测 `.tflite` 模型
- `tflite-micro` 量化会额外输出单条测试样本的推理结果

### Export

当前 `export` 支持：

- `esp-dl`
- `tflite-micro`

会基于现有量化产物生成参考文件：

- `CMakeLists.txt`
- `idf_component.yml`
- `model.h`
- `model.cpp`
- `predict.h`
- `predict.cpp`
- `test_data.h`
- `test_data.cpp`
- `model.espdl`（仅 `esp-dl`）
- `main.cpp`

导出模板当前还会附带：

- 一条和量化参考样本同 index 的原始 float 测试输入
- 该样本的真实标签/目标
- 该样本在 `.info` 中对应的 reference output

这些文件放在：

- `output_dir/export/<component_name>/`

当前定位：

- 参考模板
- `esp-dl` 直接携带 `model.espdl` 并通过组件 CMake 嵌入
- `tflite-micro` 直接生成 `model.cpp` / `model.h` 数组文件
- `predict.cpp` 在端侧按模型真实输入 dtype 完成原始输入的预处理、必要时的 INT8 量化和布局转换
- 不生成完整 ESP-IDF 工程
- export 代码按 backend 拆分组织
- 导出源码模板使用 `.tpl` 文件维护

## Current Backend Status

### `esp-dl`

- 训练框架：PyTorch
- 量化工具：ESP-PPQ
- 当前已支持：
  - 1D 模型定义检查
  - 1D train
  - PTQ INT8 quantize
  - 组件形式 export
- 部署时需要注意输入布局可能从训练侧 `NCL` 转为 `NLC`

### `tflite-micro`

- 训练框架：Keras
- 量化工具：TFLite converter
- 当前已支持：
  - 1D 模型定义检查
  - 1D train
  - PTQ INT8 quantize

## Current Examples

仓库本体不包含数据集与示例配置；`datasets/` 被 `.gitignore` 屏蔽。

README（英文 / 中文）里各给了一份最小 simple 示例和一份完整 advanced 示例，作为结构参考；使用者应按自己的数据集自行写配置。

## Next Priorities

1. 回退后重新完整测试当前功能链路
2. 继续稳固一维主线
2. 继续完善小白模式 preset 体系
3. 补更多正式示例配置
4. 再考虑导出与部署侧细节
