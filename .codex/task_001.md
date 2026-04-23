# Task 001

## Theme

聚焦一维数据流程，把一维链路做成项目当前主路径。

## Why

当前项目面向小白用户，最适合先做扎实的是：

- 一维分类
- 一维回归
- 简化配置
- 预设模型

而不是一开始就把二维任务、检测任务和部署侧细节同时展开。

## In Scope

- `task.input_dim = 1`
- `classification`
- `regression`
- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`
- `esp-dl -> torch`
- `tflite-micro -> keras`
- 小白模式 `preset + epochs`
- 高级模式 `layers + 完整 training/quantization`

## Achieved

当前这一主任务已经落地的内容：

- 配置 schema 校验
- 数据集文件校验
- 小白模式自动补全
- backend 模型定义检查
- dummy forward
- 1D train
- PTQ INT8 quantize
- `best/last` checkpoint 双保存与选择
- `esp-dl` 组件形式 export

## Current Boundary

当前不做：

- detection 训练/量化/导出
- 严格部署侧算子白名单校验
- ONNX / TFLite 导出兼容性静态分析
- MCU 内存评估

## Current Dataset Boundary

当前代码支持两类 1D CSV 数据：

- `windowed_*`
  - 每行一个完整样本
- `continuous_per_class_csv`
  - 每行一个时间点
  - 每个文件是一段连续序列
  - 训练时按 `window_size` / `window_stride` 切窗

## Runtime Rules

- 训练总是保存 `best_model.*` 和 `last_model.*`
- 量化可选 `best` 或 `last`
- calibration `shuffle=false`
- `esp-dl` 量化导出真实 test input / test output
- `Normalize` 统计量只从 train 集计算
- 同一组 `Normalize` 统计量会复用于 `train`、`val`、`calibration`、`quantize`

## Next Step

在 Task 001 范围内，下一步优先考虑：

1. 回退后从头完整测试当前一维链路
2. 继续稳固一维数据的示例与默认体验
3. 扩充和整理 preset
4. 再决定哪些二维能力进入下一阶段

当前完整测试应覆盖：

- `windowed_per_class_csv`
- `windowed_regression_csv`
- `continuous_per_class_csv`
- `esp-dl`
- `tflite-micro`
- advanced config
- simple config
