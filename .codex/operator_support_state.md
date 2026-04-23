# Operator Support State

这是 ESP-DL 算子支持状态的迁移版短文档。

## Purpose

这个文件只回答两个问题：

1. 当前 ESP-DL 的底层算子支持边界大概在哪里
2. 为什么它不能直接等同于用户可写的 `model.layers`

## Current Facts

- ESP-DL 当前支持的底层 ONNX operator 数量明显多于项目 DSL 暴露的 layer 数量
- DSL 是高层建模接口
- operator support 是部署后端能力边界
- 两者不是一一对应关系

## Current Project Rule

当前项目分三层理解：

- 用户层
  - `model.layers`
  - 只暴露适合用户写的高层 layer
- 框架层
  - `esp-dl -> torch`
  - `tflite-micro -> keras`
- 部署层
  - `torch/keras -> onnx/tflite`
  - 再对应到底层 operator support

## Current Boundary

当前阶段不做：

- 严格的 operator 白名单静态校验
- 芯片部署可行性强校验

当前阶段只做：

- 配置合法性
- 数据集合法性
- 模型定义可构建
- 训练
- PTQ INT8 量化

## Notes

- `BatchNorm1d/2d` 当前在 PC 侧模型定义是支持的
- 是否能完整落到部署侧，不在当前阶段承诺范围内

如果后续要做严格部署检查，应在这里继续补：

- DSL layer 到 ONNX op 的映射
- 各 op 的 target / dtype / restriction 规则
- 不支持项的错误提示
