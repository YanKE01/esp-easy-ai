# 时序训练计划

## 1. 样本理解

当前动作识别场景默认采用：

- 一次完整采集 = 一个序列
- 一个序列 = 一个样本
- 一个样本 = 一个标签

例如：

- 动作 `a` 和 `b` 各采集 5 次
- 总共得到 10 个样本

## 2. 端侧推理输入

端侧推理采用固定长度输入。

例如：

- 每 `1ms` 采集一次三轴加速度
- 连续采集 `1000` 个时刻
- 形成一个 `1000 x 3` 的输入样本
- 执行一次完整推理

训练输入和推理输入长度必须一致。

## 3. 模型方式

时序模型采用两层方式：

1. 系统提供预设模型
2. 用户在预设基础上调整参数

## 4. 预设模型

当前预设模型包括：

- `Fully Connected Network`
- `1D CNN`
- `CNN + MLP`
- `LSTM`

## 5. 用户可调参数

当前建议开放：

- 层数
- hidden size / channel size
- kernel size
- dropout
- activation
- batch size
- epoch
- learning rate

## 6. Training 页面结构

`Time Series -> Training` 页面包含：

1. `Task Type`
2. `Model Preset`
3. `Model Config`
4. `Training Config`
5. `Run Training`

## 7. Training 页面流程

1. 选择任务类型
2. 选择预设模型
3. 调整模型参数
4. 调整训练参数
5. 提交训练请求
6. 展示训练结果

## 8. 训练请求结构

时序训练请求中的 `config` 建议包含：

```json
{
  "model": {
    "preset": "cnn1d",
    "params": {}
  },
  "training": {}
}
```
