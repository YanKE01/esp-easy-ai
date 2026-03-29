## ADDED Requirements

### Requirement: 训练 API 必须使用统一的任务请求模型
训练后端必须接受统一的逻辑请求结构，其中包含任务标识、数据集定义、模型配置和训练配置，而不因任务类型不同而改变整体模型。

#### Scenario: 前端提交 1D 训练请求
- **WHEN** 用户启动 1D 任务训练
- **THEN** 前端发送的请求对象中包含任务标识、数据集信息、模型预设信息和训练参数

#### Scenario: 分类原始类别文件在训练启动前补充 split
- **WHEN** 用户为 `1D Data Classification` 的原始类别文件数据集点击 `Train`
- **THEN** 前端先弹出 split 配置，待用户确认后再组装训练请求

#### Scenario: 分类 MLP 允许编辑 hidden dense layers
- **WHEN** 用户在 `1D Data Classification` 训练页选择 `MLP`
- **THEN** 前端允许用户调整 hidden dense layers 的神经元数量，并在倒数第二层提供追加 dense layer 的入口

#### Scenario: 分类训练页展示输出层类别数
- **WHEN** 用户在 `1D Data Classification` 训练页查看 `MLP Structure`
- **THEN** 前端展示 `Output Layer`，并直接使用 `Dataset` 阶段识别到的类别数量作为输出类别数

#### Scenario: 分类训练页选择输入归一化策略
- **WHEN** 用户在 `1D Data Classification` 的 `Training Parameters` 中设置 `Input Normalization`
- **THEN** 前端允许用户在 `None`、`Z-Score`、`Min-Max` 中选择，并将该值作为训练数据预处理策略传递给后端

#### Scenario: Z-Score 归一化公式固定
- **WHEN** 用户选择 `Z-Score` 作为输入归一化策略
- **THEN** 平台按 `x' = (x - mean) / std` 解释该策略，并要求训练与推理阶段复用相同的训练集统计量

#### Scenario: 前端提交图像训练请求
- **WHEN** 用户启动图像任务训练
- **THEN** 前端使用相同逻辑结构发送请求，并填充图像任务对应的数据集与模型字段

### Requirement: 平台必须按任务类型路由训练执行
后端必须根据所选任务类型将每个训练请求分发到正确的训练实现。

#### Scenario: 时序分类请求路由
- **WHEN** 某个请求声明 `task_type` 为 `timeseries_classification`
- **THEN** 后端调用时序训练流水线，而不是通用或图像流水线

#### Scenario: 目标检测请求路由
- **WHEN** 某个请求声明 `task_type` 为 `object_detection`
- **THEN** 后端调用目标检测训练流水线

### Requirement: 平台必须以统一方式返回训练状态与结果
平台必须为所有任务类型的成功和失败训练返回一致的状态、消息和结果结构。

#### Scenario: 训练成功完成
- **WHEN** 某次训练成功结束
- **THEN** 平台返回成功状态，并同时返回指标、产物引用和任务元数据

#### Scenario: 训练校验失败或运行失败
- **WHEN** 某次训练无法继续或因错误终止
- **THEN** 平台返回失败状态，并附带前端可直接展示的任务级错误信息
