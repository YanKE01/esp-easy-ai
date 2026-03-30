## ADDED Requirements

### Requirement: 训练 API 必须使用统一的任务请求模型
训练后端必须接受统一的逻辑请求结构，其中包含任务标识、数据集定义、模型配置和训练配置，而不因任务类型不同而改变整体模型。

#### Scenario: 前端提交 1D 训练请求
- **WHEN** 用户启动 1D 任务训练
- **THEN** 前端发送的请求对象中包含任务标识、数据集信息、模型预设信息和训练参数

#### Scenario: 首版训练接口同步执行并返回流式日志
- **WHEN** 用户启动第一阶段的 `1D` 训练任务
- **THEN** 后端使用同步方式执行本次训练请求，并在训练过程中持续向前端返回流式日志，训练结束后在同一响应流中返回成功结果或错误信息

#### Scenario: 1D 后端按任务边界拆分模块
- **WHEN** 平台开始实现第一阶段的 `1D` 训练后端
- **THEN** 后端至少按 `api/`、`training/dispatcher.py`、`training/one_d/common/`、`training/one_d/data_classification/` 与 `training/one_d/regression/` 的边界拆分文件，而不是集中在单个训练脚本中

#### Scenario: 分类原始类别文件在训练启动前补充 split
- **WHEN** 用户为 `1D Data Classification` 的原始类别文件数据集点击 `Train`
- **THEN** 前端先弹出 split 配置，待用户确认后再组装训练请求

#### Scenario: 分类 MLP 允许编辑 hidden dense layers
- **WHEN** 用户在 `1D Data Classification` 训练页选择 `MLP`
- **THEN** 前端允许用户调整 hidden dense layers 的神经元数量，并在倒数第二层提供追加 dense layer 的入口

#### Scenario: 分类训练页展示输出层类别数
- **WHEN** 用户在 `1D Data Classification` 训练页查看 `MLP Structure`
- **THEN** 前端展示 `Output Layer`，并直接使用 `Dataset` 阶段识别到的类别数量作为输出类别数

#### Scenario: 分类 1D CNN 展示默认 CNN Block 结构
- **WHEN** 用户在 `1D Data Classification` 训练页选择 `1D CNN`
- **THEN** 前端展示默认的两个 `CNN Block`，每个 block 由一个 `Conv1D` 和一个 `MaxPooling1D` 组成，并在其后展示预设的 `GlobalAveragePooling1D -> Dense(32) -> Dropout(0.2) -> Output Layer`

#### Scenario: 分类 1D CNN 允许编辑 block 参数
- **WHEN** 用户在 `1D Data Classification` 训练页点击某个 `CNN Block`
- **THEN** 前端通过弹窗允许用户编辑该 block 的 `filters`、`kernel size` 与 `pool size`

#### Scenario: 分类 1D CNN 允许增删 CNN Block
- **WHEN** 用户在 `1D Data Classification` 训练页编辑 `1D CNN` 结构
- **THEN** 前端允许用户在 `Output Layer` 上方追加新的 `CNN Block`，并允许删除可删除的预设或自定义 `CNN Block`

#### Scenario: 分类 1D CNN 允许扩展 Dense 与 Dropout
- **WHEN** 用户在 `1D Data Classification` 训练页编辑 `1D CNN` 结构
- **THEN** 前端允许用户在 `Output Layer` 上方追加 `Dense` 与 `Dropout` 层，以继续扩展当前的 `1D CNN` 层栈

#### Scenario: 分类 1D CNN 预设层允许删除
- **WHEN** 用户在 `1D Data Classification` 训练页查看 `1D CNN` 的默认预设结构
- **THEN** 前端允许用户删除 `Output Layer` 之外的预设层，以基于预设重新搭建自定义结构

#### Scenario: 分类 1D CNN 固定 block 其余参数
- **WHEN** 用户在 `1D Data Classification` 训练页配置 `1D CNN`
- **THEN** 前端固定每个 `CNN Block` 的 `padding='same'` 与 `activation='relu'`，不在训练页中开放这些参数

#### Scenario: 分类 1D CNN 复用数据集输入与输出元数据
- **WHEN** 用户在 `1D Data Classification` 训练页查看 `1D CNN` 结构
- **THEN** 前端直接复用 `Dataset` 阶段确认的输入 channel 数作为模型输入通道数，并直接复用 `Dataset` 阶段识别到的类别数量作为输出类别数

#### Scenario: 分类 1D CNN 后端按顺序层栈构建模型
- **WHEN** 后端接收到 `1D Data Classification` 的 `1D CNN` 训练请求
- **THEN** 后端按前端传入的顺序 `layers[]` 解释允许的层类型来构建网络，并根据类别数自动补齐最终输出层

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
