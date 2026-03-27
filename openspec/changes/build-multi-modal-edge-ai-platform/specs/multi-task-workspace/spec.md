## ADDED Requirements

### Requirement: 平台必须提供四类训练任务入口
平台必须为 `1D Data Classification`、`1D Time Series Classification`、`Image Classification` 和 `Image Object Detection` 提供明确的任务入口，使用户能够从任务化流程开始，而不是手动拼装能力。

#### Scenario: 用户进入工作台
- **WHEN** 用户打开训练工作台
- **THEN** 系统按 `1D` 和 `Image` 两大类展示可用任务入口

#### Scenario: 用户选择具体任务
- **WHEN** 用户选择四类任务中的任意一个入口
- **THEN** 系统基于对应任务模板打开该任务的数据集、训练和结果流程

### Requirement: 工作台必须保持一致的任务流程
所有支持的任务即使具体字段不同，也必须遵循一致的顶层流程，包括数据集准备、训练配置、训练执行、结果查看和产物导出。

#### Scenario: 用户切换 1D 与图像任务
- **WHEN** 用户分别打开两种不同任务类型
- **THEN** 两类任务页面都以相同顺序展示一致的顶层阶段

#### Scenario: 当前 1D 数据集原型纳入统一流程
- **WHEN** 现有 `1D -> Dataset` 原型被纳入平台工作台
- **THEN** 它作为标准化任务流程的第一阶段实现存在，而不是独立特例
