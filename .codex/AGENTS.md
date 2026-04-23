# AGENTS

这个文件说明 `.codex/` 目录下各文档的职责。

## Reading Order

建议按这个顺序读：

1. [project.md](./project.md)
2. [workflows.md](./workflows.md)
3. [task_001.md](./task_001.md)
4. [operator_support_state.md](./operator_support_state.md)

## File Roles

### `project.md`

回答：

- 这个项目面向谁
- 当前主线和当前范围是什么
- 当前已经做到哪里
- 支持哪些 backend
- 当前明确不做什么

### `workflows.md`

回答：

- `validate` 做什么
- `train` 做什么
- `quantize` 做什么
- checkpoint、calibration、量化默认值怎么工作
- 当前会产出哪些文件

### `task_001.md`

回答：

- 当前主任务为什么是一维数据
- 当前任务已经完成到哪里
- 当前 in-scope / out-of-scope 是什么

### `operator_support_state.md`

回答：

- ESP-DL 底层算子支持和 DSL 的关系
- 为什么当前不直接做严格部署算子校验

## Maintenance Rule

- `.codex/` 是当前权威文档目录
- 代码行为变更时，先更新这里
- 文档尽量短、稳定、可执行
- 不把临时讨论记录直接堆进这里
