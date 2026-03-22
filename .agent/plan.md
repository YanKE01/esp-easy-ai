# 计划索引

## 1. 项目目标

这个项目用于帮助用户开发自己的 ESP 端侧 AI。

当前主线：

1. 用 Vue 搭建前端工作台
2. 用 Python 提供训练调用接口
3. 先打通时序任务
4. 再扩展图像任务

当前说明：

- `Time Series` 用于真正的序列数据
- 表格型任务后续单独作为 `Tabular` 类型规划
- 当前前端已先从 `1D -> Data Classification -> Dataset` 开始细化交互

## 2. 计划文件位置

顶层计划只保留索引，具体内容拆到独立文件。

- 总体架构：[`.agent/plans/architecture.md`](/Users/linke/project/esp-easy-ai/.agent/plans/architecture.md)
- 时序数据集：[`.agent/plans/timeseries-dataset.md`](/Users/linke/project/esp-easy-ai/.agent/plans/timeseries-dataset.md)
- 时序训练：[`.agent/plans/timeseries-training.md`](/Users/linke/project/esp-easy-ai/.agent/plans/timeseries-training.md)

## 3. 当前推进顺序

当前按下面顺序推进：

1. 完成前端基础布局
2. 明确 `1D -> Dataset`
3. 优先落地 `Data Classification -> Dataset`
4. 再继续明确 `Training`

## 4. 说明

后续新增主题时，继续在 `.agent/plans/` 下新增对应计划文件，并在这里登记入口。
