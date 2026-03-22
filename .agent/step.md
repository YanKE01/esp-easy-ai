# 当前实现记录

## 1. 已完成内容

当前已经开始实现前端部分，已经从基础页面骨架推进到 `1D -> Dataset` 的首个可交互原型。

已完成的实现包括：

1. 创建 `web/` 前端工程基础文件
2. 搭建 Vue + Vite + TypeScript 最小运行环境
3. 完成顶部深蓝色标题栏
4. 完成左侧深蓝色导航栏
5. 左侧导航支持点击切换和黄色激活态
6. 当前左侧导航调整为 `Overview`、`1D`、`Image`
7. 完成 `1D` 页面顶部 tab
8. 将 `Dataset`、`Training`、`Emulation` 拆成独立 Vue 文件
9. 完成 `Overview` 首页说明卡片
10. 开始实现 `DatasetTab.vue`
11. 当前优先推进 `Data Classification` 数据集流程
12. 支持按数据集文件夹上传，要求根目录下包含 `train/` 和 `test/`
13. 文件夹上传后弹出对话框，要求用户填写 `Number of Channels`
14. 解析类别名，支持在前端选择一个类别进行预览
15. 接入 `ECharts`，完成一行数据的多通道统一波形预览
16. 波形预览支持鼠标 hover、纵向指示线、tooltip 和 data zoom

## 2. 当前页面结构

当前页面结构如下：

- 顶部栏标题：`ESP EASY AI`
- 左侧导航：
  - `Overview`
  - `1D`
  - `Image`
- 主区域：
  - 点击 `1D` 后显示顶部 tab
  - tab 包括：
    - `Dataset`
    - `Training`
    - `Emulation`
  - 当前重点在 `Dataset`

## 3. 当前代码结构

当前前端相关文件如下：

```text
web/
  src/
    App.vue
    main.ts
    style.css
    components/
      TopBar.vue
      Sidebar.vue
      InfoPlaceholder.vue
      WaveformPreviewChart.vue
    views/
      TimeSeriesView.vue
      time-series/
        DatasetTab.vue
        TrainingTab.vue
        EmulationTab.vue
```

## 4. 当前实现说明

### 顶部栏

- 使用独立组件 `TopBar.vue`
- 样式参考 `esp_isp_tool/frontend-vue`
- 标题文字为 `ESP EASY AI`

### 左侧导航

- 使用独立组件 `Sidebar.vue`
- 保持深蓝背景
- 点击导航项后显示黄色激活背景
- 当前入口为 `Overview`、`1D`、`Image`

### Overview 页面

- 使用说明型卡片作为首页入口
- 当前作为项目范围说明页

### 1D 页面

- 使用独立页面 `TimeSeriesView.vue`
- tab 风格改为和参考项目一致的文字 tab
- 激活 tab 显示深蓝文字和底部横线

### Dataset 页面

- 当前优先实现 `Data Classification`
- 数据集上传方式为文件夹上传
- 当前要求根目录下包含：
  - `train/`
  - `test/`
- 上传后弹出对话框，让用户输入 `Number of Channels`
- 当前按文件名解析类别，例如：
  - `train_on.csv -> on`
  - `test_off.csv -> off`
- 当前支持在前端选择某个类别，只读取该类别的一行数据进行预览
- 波形图当前使用 `ECharts`
- 当前交互包括：
  - hover tooltip
  - 纵向指示线
  - 缩放
  - 底部 data zoom 滑条

### 其余子页面

仍保留三个独立文件入口：

- `DatasetTab.vue`
- `TrainingTab.vue`
- `EmulationTab.vue`

其中：

- `DatasetTab.vue` 已开始填充
- `TrainingTab.vue` 和 `EmulationTab.vue` 目前仍是后续入口

## 5. 当前与 plan 的对应关系

当前实现已经覆盖 `plan.md` 中的前端主线起点：

1. 搭建 Vue 前端基础页面
2. 建立 `1D` 任务页面入口
3. 建立页面级导航和 tab 结构
4. 开始落地 `Dataset` 具体交互

## 6. 下一步

当前最直接的下一步是继续把 `Data Classification` 的 `Dataset` 流程补完整。

建议顺序：

1. 增加 `Dataset Summary` 和数据组织说明
2. 补真实数据表格预览
3. 明确标签来源和训练输入定义
4. 再推进 `Training` 页内容
