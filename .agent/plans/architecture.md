# 总体架构计划

## 1. 目标

系统的核心链路是：

1. 用户在前端填写任务参数
2. 前端把参数提交给后端
3. 后端调用训练逻辑
4. 后端返回训练结果
5. 前端展示训练结果

## 2. 整体结构

```text
Vue Frontend
  -> Python API
    -> Train Function
    -> Return Result
```

## 3. 前端

前端统一使用 Vue。

前端职责：

- 展示任务页面
- 收集用户参数
- 发起训练请求
- 展示训练结果

当前页面主线：

- 首页
- Time Series
- Image
- Deploy

## 4. 后端

后端统一使用 Python。

后端职责：

- 提供 HTTP API
- 接收前端数据
- 校验参数
- 调用训练函数
- 返回训练结果

建议采用 FastAPI。

## 5. 基础接口

- `GET /health`
- `POST /train/timeseries`
- `POST /train/image`

## 6. 通用请求结构

```json
{
  "task_type": "classification",
  "input": {},
  "config": {}
}
```

## 7. 通用返回结构

```json
{
  "success": true,
  "message": "ok",
  "result": {}
}
```
