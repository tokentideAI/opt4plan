# Shopping 测试调用流程与 Agentic Framework 改造指南

## 脚本调用流程

`cyt_test_run.sh` 直接执行两个 Python 入口：

```text
cyt_test_run.sh
  ├─ 内嵌 Python：选前 5 条任务，复制对应数据库
  ├─ shoppingplanning/run.py：启动推理
  │   └─ agent/shopping_agent.py
  │       └─ ShoppingFnAgent：调用模型、执行工具、保存轨迹
  └─ shoppingplanning/evaluation/evaluation_pipeline.py：评估数据库结果
```

它不会调用 `run_all.sh` 或两个领域的 `run.sh`。

## 主要改造模块

如果想换成 Agentic Framework，主要改造以下模块：

| 模块 | 改造内容 |
|---|---|
| `shoppingplanning/agent/shopping_agent.py` | 核心入口。用框架替换 `ShoppingFnAgent` 的模型与工具执行循环 |
| `shoppingplanning/agent/call_llm.py` | 模型调用适配。如果框架自带模型客户端，可由框架接管 |
| `shoppingplanning/tools/` | 将现有工具包装成框架的工具接口 |
| `shoppingplanning/tools/shopping_tool_schema.json` | 将工具参数定义转换成框架需要的格式 |
| `shoppingplanning/agent/prompts/` | 按框架的消息格式传入系统提示词 |

建议首先保留 `run.py` 调用的 **`run_agent_inference(...)` 接口**，在它内部换掉 `ShoppingFnAgent`。这样现有的并发调度、5 条数据测试入口和评估流程都能继续使用。

## 接入时需要保留的约定

- **数据库隔离**：每条任务的工具必须读取和修改 `database_dir/case_{sample_id}`。
- **调用上限**：将 `max_llm_calls` 映射到框架的执行次数限制，检查两者计数方式是否一致。
- **结果与轨迹**：保持 `run_agent_inference` 返回的统计字段，并适配框架消息到现有轨迹保存格式。

评估入口主要检查工具执行后数据库中的结果，因此第一版可以只替换 Agent 执行层，继续沿用现有工具和评估代码。
