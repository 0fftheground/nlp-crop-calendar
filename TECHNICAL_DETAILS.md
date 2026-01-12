## 系统概览

```
Chainlit UI --> FastAPI 后端 --> Planner（LLM）+ Executor（工具 + LangGraph）
```

1. **Chainlit (`chainlit_app.py`)** 向 `POST /api/v1/handle` 提交用户输入，并携带 `session_id` 做多会话隔离。响应会告知是否执行了工具或产出了 LangGraph 计划，trace 会单独展示。
2. **FastAPI (`src/api/server.py`)** 暴露 `/health` 与 `/api/v1/handle`，所有请求/响应使用统一的 Pydantic 模型。
3. **Planner Router (`src/agent/router.py`)** 调用 LLM planner 选择 tool/workflow/none，然后执行并持久化 follow-up 状态（按 `session_id`）。
4. **LangGraph (`src/agent/workflows/crop_calendar_graph.py`/`src/agent/workflows/growth_stage_graph.py`)**
   - 作物日历工作流实现抽取 -> 追问 -> 工具并行 -> 农事推荐输出。
   - 抽取使用 LLM（结构化输出）并提供启发式兜底；缺失字段最多追问 2 次，仍缺失则使用默认值补齐。
   - 气象、品种与农事推荐通过工具调用完成（`weather_lookup`/`variety_lookup`/`farming_recommendation`）。

## 核心模块
- `src/infra/config.py` – 读取 `.env` 并暴露 `AppConfig`。
- `src/infra/llm.py` – 创建 Planner 与抽取模型使用的 `ChatOpenAI`。
- `src/infra/llm_extract.py` – 结构化抽取的通用封装。
- `src/infra/cache_keys.py` – 基于 `PlantingDetails` 生成缓存 key 的工具。
- `src/infra/tool_provider.py` – provider 切换与内网 HTTP 调用封装。
- `src/infra/variety_store.py` – 轻量品种检索（`src/resources/rice_variety_approvals.sqlite3`）。
- `src/infra/pending_store.py` – 追问状态持久化与 TTL（memory/sqlite）。
- `src/infra/tool_cache.py` – 工具结果缓存（memory/sqlite）。
- `src/infra/weather_cache.py` – 气象序列缓存（可持久化）。
- `src/infra/interaction_store.py` – 请求/响应审计记录（memory/sqlite）。
- `src/infra/preference_store.py` – 会话级偏好记忆（标准化 PlantingDetails）与 TTL。
- 品种检索采用关键词严格匹配，不使用 embedding/Qdrant。
- `src/schemas/models.py` – 共享 schema（`UserRequest`, `WorkflowResponse`, `ToolInvocation`, `HandleResponse`），`UserRequest` 支持 `session_id`。
- `src/agent/planner.py` – LLM planner，输出 `ActionPlan`（tool/workflow/none），综合工具/工作流清单与 pending 上下文。
- `src/agent/tools/registry.py` – 工具注册与执行（品种/气象/生育期/农事推荐），支持 `mock`/`intranet` provider。
- `src/agent/intent_rules.py` – 规则意图与取消/清除记忆关键词辅助。
- `src/agent/router.py` – 执行 planner 决策、分发工具/工作流，并更新追问状态。
- `src/domain/services.py` – 封装种植日历流水线（抽取/追问/校验/天气/生育期/农事推荐）及工具占位实现。
- `src/agent/workflows/state.py` / `crop_calendar_graph.py` / `growth_stage_graph.py` – LangGraph 状态定义与工作流实现。
- `src/api/server.py` – FastAPI 路由与依赖缓存。
- `chainlit_app.py` – UI 客户端。
- `src/resources/rice_variety_approvals.sqlite3` – 水稻品种审定数据库（可用 `VARIETY_DB_PATH` 覆盖路径）。

## LangGraph 说明
- `StateGraph` 作为调度骨架，目前包含 `extract`、`ask`、`context`、`recommend` 四个节点。
- `GraphState` 关键字段：`planting_draft`, `missing_fields`, `followup_count`, `weather_info`, `variety_info`, `recommendation_info`。
- 追问逻辑：若缺失字段存在则进入 `ask`；用户回复后与已有 draft 合并，最多追问两次，仍缺失则用默认值补齐进入 `context`。
- 作物日历与生育期 workflow 会将结果按 `PlantingDetails` 缓存，命中则短路直接返回。

## 路由逻辑
- `src/agent/router.RequestRouter` 使用 `PlannerRunner` 输出 `ActionPlan`（tool/workflow/none），并执行对应动作。
- 工具调用直接执行 `execute_tool`；工作流执行对应 LangGraph。`HandleResponse.mode` 告知前端 “tool / workflow / none”，`tool.data` 或 `plan.recommendations` 承载结果。
- 工具处理函数在 `src/agent/tools/registry.py` 返回 `ToolInvocation`（结构化 `name/message/data`），便于 UI 渲染。
- 追问状态通过 pending store 持久化（memory/sqlite 可选）并带 TTL；pending 摘要会注入 planner 用于判断继续追问或切换新问题。
- “取消追问” 类关键词会清理 pending；planner 允许按上下文切换工具或工作流。
- 如果存在偏好记忆，会在 workflow 追问时提示是否沿用；“清除记忆” 可清空偏好。

## 作物日历工作流（当前）
`src/agent/workflows/crop_calendar_graph.py` 是运行中的主流程，取代早期单体 pipeline：

1. **LLM 抽取**：`extract_planting_details(prompt, llm_extract=...)` 输出 `PlantingDetailsDraft`。
2. **缺失字段检查/追问**：`list_missing_required_fields(draft)` 判断必填项；若缺失则进入追问节点。用户回复后合并答案，最多追问两次。
3. **默认补齐**：若超出追问次数仍缺失，则用默认值补齐并记录在 `assumptions`。
4. **工具并行上下文**：`weather_lookup` 与 `variety_lookup` 并行执行，形成 `weather_info`/`variety_info`。
5. **农事推荐**：调用 `farming_recommendation`，输入为包含 `planting`、`weather`、`variety` 的 JSON 字符串，结果写入 `recommendation_info`；最终消息由 workflow 统一组织。

## 工具说明
- `growth_stage_prediction` 仅用于读取历史缓存结果；生育期预测必须走 workflow，由工作流完成抽取/追问/内网调用。
- 命中生育期缓存需要传入 `PlantingDetails` JSON（或包含 `planting` 字段的 JSON），缓存 key 由 `cache_keys.py` 生成。
- 工具/领域服务支持 `mock`/`intranet` provider；品种查询默认走本地 SQLite（`VARIETY_PROVIDER=local`），设置 `*_PROVIDER=intranet` 并配置 `*_API_URL`/`*_API_KEY` 可切换到内网接口。
- `variety_lookup` / `weather_lookup` / `farming_recommendation` 结果使用 TTL 缓存以减少重复调用。
- 品种匹配策略：先按品种名称召回全部审定记录，基于用户地点与“审定区域/适种地区”规则评分；若存在多条高分记录则交由 LLM 进行择优。

## 部署建议
- 使用 `uvicorn`/`gunicorn` 部署 FastAPI 并接入 HTTPS；Chainlit 可反向代理或独立部署。
- 如需流式输出，可提供 WebSocket/SSE，将 LangGraph 流事件转发给前端。
- 建议在 `router.handle` 与工具处理处加入结构化日志，便于分析路由准确性。

## 测试
- `python -m unittest` 运行基础测试集。
- `tests/test_intent_routing.py` 读取 `tests/intent_routing_cases.jsonl`，使用规则路由校验意图。
- `scripts/intent_routing_test.py --strategy rule|llm` 用于手动路由检查。
