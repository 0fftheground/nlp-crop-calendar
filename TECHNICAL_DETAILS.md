## System Overview

```
Chainlit UI --> FastAPI backend --> LangChain Agent (Tools + LangGraph Workflow Tool)
```

1. **Chainlit (`chainlit_app.py`)** posts user prompts to `POST /api/v1/handle` and includes a `session_id` for multi-user isolation. The response tells the UI whether a standalone tool ran or whether a LangGraph plan was produced. Debug traces are shown in a separate message.
2. **FastAPI (`src/api/server.py`)** exposes `/health` and `/api/v1/handle`. All handlers share the same Pydantic request models.
3. **Agent Router (`src/agent/router.py`)** uses a LangChain tool-calling agent. The agent can call standalone tools or invoke the LangGraph workflow via a dedicated workflow tool. Follow-up state is cached per `session_id`.
4. **LangGraph (`src/agent/workflows/crop_graph.py`)**
   - 当前工作流已实现抽取 -> 追问 -> 工具并行 -> 农事推荐输出。
   - 抽取使用 LLM（结构化输出）并提供启发式兜底；缺失字段最多追问 2 次，仍缺失则使用默认值补齐。
   - 气象、品种与农事推荐通过工具调用完成（`weather_lookup`/`variety_lookup`/`farming_recommendation`）。

## Key Modules
- `src/infra/config.py` – loads `.env`, exposes `AppConfig`.
- `src/infra/llm.py` – instantiates `ChatOpenAI` for the main agent plus a lightweight extractor model.
- `src/infra/llm_extract.py` – shared structured extraction helper that runs against the extractor model.
- `src/infra/tool_provider.py` – shared provider switch + intranet HTTP caller for tools/domain services.
- `src/infra/variety_store.py` – lightweight local retrieval for variety name hints (backed by `src/resources/varieties.json`).
- Variety retrieval prefers embedding-based similarity when possible (env override: `EMBEDDING_MODEL`); if Qdrant is configured, it is queried first (`QDRANT_URL`, `QDRANT_COLLECTIONS` with `"variety"` key).
- `src/schemas/models.py` – shared schemas (`UserRequest`, `WorkflowResponse`, `ToolInvocation`, `HandleResponse`). `UserRequest` 支持 `session_id`。
- `src/tools/registry.py` – registry of executable tools (variety/weather/growth stage/farming recommendation) and agent-friendly wrappers. Providers can switch between `mock` and `intranet` via `*_PROVIDER`/`*_API_URL`/`*_API_KEY`.
- `src/agent/tool_selector.py` – legacy LLM router (not used by the current agent-based router).
- `src/agent/intent_rules.py` – deterministic rule-based router for tests (optional).
- `src/agent/router.py` – orchestrates tool-calling agent execution and parses tool/workflow outputs.
- `src/domain/services.py` – 封装种植日历流水线（抽取/追问/校验/天气/生育期/农事推荐）及工具占位实现；天气/推荐支持 provider 切换。
- `src/agent/workflows/state.py` / `crop_graph.py` – LangGraph state types and compiled workflow.
- `src/api/server.py` – FastAPI routers and dependency caching.
- `chainlit_app.py` – UI client.

## LangGraph Notes
- `StateGraph` 作为调度骨架，目前包含 `extract`、`ask`、`context`、`recommend` 四个节点。
- `GraphState` 关键字段：`planting_draft`, `missing_fields`, `followup_count`, `weather_info`, `variety_info`, `recommendation_info`。
- 追问逻辑：若缺失字段存在则进入 `ask`；用户回复后将与已有 draft 合并，最多追问两次，仍缺失则用默认值补齐进入 `context`。

## Routing Logic
- `src/agent/router.RequestRouter` creates a tool-calling agent and includes a `crop_calendar_workflow` tool that runs LangGraph.
- Standalone tools and the workflow tool return JSON strings; the router inspects `intermediate_steps` to decide whether to respond with `mode="tool"` or `mode="workflow"`. If no tool/workflow is invoked, it returns `mode="none"` and keeps the assistant reply in `plan.message`.
- `HandleResponse.mode` 告知前端“tool / workflow / none”，`tool.data` 或 `plan.recommendations` 继续承载结果。

## Crop Calendar Workflow (Current)
`src/agent/workflows/crop_graph.py` 已成为运行中的主流程，取代早期的单体 pipeline：

1. **LLM 抽取**：`extract_planting_details(prompt, llm_extract=...)` 输出 `PlantingDetailsDraft`。
2. **缺失字段检查/追问**：`list_missing_required_fields(draft)` 判断必填项；若缺失则进入追问节点。用户回复后合并答案，最多追问两次。
3. **默认补齐**：若超出追问次数仍缺失，则用默认值补齐并记录在 `assumptions`。
4. **工具并行上下文**：`weather_lookup` 和 `variety_lookup` 并行执行，形成 `weather_info`/`variety_info`。
5. **农事推荐**：调用 `farming_recommendation`，输入为包含 `planting`、`weather`、`variety` 的 JSON 字符串，结果写入 `recommendation_info`；最终消息由 workflow 统一组织。

## Tool Notes
- `growth_stage_prediction` 使用 `PlantingDetailsDraft` 的结构化抽取；若缺少作物/种植方式/播种日期/地区，会返回追问提示，待用户补充后继续调用品种与气象工具并做积温计算。
- Tools/domain services support `mock`/`intranet` providers; set `*_PROVIDER=intranet` with `*_API_URL`/`*_API_KEY` to call internal endpoints.

## Deployment Tips
- Serve FastAPI via `uvicorn`/`gunicorn` behind HTTPS; Chainlit can be reverse-proxied or hosted separately.
- For streaming, expose a websocket or Server-Sent Events endpoint that forwards LangGraph stream events.
- Instrument `router.handle` and tool handlers with structured logging to monitor routing accuracy.

## Testing
- `python -m unittest` runs the minimal suite.
- `tests/test_intent_routing.py` validates routing against `tests/intent_routing_cases.jsonl` via rule-based router.
- `scripts/intent_routing_test.py --strategy rule|llm` supports manual routing checks.
