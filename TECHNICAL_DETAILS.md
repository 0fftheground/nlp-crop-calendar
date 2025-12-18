## System Overview

```
Chainlit UI --> FastAPI backend --> (LangChain Tools | LangGraph Workflow)
```

1. **Chainlit (`chainlit_app.py`)** posts user prompts to `POST /api/v1/handle`. The response tells the UI whether a standalone tool ran or whether a LangGraph plan was produced. Debug traces are shown in a separate message.
2. **FastAPI (`src/api/server.py`)** exposes `/health`, `/api/v1/plan` (always run LangGraph), and `/api/v1/handle` (run router). All handlers share the same Pydantic request models.
3. **Router (`src/agent/router.py`)** asks the LLM (via `src/agent/tool_selector.py`) whether a quick tool should run; otherwise it falls back to the LangGraph workflow runner.
4. **LangGraph (`src/agent/workflows/crop_graph.py`)**
   - 当前文件仅提供 `StateGraph` 构建模板，开发者可在此接入最新的种植日历流水线。
   - 推荐直接在节点中调用 `src/domain/services.generate_crop_calendar`，或拆分为“抽取/补问/调用工具”等多个节点。

## Key Modules
- `src/data/config.py` – loads `.env`, exposes `AppConfig`.
- `src/data/llm.py` – instantiates `ChatOpenAI` using the configured API key (mock mode removed).
- `src/schemas/models.py` – shared schemas (`UserRequest`, `WorkflowResponse`, `ToolInvocation`, `HandleResponse`).
- `src/tools/registry.py` – registry of executable tools (weather lookup, soil sampling guidance, market price lookup).
- `src/agent/tool_selector.py` – prompts the LLM to pick `"tool"` or `"workflow"` plus the tool name.
- `src/agent/router.py` – orchestrates “tool vs workflow” decision using the selector and executes the chosen path.
- `src/domain/knowledge_base.py` – stage-aware agronomy facts consumed by the planner.
- `src/agent/workflows/state.py` / `crop_graph.py` – LangGraph state types and compiled workflow.
- `src/api/server.py` – FastAPI routers and dependency caching.
- `chainlit_app.py` – UI client.

## LangGraph Notes
- `StateGraph` 仍然作为调度骨架；根据实际需求在 `build_graph()` 中注册节点并设置 entry/edges。
- 建议 GraphState 至少包含 `user_prompt`, `planting_draft`, `assumptions`, `weather_series`, `growth_stage`, `operation_plan` 等字段，方便在各节点传递。
- 若需要工具/LLM 串联，可在节点内部复用本文件下述的“Crop Calendar Pipeline”函数，以保持与 API 路由一致的逻辑。

## Routing Logic
- `src/agent.tool_selector.ToolSelector` 仍按 JSON schema `{"action":"tool|workflow","tool":"name","reason":"..."}` 让 LLM 决定执行链路。
- 当 selector 选择 “workflow” 时，请在 LangGraph 或 API handler 中调用 `generate_crop_calendar`；若选择具体工具，则直接执行该工具并返回 `ToolInvocation`。
- `HandleResponse.mode` 告知前端“tool / workflow”，`tool.data` 或 `plan.recommendations` 继续承载结果。

## Crop Calendar Pipeline
`src/domain/services.py` 提供了从“自由文本到日历”的全链路实现，方便在 router 或 LangGraph 中复用。

1. **LLM 抽取**：`extract_planting_details(prompt, llm_extract=...)` 通过真实 LLM 解析用户自然语言，生成 `PlantingDetailsDraft`（`src/schemas/models.py`）。
2. **缺失字段检查**：`list_missing_required_fields(draft)` 返回必填项缺口（默认 crop / planting_method / sowing_date）。如果返回非空，上层应“一次性”追问这些字段。
3. **补齐/默认**：用 `merge_planting_answers(draft, answers=..., unknown_fields=..., fallback=...)` 将用户补充合并回 Draft；若用户回答“不知道”，在 `unknown_fields` 中声明并提供 fallback（默认值），函数会把该默认写入 draft 并记录在 `assumptions`。
4. **严格校验**：`normalize_and_validate_planting(draft)` 仅在必填项齐全时生成 `PlantingDetails`；否则抛出 `MissingPlantingInfoError`，提示继续追问。
5. **后续推理**：`generate_crop_calendar(...)` 内部依次执行：
   - `derive_weather_range`：依据播种/移栽日期推算天气查询窗口；
   - `fetch_weather`（可替换为真实 tool）并通过 `assemble_weather_series` 生成统一的 `WeatherSeries`；
   - `predict_growth_stage`（包装 `predict_growth_stage_gdd`）输出 `GrowthStageResult`；
   - `build_operation_plan`（包装 `recommend_ops`）生成 `OperationPlanResult`。
   返回值 `CropCalendarArtifacts` 中包含最终 `planting`、`weather_series`、`growth_stage`、`operation_plan` 以及 `assumptions`，可直接序列化给前端或写入 LangGraph state。

## Deployment Tips
- Serve FastAPI via `uvicorn`/`gunicorn` behind HTTPS; Chainlit can be reverse-proxied or hosted separately.
- For streaming, expose a websocket or Server-Sent Events endpoint that forwards LangGraph stream events.
- Instrument `router.handle` and tool handlers with structured logging to monitor routing accuracy.
