## System Overview

```
Chainlit UI --> FastAPI backend --> Planner (LLM) + Executor (tools + LangGraph)
```

1. **Chainlit (`chainlit_app.py`)** sends user input to `POST /api/v1/handle` with `session_id` for multi-session isolation (optionally `user_id` for user-level context). The response indicates whether a tool or a LangGraph plan ran, and traces are shown separately.
2. **FastAPI (`src/api/server.py`)** exposes `/health` and `/api/v1/handle`; all requests/responses use unified Pydantic models.
3. **Planner Router (`src/agent/router.py`)** calls the LLM planner to choose tool/workflow/none, then executes and persists follow-up state by `session_id`.
4. **LangGraph (`src/agent/workflows/crop_calendar_graph.py`/`src/agent/workflows/growth_stage_graph.py`)**
   - The crop calendar workflow implements extraction -> follow-up -> external crop calendar API -> recommendation output.
   - The growth-stage workflow implements extraction -> follow-up -> DB lookup (plant plan + forecast) -> response formatting.
   - Extraction uses an LLM (structured output) with heuristic fallback; missing fields are asked up to 2 times, and any remaining fields are filled with defaults.
   - Crop calendar recommendations are generated via the external crop calendar API when configured; weather/variety tools are used for their standalone queries.

## Core Modules
- `src/infra/config.py` - Reads `.env` and exposes `AppConfig`.
- `src/infra/db_catalog.py` - Central DB table metadata and region-lookup source resolution.
- `src/infra/llm.py` - Creates `ChatOpenAI` for the planner and extractor models.
- `src/infra/llm_extract.py` - Common wrapper for structured extraction.
- `src/infra/cache_keys.py` - Utility for generating cache keys from `PlantingDetails`.
- `src/infra/tool_provider.py` - Provider normalization helpers.
- `src/infra/variety_store.py` - Lightweight variety lookup (Postgres via `AGRI_DB_URL`).
- `src/infra/pending_store.py` - Follow-up state persistence with TTL (memory/sqlite/postgres).
- `src/infra/tool_cache.py` - Tool result cache (memory/sqlite/postgres).
- `src/infra/interaction_store.py` - Request/response audit records (memory/sqlite/postgres).
- `src/prompts/*` - LLM prompts and workflow/tool user copy (planner/extract/fallback prompts).
- Variety retrieval uses candidate-name matching + fuzzy tokens, no embedding/Qdrant.
- `src/schemas/models.py` - Shared schemas (`UserRequest`, `WorkflowResponse`, `ToolInvocation`, `HandleResponse`), `UserRequest` supports `session_id` and optional `user_id`.
- `src/agent/planner.py` - LLM planner that outputs `ActionPlan` (tool/workflow/none) using tool/workflow lists and pending context (prompt in `src/prompts/planner.py`).
- `src/agent/tools/registry.py` - Tool registration and execution (variety/weather/growth-stage/memory).
- `src/agent/router.py` - Orchestrator that composes intent routing, pending management, and execution.
- `src/agent/intent_router.py` - Intent planning/routing (rules/fast path/LLM planner).
- `src/agent/pending_manager.py` - Pending follow-up state lifecycle.
- `src/agent/plan_executor.py` - Tool/workflow execution and validation path.
- `src/application/services/*` - Application-layer services (variety/weather/recommendation/crop calendar/planting extraction) used by tools and workflows.
- `src/application/ports.py` / `src/application/adapters.py` - App-level Port/Adapter boundary for config/sql/http dependencies.
- `src/domain/planting.py` + `src/domain/planting_models.py` - Domain logic and models for planting extraction/validation.
- `src/agent/workflows/state.py` / `crop_calendar_graph.py` / `growth_stage_graph.py` - LangGraph state definition and workflow implementation.
- `src/api/server.py` - FastAPI routes and dependency cache.
- `chainlit_app.py` - UI client.
- 品种与生育期预测数据通过 Postgres 读取（`AGRI_DB_URL`，或由 `AGRI_DB_HOST/PORT/NAME/USER/PASSWORD/SSLMODE` 拼接）。

## LangGraph Details
- `StateGraph` is the orchestration skeleton; crop calendar uses `extract`/`ask`/`context`/`recommend`, growth-stage uses `extract`/`ask`/`predict`.
- `GraphState` key fields: `planting_draft`, `missing_fields`, `followup_count`, `weather_info`, `variety_info`, `recommendation_info`.
- Follow-up logic: if missing fields exist, go to `ask`; user replies are merged with the existing draft, up to two rounds; remaining missing fields are filled with defaults before entering `context`.
- Crop calendar workflow has cache hooks keyed by `PlantingDetails` (currently disabled via `tool_cache`).

Growth-stage workflow specifics:
- Parses user variety/plan info, queries `agri_plant_plan`; if multiple matches, asks the user to pick one, then reads `agri_growth_stage_forecast`.
- Maps `sowing_method` / `culti_type` / `stage_name` via `agri_code_dict` categories (`sowingmtd` / `culti_type` / `growth_stage`).

## Routing Logic
- `src/agent/router.RequestRouter` orchestrates three collaborators:
  - `IntentRouter` (plan generation),
  - `PendingManager` (follow-up state),
  - `PlanExecutor` (action execution).
- Tools are invoked via `execute_tool`; workflows execute the corresponding LangGraph. `HandleResponse.mode` tells the frontend "tool / workflow / none"; `tool.data` or `plan.recommendations` carry results.
- Tool handlers in `src/agent/tools/registry.py` return `ToolInvocation` (structured `name/message/data`) for UI rendering.
- Pending state is persisted in the pending store (memory/sqlite/postgres optional) with TTL; pending summaries are injected into the planner to decide follow-up or switch to new questions.

## Config Governance
- Environment-level config (DB URL/API keys/providers) stays in `.env`/`AppConfig`.
- DB object metadata (table names/region lookup sources) is centrally resolved by `src/infra/db_catalog.py`.
- Preferred env overrides are JSON-based:
  - `DB_TABLE_OVERRIDES`
  - `DB_REGION_LOOKUP_CANDIDATES`
- Legacy table keys remain backward-compatible fallbacks.

## Crop Calendar Workflow (Current)
`src/agent/workflows/crop_calendar_graph.py` is the active main flow, replacing the earlier monolithic pipeline:

1. **LLM extraction**: `extract_planting_details(prompt, llm_extract=...)` outputs `PlantingDetailsDraft`.
2. **Missing field check/follow-up**: `list_missing_required_fields(draft)` checks required fields; missing fields enter the follow-up node. User replies are merged, up to two rounds.
3. **Default fill**: if fields are still missing after follow-ups, defaults are applied and recorded in `assumptions`.
4. **Parallel tool context**: `weather_lookup` and `variety_lookup` run in parallel to produce `weather_info`/`variety_info`.
5. **Farming recommendation**: call the external crop calendar API (when configured) using normalized planting data; output is stored in `recommendation_info`, and the workflow composes the final message.

## Tool Notes
- Tools/services support `mock`/`local` providers; variety lookup reads Postgres via `AGRI_DB_URL` when `VARIETY_PROVIDER=local`. Weather can use `WEATHER_PROVIDER=91weather` for external forecasts.
- Tool cache is currently disabled (no-op implementation).
- Variety matching strategy: first recall all approval records by variety name, score using user location and "approval region/suitable region" rules; if multiple high-score records exist, an LLM chooses the best.
 - Historical weather data is fetched via `goso_day` inside the crop calendar workflow.

## Deployment Notes
- Deploy FastAPI with `uvicorn`/`gunicorn` and HTTPS; Chainlit can be reverse-proxied or deployed separately.
- For streaming output, provide WebSocket/SSE and forward LangGraph stream events to the frontend.
- Add structured logging around `router.handle` and tool handlers to analyze routing accuracy.

## Tests
- `python -m unittest` runs the basic test suite.
