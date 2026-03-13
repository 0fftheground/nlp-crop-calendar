## System Overview

```
Chainlit UI --> FastAPI backend --> Planner (LLM) + Executor (tools + LangGraph)
```

1. **Chainlit (`chainlit_app.py`)** sends user input to `POST /api/v1/handle` with `session_id` for multi-session isolation (optionally `user_id` for user-level context). The response indicates whether a tool or a LangGraph plan ran, and traces are shown separately.
2. **FastAPI (`src/api/server.py`)** exposes `/health` and `/api/v1/handle`; all requests/responses use unified Pydantic models.
3. **Planner Router (`src/agent/router.py`)** calls the LLM planner to choose tool/workflow/none, then executes and persists follow-up state by `session_id`.
4. **LangGraph (`src/agent/workflows/crop_calendar_graph.py`/`src/agent/workflows/growth_stage_graph.py`)**
   - The crop calendar workflow implements extraction -> follow-up -> external crop calendar API -> recommendation output.
   - The growth-stage workflow implements extraction -> follow-up -> business API lookup (planting plan + growth-stage result) -> response formatting.
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
- `src/agent/followup.py` - Shared follow-up contract, accessors, renderers, and payload builders.
- `src/agent/plan_executor.py` - Tool/workflow execution and validation path.
- `src/application/services/*` - Application-layer services (variety/weather/recommendation/crop calendar/planting extraction) used by tools and workflows.
- `src/application/ports.py` / `src/application/adapters.py` - App-level Port/Adapter boundary for config/sql/http dependencies.
- `src/domain/planting.py` + `src/domain/planting_models.py` - Domain logic and models for planting extraction/validation.
- `src/agent/workflows/state.py` / `crop_calendar_graph.py` / `growth_stage_graph.py` - LangGraph state definition and workflow implementation.
- `src/api/server.py` - FastAPI routes and dependency cache.
- `chainlit_app.py` - UI client.
- 品种与区域辅助查询仍可通过 Postgres 读取（`AGRI_DB_URL`，或由 `AGRI_DB_HOST/PORT/NAME/USER/PASSWORD/SSLMODE` 拼接）；种植计划、生育期结果、农场天气业务数据改由 HTTP business API 获取。

## Agent Module Map
- `src/agent/router.py`
  - Top-level request entry for the agent layer.
  - Wires together rule engine, fast intent, LLM planner, pending manager, and executor.
- `src/agent/intent_router.py`
  - Decision layer for intent routing.
  - Combines rule hits, fast-intent results, planner output, and a small in-memory cache.
- `src/agent/intent_rules.py`
  - Rule-engine implementation for `resources/intent_rules.json`.
  - Supports `priority`, `any`, `all`, `regex`, `negative`, and hot-reload by file mtime.
- `src/agent/fast_intent.py`
  - Lightweight intent classifier used as a high-confidence fast path before the full planner.
  - Returns `tool/workflow/none` plus confidence and optional structured input.
- `src/agent/planner.py`
  - Main LLM planner.
  - Builds the full routing prompt from tool/workflow specs and pending summary, and normalizes model output into `ActionPlan`.
- `src/agent/input_specs.py`
  - Canonical input-schema registry for tools and workflows.
  - Defines which Pydantic model each action uses, required fields, field labels, and how validated input is converted back into prompt/json payloads.
- `src/agent/pending_manager.py`
  - Persistence-facing lifecycle manager for follow-up state.
  - Decides whether a user turn should resume pending work or start a new topic.
- `src/agent/followup.py`
  - Shared protocol layer for follow-up payloads.
  - Centralizes accessors, option parsing, message rendering, pending summaries, and builder helpers for tool/workflow follow-up state.
- `src/agent/plan_executor.py`
  - Executes the selected action after routing.
  - Applies input validation, resumes pending tool/workflow runs, and invokes the concrete tool or LangGraph workflow.
- `src/agent/tools/registry.py`
  - Tool registration, dispatch, tracing, and cache interaction.
  - Hides follow-up tool responses from tool-result cache reuse.
- `src/agent/tools/weather.py`, `src/agent/tools/variety.py`, `src/agent/tools/plant_plan.py`, `src/agent/tools/memory.py`
  - Thin adapters around application services.
  - Keep tool-layer logic minimal and return canonical `ToolInvocation`.
- `src/agent/workflows/state.py`
  - Shared LangGraph state contract.
- `src/agent/workflows/common.py`
  - Workflow-shared helpers such as draft coercion, fallback planting defaults, and LLM extraction wrapper functions.
- `src/agent/workflows/crop_calendar_graph.py`
  - Main crop-calendar workflow graph.
- `src/agent/workflows/growth_stage_graph.py`
  - Growth-stage query workflow graph.
- `src/agent/workflows/registry.py`
  - Workflow registration/lookup used by the router/executor layer.

## LangGraph Details
- `StateGraph` is the orchestration skeleton; crop calendar uses `extract`/`ask`/`context`/`recommend`, growth-stage uses `extract`/`ask`/`predict`.
- `GraphState` key fields: `draft`, `options`, `missing_fields`, `followup_count`, `weather_info`, `variety_info`, `recommendation_info`.
- Follow-up logic: if missing fields exist, go to `ask`; user replies are merged with the existing draft, up to two rounds; remaining missing fields are filled with defaults before entering `context`.
- Follow-up state contract is unified across tools and workflows:
  - canonical keys are `draft`, `options`, `missing_fields`, `followup_count`, `pending_message`
  - `PendingManager`, planner summaries, tool cache skip-rules, and workflow ask nodes all consume that shared contract via `src/agent/followup.py`
- Crop calendar workflow has cache hooks keyed by `PlantingDetails` (currently disabled via `tool_cache`).

Growth-stage workflow specifics:
- Parses user variety/plan info, queries business APIs for planting plans; if multiple matches, asks the user to pick one, then fetches the growth-stage result API by `plan_id`.
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
- Business API endpoints for planting plans / growth-stage / farm weather are configured via:
  - `BUSINESS_API_BASE_URL`
  - `BUSINESS_API_KEY`
  - optional explicit endpoint overrides such as `PLANTING_PLAN_SEARCH_API_URL`, `PLANTING_PLAN_ACTIVE_API_URL`, `PLANTING_PLAN_DETAIL_API_URL`, `FARM_WEATHER_API_URL`
- Region lookup sources can be configured via `DB_REGION_LOOKUP_CANDIDATES`.
- Legacy table fallback currently only remains for `VARIETY_DB_TABLE`.

## Crop Calendar Workflow (Current)
`src/agent/workflows/crop_calendar_graph.py` is the active main flow, replacing the earlier monolithic pipeline:

1. **LLM extraction**: `extract_planting_details(prompt, llm_extract=...)` outputs `PlantingDetailsDraft`.
2. **Missing field check/follow-up**: `list_missing_required_fields(draft)` checks required fields; missing fields enter the follow-up node. User replies are merged, up to two rounds.
3. **Default fill**: if fields are still missing after follow-ups, defaults are applied and recorded in `assumptions`.
4. **Parallel tool context**: `weather_lookup` and `variety_lookup` run in parallel to produce `weather_info`/`variety_info`.
5. **Farming recommendation**: call the external crop calendar API (when configured) using normalized planting data; output is stored in `recommendation_info`, and the workflow composes the final message.

## Tool Notes
- Tools/services support `mock`/`local` providers where still applicable; variety lookup reads Postgres via `AGRI_DB_URL` when `VARIETY_PROVIDER=local`.
- Planting plan search, growth-stage result lookup, and farm weather business-data access now use HTTP business APIs only; those paths no longer keep DB fallback branches.
- Tool cache is currently disabled (no-op implementation).
- Variety matching strategy: first recall all approval records by variety name, score using user location and "approval region/suitable region" rules; if multiple high-score records exist, an LLM chooses the best.
 - Historical weather data is fetched via `goso_day` inside the crop calendar workflow.

## Deployment Notes
- Deploy FastAPI with `uvicorn`/`gunicorn` and HTTPS; Chainlit can be reverse-proxied or deployed separately.
- For streaming output, provide WebSocket/SSE and forward LangGraph stream events to the frontend.
- Add structured logging around `router.handle` and tool handlers to analyze routing accuracy.

## Tests
- `python -m unittest` runs the basic test suite.
- Test modules are organized by domain:
  - `tests/weather/`
  - `tests/sowing/`
  - `tests/variety/`
  - `tests/workflow/`
  - `tests/router/`
  - `tests/domain/`
  - `tests/architecture/`
- Scenario-driven suites use YAML fixtures under `tests/scenarios/<domain>/`.
- Shared test utilities live in:
  - `tests/scenario_loader.py`
  - `tests/support.py`
- Weather regression entrypoint:
  - `scripts/run_weather_regression.ps1`
