## System Overview

```
Chainlit UI --> FastAPI backend --> Planner (LLM) + Executor (tools + LangGraph)
```

1. **Chainlit (`chainlit_app.py`)** sends user input to `POST /api/v1/handle` with `session_id` for multi-session isolation (optionally `user_id` for cross-session experience memory). The response indicates whether a tool or a LangGraph plan ran, and traces are shown separately.
2. **FastAPI (`src/api/server.py`)** exposes `/health` and `/api/v1/handle`; all requests/responses use unified Pydantic models.
3. **Planner Router (`src/agent/router.py`)** calls the LLM planner to choose tool/workflow/none, then executes and persists follow-up state by `session_id`.
4. **LangGraph (`src/agent/workflows/crop_calendar_graph.py`/`src/agent/workflows/growth_stage_graph.py`)**
   - The crop calendar workflow implements extraction -> follow-up -> parallel tools -> recommendation output.
   - Extraction uses an LLM (structured output) with heuristic fallback; missing fields are asked up to 2 times, and any remaining fields are filled with defaults.
   - Weather, variety, and recommendations are fetched via tools or workflow services (`weather_lookup`/`variety_lookup`/`farming_recommendation` plus historical weather in workflows).

## Core Modules
- `src/infra/config.py` - Reads `.env` and exposes `AppConfig`.
- `src/infra/llm.py` - Creates `ChatOpenAI` for the planner and extractor models.
- `src/infra/llm_extract.py` - Common wrapper for structured extraction.
- `src/infra/cache_keys.py` - Utility for generating cache keys from `PlantingDetails`.
- `src/infra/tool_provider.py` - Provider normalization helpers.
- `src/infra/variety_store.py` - Lightweight variety lookup (`resources/rice_variety_approvals.sqlite3`).
- `src/infra/pending_store.py` - Follow-up state persistence with TTL (memory/sqlite).
- `src/infra/tool_cache.py` - Tool result cache (memory/sqlite).
- `src/infra/weather_cache.py` - Weather series cache (persistable).
- `src/infra/weather_archive_store.py` - Historical weather archive index (SQLite) + local CSV storage.
- `src/infra/interaction_store.py` - Request/response audit records (memory/sqlite).
- `src/infra/planting_choice_store.py` - Experience memory for planting details (keyed by `user_id + crop + region`, TTL).
- `src/infra/variety_choice_store.py` - Experience memory for variety choices (keyed by `user_id + query_key`, TTL).
- `src/prompts/*` - LLM prompts and workflow/tool user copy (planner/extract/fallback prompts).
- Variety retrieval uses candidate-name matching + fuzzy tokens, no embedding/Qdrant.
- `src/schemas/models.py` - Shared schemas (`UserRequest`, `WorkflowResponse`, `ToolInvocation`, `HandleResponse`), `UserRequest` supports `session_id` and optional `user_id`.
- `src/agent/planner.py` - LLM planner that outputs `ActionPlan` (tool/workflow/none) using tool/workflow lists and pending context (prompt in `src/prompts/planner.py`).
- `src/agent/tools/registry.py` - Tool registration and execution (variety/weather/growth-stage/recommendation).
- `src/agent/router.py` - Executes planner decisions, dispatches tools/workflows, and updates follow-up state.
- `src/application/services/*` - Application-layer services (variety/weather/recommendation/crop calendar/planting extraction) used by tools and workflows.
- `src/domain/planting.py` - Domain logic for planting extraction/validation and heuristic rules.
- `src/agent/workflows/state.py` / `crop_calendar_graph.py` / `growth_stage_graph.py` - LangGraph state definition and workflow implementation.
- `src/api/server.py` - FastAPI routes and dependency cache.
- `chainlit_app.py` - UI client.
- `resources/rice_variety_approvals.sqlite3` - Rice variety approvals database (override with `VARIETY_DB_PATH`).

## LangGraph Details
- `StateGraph` is the orchestration skeleton and currently includes `extract`, `ask`, `context`, and `recommend` nodes.
- `GraphState` key fields: `planting_draft`, `missing_fields`, `followup_count`, `weather_info`, `variety_info`, `recommendation_info`.
- Follow-up logic: if missing fields exist, go to `ask`; user replies are merged with the existing draft, up to two rounds; remaining missing fields are filled with defaults before entering `context`.
- Crop calendar and growth-stage workflows cache results by `PlantingDetails`; cache hits short-circuit and return immediately.

Growth-stage workflow specifics:
- Uses the `variety_lookup` tool flow with follow-ups to resolve the exact approval record when needed.
- Uses historical weather via `goso_day` in the workflow (not the generic weather tool).
- If sowing date >= 2026, the workflow asks for a new date before continuing.

## Routing Logic
- `src/agent/router.RequestRouter` uses `PlannerRunner` to output `ActionPlan` (tool/workflow/none) and executes the action.
- Tools are invoked via `execute_tool`; workflows execute the corresponding LangGraph. `HandleResponse.mode` tells the frontend "tool / workflow / none"; `tool.data` or `plan.recommendations` carry results.
- Tool handlers in `src/agent/tools/registry.py` return `ToolInvocation` (structured `name/message/data`) for UI rendering.
- Pending state is persisted in the pending store (memory/sqlite optional) with TTL; pending summaries are injected into the planner to decide follow-up or switch to new questions.
- Experience memory can auto-fill missing planting fields for the same `user_id + crop + region`; memory can be cleared by the LLM selecting the `memory_clear` tool.

## Crop Calendar Workflow (Current)
`src/agent/workflows/crop_calendar_graph.py` is the active main flow, replacing the earlier monolithic pipeline:

1. **LLM extraction**: `extract_planting_details(prompt, llm_extract=...)` outputs `PlantingDetailsDraft`.
2. **Missing field check/follow-up**: `list_missing_required_fields(draft)` checks required fields; missing fields enter the follow-up node. User replies are merged, up to two rounds.
3. **Default fill**: if fields are still missing after follow-ups, defaults are applied and recorded in `assumptions`.
4. **Parallel tool context**: `weather_lookup` and `variety_lookup` run in parallel to produce `weather_info`/`variety_info`.
5. **Farming recommendation**: call `farming_recommendation` with a JSON string containing `planting`, `weather`, and `variety`; output is stored in `recommendation_info`, and the workflow composes the final message.

## Tool Notes
- `growth_stage_prediction` only reads cached results; growth-stage prediction must go through the workflow for extraction/follow-up and uses local SQLite.
- To hit the growth-stage cache, pass `PlantingDetails` JSON (or JSON containing a `planting` field); the cache key is generated by `cache_keys.py`.
- Tools/services support `mock`/`local` providers; variety lookup defaults to local SQLite (`VARIETY_PROVIDER=local`). Weather can use `WEATHER_PROVIDER=91weather` for external forecasts.
- `variety_lookup` / `weather_lookup` / `farming_recommendation` results are cached with TTL to reduce repeated calls.
- Variety matching strategy: first recall all approval records by variety name, score using user location and "approval region/suitable region" rules; if multiple high-score records exist, an LLM chooses the best.
 - Historical weather data is fetched via `goso_day` inside workflows and stored locally with a 0.05° grid cache.

## Deployment Notes
- Deploy FastAPI with `uvicorn`/`gunicorn` and HTTPS; Chainlit can be reverse-proxied or deployed separately.
- For streaming output, provide WebSocket/SSE and forward LangGraph stream events to the frontend.
- Add structured logging around `router.handle` and tool handlers to analyze routing accuracy.

## Tests
- `python -m unittest` runs the basic test suite.
