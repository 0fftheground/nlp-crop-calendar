## System Overview

```
Chainlit UI --> FastAPI backend --> (LangChain Tools | LangGraph Workflow)
```

1. **Chainlit (`chainlit_app.py`)** posts user prompts to `POST /api/v1/handle`. The response tells the UI whether a standalone tool ran or whether a LangGraph plan was produced. Debug traces are shown in a separate message.
2. **FastAPI (`src/server.py`)** exposes `/health`, `/api/v1/plan` (always run LangGraph), and `/api/v1/handle` (run router). All handlers share the same Pydantic request models.
3. **Router (`src/router.py`)** asks the LLM (via `src/tool_selector.py`) whether a quick tool should run; otherwise it falls back to the LangGraph workflow runner.
4. **LangGraph (`src/workflows/crop_graph.py`)**
   - `analysis_node`: prompts the LLM to emit normalized `FarmerQuery`.
   - `plan_node`: maps query attributes to tasks in `CROP_PLAYBOOK`.
   - `finalize_node`: builds a markdown summary for clients.

## Key Modules
- `src/config.py` – loads `.env`, exposes `AppConfig`.
- `src/llm.py` – instantiates `ChatOpenAI` using the configured API key (mock mode removed).
- `src/models.py` – shared schemas (`PlanRequest`, `PlanResponse`, `ToolInvocation`, `HandleResponse`).
- `src/tools.py` – registry of executable tools (weather lookup, soil sampling guidance, market price lookup).
- `src/tool_selector.py` – prompts the LLM to pick `"tool"` or `"workflow"` plus the tool name.
- `src/router.py` – orchestrates “tool vs workflow” decision using the selector and executes the chosen path.
- `src/knowledge_base.py` – stage-aware agronomy facts consumed by the planner.
- `src/workflows/state.py` / `crop_graph.py` – LangGraph state types and compiled workflow.
- `src/server.py` – FastAPI routers and dependency caching.
- `chainlit_app.py` – UI client.

## LangGraph Notes
- Built with `StateGraph` and cached per process. Add nodes via `graph.add_node(...)` and wire edges in `build_graph`.
- Graph input currently only needs `user_prompt`; extend `GraphState` for richer contexts (e.g., historical weather).
- `plan_node` uses simple rule filtering; swap in retrieval or vector search if more sophisticated planning is required.

## Routing Logic
- `src/tool_selector.ToolSelector` uses the LLM to evaluate the JSON schema `{"action":"tool|workflow","tool":"name","reason":"..."}`. Update the system prompt to tweak routing bias.
- If the selector outputs a tool name the backend executes it; otherwise the LangGraph workflow runs. `HandleResponse.mode` informs clients which path was executed; `tool.data` is free-form JSON for additional detail.

## Deployment Tips
- Serve FastAPI via `uvicorn`/`gunicorn` behind HTTPS; Chainlit can be reverse-proxied or hosted separately.
- For streaming, expose a websocket or Server-Sent Events endpoint that forwards LangGraph stream events.
- Instrument `router.handle` and tool handlers with structured logging to monitor routing accuracy.
