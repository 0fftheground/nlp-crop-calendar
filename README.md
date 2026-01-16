# NLP Crop Calendar (Chainlit + FastAPI + LangGraph)

This project demonstrates an end-to-end flow that generates planting recommendations from farmer questions. Chainlit collects input, FastAPI provides the backend, routing uses an LLM-driven Planner+Executor, and LangGraph handles fixed-step planting workflows. Simple requests are handled by a single tool chosen by the planner, while complex planning goes through LangGraph workflows. **An OpenAI GPT model is required (no mock LLM is provided).**

## Components
- **Chainlit frontend (`chainlit_app.py`)** - Chat UI that sends requests to the backend and shows results/trace.
- **FastAPI backend (`src/api/server.py`)** - Exposes `/api/v1/handle`; the planner decides tool or workflow.
- **Request routing (`src/agent/router.py`)** - Planner+Executor: uses `src/agent/planner.py` to choose tool/workflow and execute, while managing follow-up state.
- **LangGraph workflows (`src/agent/workflows/crop_calendar_graph.py`/`src/agent/workflows/growth_stage_graph.py`)** - Fixed-step planning flows (LLM extraction + follow-up + parallel tools).

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. **Start backend and Chainlit**
   ```bash
   python run_all.py
   ```
   This command starts `uvicorn` (`src.api.server:app`) and `chainlit run chainlit_app.py --watch` in parallel. Press `Ctrl+C` once to stop.
2. Open the Chainlit URL from the console output and chat with the assistant.

## Environment Variables
Create a `.env` file (see `.env.example`) and configure the LLM and tool API parameters, for example:
```
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
EXTRACTOR_PROVIDER=openai
EXTRACTOR_MODEL=gpt-4o-mini
EXTRACTOR_API_KEY=
EXTRACTOR_API_BASE=
EXTRACTOR_TEMPERATURE=0.0
DEFAULT_REGION=global
VARIETY_PROVIDER=local
VARIETY_API_URL=
VARIETY_API_KEY=
WEATHER_PROVIDER=mock
WEATHER_API_URL=
WEATHER_API_KEY=
GROWTH_STAGE_PROVIDER=mock
GROWTH_STAGE_API_URL=
GROWTH_STAGE_API_KEY=
RECOMMENDATION_PROVIDER=mock
RECOMMENDATION_API_URL=
RECOMMENDATION_API_KEY=
```
If `EXTRACTOR_API_KEY` is empty, the extractor falls back to `OPENAI_API_KEY`.
Tools default to `mock`. Variety lookup reads local SQLite by default (`VARIETY_PROVIDER=local`); set `VARIETY_PROVIDER=intranet` to switch to intranet APIs. Other tools can set `*_PROVIDER=intranet` and configure `*_API_URL`/`*_API_KEY` to use intranet APIs; growth stage prediction must use `GROWTH_STAGE_PROVIDER=intranet`.
To use the external 15-day weather API, set `WEATHER_PROVIDER=91weather` and ensure the request includes `lat`/`lon` (the tool accepts `WeatherQueryInput` JSON with `lat`/`lon`).

## Development Notes
- `src/agent/router.py` + `src/agent/planner.py` implement Planner+Executor logic; you can adjust the planner prompt or add tools to extend capability (prompt lives in `src/prompts/planner.py`).
- LangGraph state types are in `src/agent/workflows/state.py`; edit nodes/branches in `src/agent/workflows/crop_calendar_graph.py` and `src/agent/workflows/growth_stage_graph.py`.
- Workflow flow: LLM extracts planting info, missing fields trigger follow-ups (max 2 rounds), then weather/variety tools run in parallel, and `farming_recommendation` consumes context to output recommendations.
- Business services live in `src/application/services`; tools are thin adapters calling application services.
- LLM prompts and workflow user-facing text are centralized in `src/prompts` (planner / extraction / workflow copy / tool fallbacks).
- `src/api/server.py` binds HTTP requests to router/graph; extend auth, logging, or persistence as needed.
- An OpenAI API key is required. The system uses `ChatOpenAI` for planning and extraction; extraction can use a lighter model via `EXTRACTOR_*`.
- `growth_stage_prediction` only reads cached results; growth-stage prediction must go through the workflow for extraction/follow-up/intranet calls.
- Crop calendar/growth-stage workflow results build cache keys from normalized `PlantingDetails`; cache hits return immediately.
- To hit growth-stage cache, pass `PlantingDetails` JSON (or JSON containing a `planting` field).
- Follow-up control: pending state is passed into the LLM planner; the LLM decides whether to continue follow-up or switch to a new question; when it selects a new tool/workflow or action=none, pending is cleared.
- Experience memory: for the same `user_id + crop + region`, missing planting fields are auto-reused with TTL; to clear, ask the assistant to clear memory (LLM selects the `memory_clear` tool).
- Infrastructure adapters live in `src/infra` (config, LLM client, structured extraction, etc.).
- Non-agronomy requests return `mode="none"` and skip tools/workflows.
- Variety extraction uses candidate-name matching + fuzzy tokens; data source is `resources/rice_variety_approvals.sqlite3` (override with `VARIETY_DB_PATH`).

## Frontend client_id (user_id)
To enable cross-session memory without login, generate a UUID in the browser, store it in `localStorage` (or a long-lived cookie), and send it as `user_id` on each request. Keep `session_id` for per-chat isolation if needed.

```html
<script>
const KEY = "client_id";
let clientId = localStorage.getItem(KEY);
if (!clientId) {
  clientId =
    (crypto.randomUUID && crypto.randomUUID()) ||
    (Date.now().toString(36) + Math.random().toString(36).slice(2));
  localStorage.setItem(KEY, clientId);
}

const payload = { prompt, user_id: clientId, session_id };
fetch("/api/v1/handle", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
});
</script>
```

## Tests
```bash
python -m unittest
```
See `TECHNICAL_DETAILS.md` for more details.


