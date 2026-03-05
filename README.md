# NLP Crop Calendar (Chainlit + FastAPI + LangGraph)

An end-to-end crop calendar assistant:
- `Chainlit` provides the chat UI
- `FastAPI` provides the backend API
- `Planner + tools + LangGraph workflows` handle routing and execution

This project requires a real OpenAI-compatible LLM API (no mock LLM).

## What It Does
- Simple requests: planner picks a single tool (e.g. weather / variety)
- Complex crop-calendar requests: LangGraph workflow runs fixed steps (extract planting info -> follow-up -> query services -> generate result)
- Growth-stage queries: workflow resolves planting plan and reads prediction results from Postgres

## Main Components
- `chainlit_app.py`: Chat frontend
- `src/api/server.py`: FastAPI app (`/health`, `/api/v1/handle`)
- `src/agent/router.py`: Request orchestration
- `src/agent/intent_router.py`: Intent planning/routing
- `src/agent/pending_manager.py`: Follow-up state management
- `src/agent/plan_executor.py`: Tool/workflow execution
- `src/agent/workflows/`: Crop calendar / growth-stage workflows
- `src/application/services/`: Business services
- `src/agent/tools/`: Thin tool adapters

## Quick Start (Local)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Create `.env` from `.env.example` and fill required values (see next section).
2. Start the app:
   ```bash
   python run_all.py
   ```
3. Open:
   - API health: `http://127.0.0.1:8000/health`
   - Chainlit: `http://127.0.0.1:8001`

## Environment Variables (Minimal)

Do not copy a huge list manually. Start from:
- Local: `.env.example`
- Docker: `.env.docker`

Minimum required for most real runs:
- `OPENAI_API_KEY`
- `DATABASE_URL` (Chainlit persistence)
- `AGRI_DB_URL` (agronomy data)
- `CACHE_DB_URL` (pending/tool/interaction/geocode cache)

Common optional values:
- `OPENAI_API_BASE` (OpenAI-compatible gateway/proxy base URL, usually ends with `/v1`)
- `PUBLIC_BASE_URL`
- `AMAP_API_KEY` (if using geocoding / weather flows that need it)

Notes:
- If `EXTRACTOR_API_KEY` is empty, extraction falls back to `OPENAI_API_KEY`.
- `/health` does not perform a real LLM request; it only checks API status/config.

## DB Config Strategy

Database table metadata is now managed centrally. Prefer:

- `DB_TABLE_OVERRIDES` (JSON object):
  - Example: `{"variety":"agri_rice_variety","weather":"agri_weather","growth_stage_forecast":"agri_growth_stage_forecast","planting_plan":"agri_plant_plan"}`
- `DB_REGION_LOOKUP_CANDIDATES` (JSON array):
  - Example: `[{"table":"agri_region","id_column":"region_id","name_column":"region_name"}]`

Legacy env keys (such as `VARIETY_DB_TABLE`, `WEATHER_DB_TABLE`) are still supported as fallback, but no longer recommended for new deployments.

## Docker Deployment (Server)

This repo provides:
- `docker-compose.yml` (API + Chainlit + OTEL Collector)
- `docker-compose.observability.yml` (Tempo + Loki + Grafana)

### 1) Prepare `.env.docker`
At minimum, set:
- `OPENAI_API_KEY`
- `DATABASE_URL`
- `AGRI_DB_URL`
- `CACHE_DB_URL`
- `BACKEND_URL=http://api:8000`
- `PUBLIC_BASE_URL=http://<server-ip>:8000` (if you need absolute URLs)

If using an OpenAI-compatible proxy:
- `OPENAI_API_BASE=https://<host>/v1`

### 2) Start services

Full stack (recommended):
```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

App only (no Grafana/Tempo/Loki):
```bash
docker compose -f docker-compose.yml up -d --build
```

### 3) Verify
- API: `http://<server-ip>:8000/health`
- Chainlit: `http://<server-ip>:8001`
- Grafana: `http://<server-ip>:3000` (if enabled)

Recommended LLM check (real request, not just URL connectivity):
```bash
curl -sS "$OPENAI_API_BASE/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
```

More deployment details:
- `DEPLOY.md`

## Development Notes
- Planner logic: `src/agent/planner.py`, `src/agent/router.py`
- Workflow state/nodes: `src/agent/workflows/state.py`, `src/agent/workflows/*.py`
- Prompts and user-facing workflow copy: `src/prompts/`
- Infrastructure adapters (LLM, DB, cache, config): `src/infra/`

## Tests

```bash
python -m unittest
```

## More Docs
- Deployment details: `DEPLOY.md`
- Technical details: `TECHNICAL_DETAILS.md`
