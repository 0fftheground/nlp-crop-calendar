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

1. **(Optional) Start local OpenTelemetry collector**
   ```bash
   powershell -ExecutionPolicy Bypass -File scripts/run_otel_collector.ps1
   ```
2. **Start backend and Chainlit**
   ```bash
   python run_all.py
   ```
   This command starts `uvicorn` (`src.api.server:app`) and `chainlit run chainlit_app.py --watch` in parallel. Press `Ctrl+C` once to stop.
3. Open the Chainlit URL from the console output and chat with the assistant.

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
FASTAPI_PORT=8000
CHAINLIT_PORT=8001
HOST=0.0.0.0
PUBLIC_BASE_URL=http://127.0.0.1:8000
VARIETY_PROVIDER=local
VARIETY_API_URL=
VARIETY_API_KEY=
WEATHER_PROVIDER=mock
WEATHER_API_URL=
WEATHER_API_KEY=
GROWTH_STAGE_PROVIDER=local
GROWTH_STAGE_API_URL=
GROWTH_STAGE_API_KEY=
GROWTH_STAGE_DB_PATH=
RECOMMENDATION_PROVIDER=mock
RECOMMENDATION_API_URL=
RECOMMENDATION_API_KEY=
```
If `EXTRACTOR_API_KEY` is empty, the extractor falls back to `OPENAI_API_KEY`.
Tools default to `mock`. Variety lookup reads local SQLite by default (`VARIETY_PROVIDER=local`). Growth stage prediction supports:
- `GROWTH_STAGE_PROVIDER=local`: read GDD parameters from SQLite (`GROWTH_STAGE_DB_PATH`, default `resources/gdd.sqlite3`)
To use the external 15-day weather API, set `WEATHER_PROVIDER=91weather` and ensure the request includes `lat`/`lon` (the tool accepts `WeatherQueryInput` JSON with `lat`/`lon`).
Growth-stage prediction and crop calendar workflows use historical weather (`goso_day`) and do not support future dates yet.

To build the local GDD SQLite once:
```bash
python scripts/import_gdd_to_sqlite.py --excel "C:\Users\00778807\Downloads\gdd_new_20250710.xlsx" --db "resources\gdd.sqlite3"
```

## OpenTelemetry (Local Collector)
The API initializes OpenTelemetry when OTLP endpoints are configured. A local collector config is provided in `otel-collector.yaml`.

1. Start the collector:
   ```bash
   powershell -ExecutionPolicy Bypass -File scripts/run_otel_collector.ps1
   ```
2. Set OTEL env vars in `.env` (gRPC default):
   ```
   OTEL_EXPORTER_OTLP_ENDPOINT=localhost:4317
   OTEL_EXPORTER_OTLP_PROTOCOL=grpc
   OTEL_TRACES_EXPORTER=otlp
   OTEL_LOGS_EXPORTER=otlp
   OTEL_SERVICE_NAME=nlp-crop-calendar
   ```
3. Start the app and send a request to `/api/v1/handle`. Traces will print in the collector console and logs will append to `otel-logs.json` (relative to the collector working directory).
4. This project adds spans for each workflow node and each tool invocation (with summarized input/output), so a single request forms a full trace across the workflow.

## Docker Deployment
This project runs in Docker with separate **api** and **chainlit** services, plus **db (Postgres)** and **otel-collector** by default.

1. Ensure `Dockerfile` and `docker-compose.yml` exist (already in this repo).
2. Make sure `.env.docker` includes:
   - `BACKEND_URL=http://api:8000`
   - `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
   - OTLP settings (pointing to `otel-collector:4317`)
3. Build and run:
   ```bash
   docker compose --env-file .env.docker up -d --build
   ```

### Example docker-compose.yml
```yaml
services:
  api:
    build: .
    container_name: nlp-crop-calendar-api
    env_file: .env.docker
    command:
      - python
      - -m
      - uvicorn
      - src.api.server:app
      - --host
      - 0.0.0.0
      - --port
      - "8000"
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy
      otel-collector:
        condition: service_started
    restart: unless-stopped
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  chainlit:
    build: .
    container_name: nlp-crop-calendar-chainlit
    env_file: .env.docker
    command:
      - python
      - -m
      - chainlit
      - run
      - chainlit_app.py
      - --host
      - 0.0.0.0
      - --port
      - "8001"
    ports:
      - "8001:8001"
    depends_on:
      db:
        condition: service_healthy
      api:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test:
        - CMD
        - python
        - -c
        - "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/', timeout=2)"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  db:
    image: postgres:15
    container_name: nlp-crop-calendar-db
    env_file: .env.docker
    environment:
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: ${POSTGRES_DB}
    volumes:
      - db-data:/var/lib/postgresql/data
    restart: unless-stopped
    healthcheck:
      test:
        - CMD-SHELL
        - "pg_isready -U $POSTGRES_USER -d $POSTGRES_DB"
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s

  otel-collector:
    image: otel/opentelemetry-collector:latest
    container_name: otel-collector
    volumes:
      - ./otel-collector.yaml:/etc/otelcol/config.yaml
      - otel-data:/otel-data
    command: ["--config", "/etc/otelcol/config.yaml"]
    environment:
      OTEL_LOG_PATH: /otel-data/otel-logs.json
      OTEL_TRACES_PATH: /otel-data/otel-traces.json
    restart: unless-stopped

volumes:
  db-data:
  otel-data:
```

### Local vs Docker environment files
- Local run uses `.env` (copied from `.env.local` when needed).
- Docker run uses `.env.docker` (default in `docker-compose.yml`).

Suggested commands:
```bash
# local
copy .env.local .env
python run_all.py

# docker
docker compose --env-file .env.docker up -d --build
```

### Notes
- `db` stores Chainlit persistence; other caches remain local `.sqlite3` files inside containers.
- `otel-collector` runs internally (no host ports exposed). Logs/traces are persisted to the `otel-data` volume.

## External Access
To allow other machines to access the app and download CSV exports, set:
- `HOST=0.0.0.0` to listen on all interfaces
- `PUBLIC_BASE_URL=http://<your-host>:8000` so download links are absolute

Start with:
```bash
python run_all.py
```
Ensure your firewall/security group allows access to ports `8000` (FastAPI) and `8001` (Chainlit).

## Development Notes
- `src/agent/router.py` + `src/agent/planner.py` implement Planner+Executor logic; you can adjust the planner prompt or add tools to extend capability (prompt lives in `src/prompts/planner.py`).
- LangGraph state types are in `src/agent/workflows/state.py`; edit nodes/branches in `src/agent/workflows/crop_calendar_graph.py` and `src/agent/workflows/growth_stage_graph.py`.
- Workflow flow: LLM extracts planting info, missing fields trigger follow-ups (max 2 rounds), then weather/variety tools run in parallel, and `farming_recommendation` consumes context to output recommendations.
- Business services live in `src/application/services`; tools are thin adapters calling application services.
- LLM prompts and workflow user-facing text are centralized in `src/prompts` (planner / extraction / workflow copy / tool fallbacks).
- `src/api/server.py` binds HTTP requests to router/graph; extend auth, logging, or persistence as needed.
- An OpenAI API key is required. The system uses `ChatOpenAI` for planning and extraction; extraction can use a lighter model via `EXTRACTOR_*`.
- `growth_stage_prediction` only reads cached results; growth-stage prediction must go through the workflow for extraction/follow-up and then calls local SQLite.
- Crop calendar/growth-stage workflow results build cache keys from normalized `PlantingDetails`; cache hits return immediately.
- To hit growth-stage cache, pass `PlantingDetails` JSON (or JSON containing a `planting` field).
- Follow-up control: pending state is passed into the LLM planner; the LLM decides whether to continue follow-up or switch to a new question; when it selects a new tool/workflow or action=none, pending is cleared.
- Experience memory: for the same `user_id + crop + region`, missing planting fields are auto-reused with TTL; to clear, ask the assistant to clear memory (LLM selects the `memory_clear` tool).
- Infrastructure adapters live in `src/infra` (config, LLM client, structured extraction, etc.).
- Non-agronomy requests return `mode="none"` and skip tools/workflows.
- Variety extraction uses candidate-name matching + fuzzy tokens; data source is `resources/rice_variety_approvals.sqlite3` (override with `VARIETY_DB_PATH`).
- 稻区范围映射用于生育期预测的审定地区匹配，配置文件为 `resources/rice_region_map.json`。

## Recent Updates
- Growth-stage prediction uses the `variety_lookup` tool flow with follow-ups and can prompt for specific approval records when region matching is ambiguous.
- Historical weather (`goso_day`) is used for growth-stage and crop calendar workflows; results are archived locally with a 0.05° grid cache.
- Future sowing dates (>=2026) trigger a re-ask for a valid date before continuing.
- Growth-stage outputs now include the full ordered stage date list.

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


