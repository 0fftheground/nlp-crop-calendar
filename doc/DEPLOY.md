# Deployment Guide

This guide covers two paths:

1. Local startup (development on your machine)
2. Server deployment (base stack / full stack + observability)

The current setup assumes **external Postgres** for data and cache.

## Local Startup

### 1) Prepare `.env`
Use `.env` for local runs (do not use `.env.docker` locally).

Required:
- `OPENAI_API_KEY` (or your LLM key)
- `DATABASE_URL` (Chainlit persistence)
- `AGRI_DB_URL` (agronomy data)
- `CACHE_DB_URL` (pending/tool/interaction/geocode)

Recommended:
- `BACKEND_URL` is not used in local Python run
- `BACKEND_TIMEOUT_SECONDS=90` (or another explicit timeout suitable for your model/provider)

Observability (optional):
- If you do not run an OTEL backend locally, disable exporters:
  - `OTEL_TRACES_EXPORTER=none`
  - `OTEL_LOGS_EXPORTER=none`

### 2) Run locally
```bash
python run_all.py
```

Open:
- API: http://127.0.0.1:8000/health
- Chainlit: http://127.0.0.1:18001

## Server Deployment

### 1) Prepare `.env.docker`
Use `.env.docker` on the server. It is loaded by Docker Compose.

Required:
- `OPENAI_API_KEY`
- `DATABASE_URL`
- `AGRI_DB_URL`
- `CACHE_DB_URL`
- `BACKEND_URL=http://api:8000`

Base stack only (recommended first step):
- `OTEL_TRACES_EXPORTER=none`
- `OTEL_LOGS_EXPORTER=none`

If enabling OTEL collector / observability:
- `OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4317`
- `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`

Optional (but recommended):
- `CHAINLIT_AUTH_*` (enable login)
- `PUBLIC_BASE_URL=http://<server-ip>:8000` (only if you need absolute URLs in responses)
- `BACKEND_TIMEOUT_SECONDS=90` (keeps planner/extractor calls from hanging indefinitely)

### 2) Start base stack (API + Chainlit)
```bash
docker-compose -f docker-compose.yml up -d --build
```

### 3) Start full stack (API + Chainlit + Observability)
```bash
docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

## Common Ops Commands

### Local run
Start:
```bash
python run_all.py
```

Stop:
- Press `Ctrl+C` in the terminal that started the service.

Check ports:
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:18001/
```

If you need to find the local process by port:
```bash
netstat -ano | findstr :8000
netstat -ano | findstr :18001
```

### Docker Compose
Start base stack:
```bash
docker-compose -f docker-compose.yml up -d --build
```

Start full stack:
```bash
docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

Stop services:
```bash
docker-compose -f docker-compose.yml down
```

Stop full stack:
```bash
docker-compose -f docker-compose.yml -f docker-compose.observability.yml down
```

Restart one service:
```bash
docker-compose -f docker-compose.yml restart api
docker-compose -f docker-compose.yml restart chainlit
```

View service status:
```bash
docker-compose -f docker-compose.yml ps
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

View recent logs:
```bash
docker-compose -f docker-compose.yml logs --tail=200 api
docker-compose -f docker-compose.yml logs --tail=200 chainlit
```

Follow logs continuously:
```bash
docker-compose -f docker-compose.yml logs -f api
docker-compose -f docker-compose.yml logs -f chainlit
```

View both main services together:
```bash
docker-compose -f docker-compose.yml logs -f api chainlit
```

Rebuild and restart a single service:
```bash
docker-compose -f docker-compose.yml up -d --build api
docker-compose -f docker-compose.yml up -d --build chainlit
```

Enter a container:
```bash
docker exec -it nlp-crop-calendar-api /bin/sh
docker exec -it nlp-crop-calendar-chainlit /bin/sh
```

Check container health:
```bash
docker inspect --format "{{.Name}} {{.State.Status}} {{.State.Health.Status}}" nlp-crop-calendar-api
docker inspect --format "{{.Name}} {{.State.Status}} {{.State.Health.Status}}" nlp-crop-calendar-chainlit
```

### Log files in the API container
The API process writes local files under `.cache/logs`:
- `api_errors.log`
- `observability.log`

If you are running inside Docker, you can inspect them with:
```bash
docker exec -it nlp-crop-calendar-api ls -lah .cache/logs
docker exec -it nlp-crop-calendar-api tail -n 200 .cache/logs/api_errors.log
docker exec -it nlp-crop-calendar-api tail -n 200 .cache/logs/observability.log
```

Useful events in `observability.log`:
- `request_received`
- `pending_resume`
- `session_candidate_built`
- `session_candidate_selected`
- `llm_extract_call`
- `llm_extract_response`
- `llm_extract_model_error`
- `llm_extract_error`
- `response_ready`

### 4) Open ports on the server
Allow inbound traffic:
- `8000` (API)
- `18001` (Chainlit)
- `3000` (Grafana, only if observability stack is enabled)

Optional (only if other machines send OTLP directly):
- `4317` (OTLP gRPC)
- `4318` (OTLP HTTP)

### 5) Verify
- API health: `http://<server-ip>:8000/health`
- Chainlit: `http://<server-ip>:18001`
- Grafana: `http://<server-ip>:3000` (default `admin/admin`, only if enabled)

Note:
- `/health` only checks API process/config status and does **not** perform a real LLM request.
- Verifying only `OPENAI_API_BASE` network connectivity is **not sufficient**.

### 6) Verify LLM API (recommended)
On the server, validate the LLM endpoint with a real OpenAI-compatible request (base URL + API key + model).

Quick check (list models):
```bash
curl -sS "$OPENAI_API_BASE/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

Recommended check (real chat completion):
```bash
curl -sS "$OPENAI_API_BASE/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
```

Success criteria:
- Endpoint is reachable from the server (DNS/TLS/network OK)
- `OPENAI_API_KEY` is valid
- The provider supports the configured model (default in code: `gpt-4.1-mini`)

## Troubleshooting Request Failures

If the UI only shows `请求失败:`:

1. Find the request chain in `.cache/logs/observability.log`.
   ```bash
   docker exec -it nlp-crop-calendar-api tail -n 200 .cache/logs/observability.log
   ```
2. Check whether the request reached `response_ready`.
   - If yes, the backend finished successfully and the issue is likely outside the main request handler.
   - If no, look at the last event before the gap.
3. Check `.cache/logs/api_errors.log` for uncaught exceptions.
   ```bash
   docker exec -it nlp-crop-calendar-api tail -n 200 .cache/logs/api_errors.log
   ```
4. For extractor-related failures, look specifically for:
   - `llm_extract_model_error`: model construction/config failure
   - `llm_extract_error`: invocation/runtime failure after the call started
5. If OTEL is enabled, inspect the request trace for:
   - `session.contextual_candidate`
   - `session.standalone_plan`
   - `session.resolution`

Notes:
- `interactions` records are written after a successful request response is produced. Failed requests may therefore be absent from the interaction store.
- Repeated `Failed to export logs` messages usually mean OTEL exporters are enabled without a reachable collector; disable them locally if you are not using OTEL.

## Port Conflicts

If a port is already used, change only the **left side** of the mapping in
`docker-compose.yml` / `docker-compose.observability.yml`.

Example:
```yaml
ports:
  - "18000:8000"
  - "18001:18001"
  - "13000:3000"
```

Then update:
- Your browser URLs for Chainlit / Grafana
