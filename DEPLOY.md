# Deployment Guide

This guide covers two paths:

1. Local startup (development on your machine)
2. Server deployment (full stack + observability)

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
- Chainlit: http://127.0.0.1:8001

## Server Deployment

### 1) Prepare `.env.docker`
Use `.env.docker` on the server. It is loaded by Docker Compose.

Required:
- `OPENAI_API_KEY`
- `DATABASE_URL`
- `AGRI_DB_URL`
- `CACHE_DB_URL`
- `BACKEND_URL=http://api:8000`
- `OTEL_EXPORTER_OTLP_ENDPOINT=otel-collector:4317`
- `OTEL_EXPORTER_OTLP_PROTOCOL=grpc`

Optional (but recommended):
- `CHAINLIT_AUTH_*` (enable login)
- `PUBLIC_BASE_URL=http://<server-ip>:8000` (only if you need absolute URLs in responses)

### 2) Start full stack (API + Chainlit + Observability)
```bash
docker compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

### 3) Open ports on the server
Allow inbound traffic:
- `8000` (API)
- `8001` (Chainlit)
- `3000` (Grafana)

Optional (only if other machines send OTLP directly):
- `4317` (OTLP gRPC)
- `4318` (OTLP HTTP)

### 4) Verify
- API health: `http://<server-ip>:8000/health`
- Chainlit: `http://<server-ip>:8001`
- Grafana: `http://<server-ip>:3000` (default `admin/admin`)

Note:
- `/health` only checks API process/config status and does **not** perform a real LLM request.
- Verifying only `OPENAI_API_BASE` network connectivity is **not sufficient**.

### 5) Verify LLM API (recommended)
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

## Port Conflicts

If a port is already used, change only the **left side** of the mapping in
`docker-compose.yml` / `docker-compose.observability.yml`.

Example:
```yaml
ports:
  - "18000:8000"
  - "18001:8001"
  - "13000:3000"
```

Then update:
- Your browser URLs for Chainlit / Grafana
