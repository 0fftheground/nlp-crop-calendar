# Deployment Guide

## Contents

- [Local Run](#local-run)
- [Docker Deploy](#docker-deploy)
- [Verification](#verification)
- [Logs](#logs)

This document covers the minimum runtime setup for local development and Docker deployment.

## Local Run

Use `.env` locally.

Required:

- `OPENAI_API_KEY`
- `DATABASE_URL`
- `AGRI_DB_URL`
- `CACHE_DB_URL`

Recommended:

- `BACKEND_TIMEOUT_SECONDS=90`

If you do not use OTEL locally:

- `OTEL_TRACES_EXPORTER=none`
- `OTEL_LOGS_EXPORTER=none`

Start:

```bash
python run_all.py
```

Useful endpoints:

- API: `http://127.0.0.1:8000/health`
- Chainlit: `http://127.0.0.1:18001`

## Docker Deploy

Use `.env.docker` on the server.

Required:

- `OPENAI_API_KEY`
- `DATABASE_URL`
- `AGRI_DB_URL`
- `CACHE_DB_URL`
- `BACKEND_URL=http://api:8000`

Base stack:

```bash
docker-compose -f docker-compose.yml up -d --build
```

Full stack with observability:

```bash
docker-compose -f docker-compose.yml -f docker-compose.observability.yml up -d --build
```

## Verification

Check service status:

```bash
docker-compose -f docker-compose.yml ps
```

Check API:

```bash
curl http://127.0.0.1:8000/health
```

If needed, verify model endpoint with a real completion request instead of only checking `/health`.

## Logs

Main local log files:

- `.cache/logs/observability.log`
- `.cache/logs/api_errors.log`

In Docker:

```bash
docker-compose -f docker-compose.yml logs --tail=200 api
docker-compose -f docker-compose.yml logs --tail=200 chainlit
docker exec -it nlp-crop-calendar-api tail -n 200 .cache/logs/observability.log
docker exec -it nlp-crop-calendar-api tail -n 200 .cache/logs/api_errors.log
```
