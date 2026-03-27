# NLP Crop Calendar (Chainlit + FastAPI + LangGraph)

An end-to-end crop calendar assistant:
- `Chainlit` provides the chat UI
- `FastAPI` provides the backend API
- `Planner + tools + LangGraph workflows` handle routing and execution

This project requires a real OpenAI-compatible LLM API (no mock LLM).

## What It Does
- Simple requests: planner picks a single tool (e.g. weather / variety)
- Complex crop-calendar requests: LangGraph workflow runs fixed steps (extract planting info -> follow-up -> query services -> generate result)
- Growth-stage queries: workflow resolves planting plan and growth-stage result through business APIs
- Session-aware follow-up merge: short follow-up turns such as region/date/operation refinements are merged with the active task before standalone routing is finalized

## Main Components
- `chainlit_app.py`: Chat frontend
- `src/api/server.py`: FastAPI app (`/health`, `/api/v1/handle`)
- `src/agent/router.py`: Request orchestration
- `src/agent/intent_router.py`: Intent planning/routing
- `src/agent/pending_manager.py`: Follow-up state management
- `src/agent/session_context.py`: Session-aware contextual candidate builder and standalone/contextual resolver
- `src/agent/plan_executor.py`: Tool/workflow execution
- `src/agent/followup.py`: Shared follow-up contract/accessors/builders
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
   - Chainlit: `http://127.0.0.1:18001`

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
- `BUSINESS_API_BASE_URL` / `BUSINESS_API_KEY` (required for planting plan / growth-stage / farm weather business APIs)
- `BACKEND_TIMEOUT_SECONDS` (LLM request timeout for planner/extractor calls; default `90`)
- `OTEL_TRACES_EXPORTER` / `OTEL_LOGS_EXPORTER` (set to `none` locally if you do not run an OTEL backend)

Notes:
- If `EXTRACTOR_API_KEY` is empty, extraction falls back to `OPENAI_API_KEY`.
- `/health` does not perform a real LLM request; it only checks API status/config.
- Current business-data access for planting plans, growth-stage results, and farm weather is API-first only; there is no DB fallback for those paths.

## DB Config Strategy

Database table metadata is now managed centrally. Prefer:

- `DB_REGION_LOOKUP_CANDIDATES` (JSON array):
  - Example: `[{"table":"public.agri_region","id_column":"id","name_column":"name"}]`

Legacy env keys such as `VARIETY_DB_TABLE` are still supported as fallback, but no longer recommended for new deployments.

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

The runtime image excludes `tests/`, `src/eval_platform/`, and `src/eval_assets/` via `.dockerignore`; model-evaluation and test assets stay out of production containers by default.

### 3) Verify
- API: `http://<server-ip>:8000/health`
- Chainlit: `http://<server-ip>:18001`
- Grafana: `http://<server-ip>:3000` (if enabled)

Recommended LLM check (real request, not just URL connectivity):
```bash
curl -sS "$OPENAI_API_BASE/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4.1-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
```

More deployment details:
- `doc/deploy.md`

## Development Notes
- Planner logic: `src/agent/planner.py`, `src/agent/router.py`
- Session routing/merge: `src/agent/session_context.py`
- Workflow state/nodes: `src/agent/workflows/state.py`, `src/agent/workflows/*.py`
- Follow-up state contract: `src/agent/followup.py` and `src/agent/pending_manager.py`
  - Unified keys are `draft`, `options`, `missing_fields`, `followup_count`, `pending_message`
- Prompts and user-facing workflow copy: `src/prompts/`
- Infrastructure adapters (LLM, DB, cache, config): `src/infra/`

## Session Routing
- Request handling now resolves turns in this order: `pending` resume -> `session_context` contextual candidate -> standalone intent routing -> final standalone/contextual resolution.
- Contextual follow-up merge is currently supported for:
  - `weather_lookup`
  - `variety_lookup`
  - `sowing_suitability_lookup`
  - `plant_plan_list_active` follow-ups into `plant_plan_delete` / `growth_stage_lookup`
  - `crop_calendar_workflow` follow-ups into itself, `plant_plan_delete`, or `growth_stage_lookup`
  - `growth_stage_lookup` follow-ups into itself
- `memory_clear` intentionally does not participate in contextual merge.

## Observability And Debugging
- Local request diagnostics are written to `.cache/logs/observability.log` and `.cache/logs/api_errors.log`.
- The router writes session-resolution details into the active trace/span under:
  - `session.contextual_candidate`
  - `session.standalone_plan`
  - `session.resolution`
- Structured extraction logs:
  - `llm_extract_call`
  - `llm_extract_response`
  - `llm_extract_model_error`
  - `llm_extract_error`
- If you do not run an OTEL backend locally, set `OTEL_TRACES_EXPORTER=none` and `OTEL_LOGS_EXPORTER=none` to avoid exporter noise such as `Failed to export logs`.

## Tests

```bash
python -m unittest
```

Recommended targeted runs:

- Weather regression:
  ```bash
  powershell -ExecutionPolicy Bypass -File scripts/run_weather_regression.ps1
  ```
- Scenario-driven suites:
  ```bash
  python -m unittest tests.weather.test_service tests.weather.test_session tests.weather.test_ui tests.weather.test_regression
  python -m unittest tests.sowing.test_service tests.sowing.test_session
  python -m unittest tests.variety.test_service tests.variety.test_session
  python -m unittest tests.workflow.test_service tests.workflow.test_session
  python -m unittest tests.router.test_planner tests.router.test_intent_rules tests.infra.test_llm_extract
  ```

Current test layout:

- `tests/weather/`: weather service, session, UI, regression
- `tests/sowing/`: sowing suitability service and session reuse
- `tests/variety/`: variety service and session reuse
- `tests/workflow/`: growth-stage and crop-calendar workflow/service tests
- `tests/router/`: router / planner / intent-rule tests
- `tests/domain/`: domain and payload-building tests
- `tests/architecture/`: dependency boundary checks
- `tests/scenarios/<domain>/`: YAML scenario files used by scenario-driven tests

Scenario-driven tests follow a `YAML + Python executor` pattern:

- YAML keeps user scenarios, follow-up turns, expected payloads, and expected copy
- Python test files stay small and focus on replaying scenarios plus assertions

See also:

- `doc/tests.md`

## LLM Eval

There is now a minimal eval runner for comparing real-model output quality on three tasks:

- `planner`
- `extractor` (planting info extraction)
- `variety_match`

The eval system is organized into two lines:

- `expert`: business-expert-maintained release gates and regression sets
- `production_audit`: deidentified real-interaction samples for AI judge review and human spot checks

Boundary with `tests/`:

- `src/eval_platform` answers whether a model or prompt variant is safe to ship.
- `tests/` answers whether a code change broke routing, services, workflows, or state handling.
- Session continuity appears in both places on purpose:
  - `src/eval_platform` keeps only a small release-gating subset such as `session_context` and `followup_resume`
  - `tests/` remains the main system-regression surface for broader multi-turn behavior

Governance profiles are defined in `src/eval_assets/governance.yaml`.

`production_audit` also has a closed-loop utility pipeline:

1. sample deidentified interactions into audit batches
2. run AI judge to generate review files
3. build a human spot-check queue
4. export expert promotion candidates from confirmed issues

Run the hard release gate:

```bash
python -m src.eval_platform run --profile expert_blocking_gate
```

Run the broader expert regression set:

```bash
python -m src.eval_platform run --profile expert_regression_gate
```

Run the production-audit line:

```bash
python -m src.eval_platform run --profile production_audit_review
```

Compare candidate models on the expert regression line:

```bash
python -m src.eval_platform run \
  --profile expert_regression_gate \
  --llm-model gpt-5-mini \
  --extractor-model gpt-5-mini \
  --json-out .cache/eval/release-regression.json
```

Run a full baseline-vs-candidate release comparison:

```bash
python -m src.eval_platform compare \
  --baseline-llm-model gpt-4.1-mini \
  --baseline-extractor-model gpt-4.1-mini \
  --candidate-llm-model gpt-5-mini \
  --candidate-extractor-model gpt-5-mini \
  --json-out .cache/eval/release-compare.json
```

If `--json-out` is omitted, compare now writes to `.cache/eval/release_compare/latest.json` by default.
The terminal output is intentionally reduced to a short summary plus the JSON path; detailed per-dataset results live in the JSON file.

`compare` now detects which model dimension actually changed and only runs the impacted tasks:

- `llm-model` changes: `planner`, `variety_match`
- `extractor-model` changes: `extractor`, `workflow_extract`
- deterministic continuity tasks such as `session_context` and `followup_resume` are skipped in model-vs-model compare because they do not depend on model choice

If baseline and candidate resolve to the same `llm-model` and `extractor-model`, `compare` exits early without running dataset comparisons.

If you only want to compare one dimension, explicitly pin the other one on both sides.

Compare only `llm-model`:

```bash
python -m src.eval_platform compare \
  --baseline-llm-model gpt-4.1-mini \
  --candidate-llm-model gpt-5-mini \
  --baseline-extractor-model gpt-4.1-mini \
  --candidate-extractor-model gpt-4.1-mini
```

Compare only `extractor-model`:

```bash
python -m src.eval_platform compare \
  --baseline-llm-model gpt-4.1-mini \
  --candidate-llm-model gpt-4.1-mini \
  --baseline-extractor-model gpt-4.1-mini \
  --candidate-extractor-model gpt-5-mini
```

If you omit one side, that model falls back to the current environment configuration, so explicit pinning is safer for controlled comparisons.

If a provided model name does not exist or is not accessible on the configured OpenAI-compatible endpoint, the experiment now stops during preflight validation before any eval datasets run.

Notes:

- `LLM_MODEL` now controls the shared chat model used by planner and variety match.
- `EXTRACTOR_MODEL` controls the structured extraction model.
- `AUDIT_JUDGE_MODEL` controls the production-audit AI judge model. If it is empty, audit judge falls back to `LLM_MODEL`.
- Dataset grading is subset-based: only the fields in `expected` are scored.
- `blocking` cases are hard gates; `regression` cases are broader comparison sets; `audit` cases are monitor-only by default.
- Expert gates now also cover deterministic session continuity via `session_context` and `followup_resume` datasets.
- Expert gates now also cover crop-calendar workflow extraction via `workflow_extract`.
- Eval summaries now include `avg_latency_ms`, `p95_latency_ms`, and `estimated_*_tokens`.
- `expert_regression_gate` also checks relative latency and token regression against the baseline model.
- Token metrics are lightweight estimates from the active model tokenizer; there is no pricing-table-based cost calculation yet.
- Detailed governance rules, the end-to-end operating flow, and an operator guide for `run / compare / audit / promote` live in `doc/eval-governance.md`.

Production-audit closed loop:

- `audit run-latest`
  - one-shot command
  - already includes: `sample -> judge -> review-queue`
- manual steps below
  - use only when you want to rerun one stage or inspect each step separately
- post-review steps
  - happen after human review
  - include CSV export/import and promotion

```bash
python -m src.eval_platform audit sample --limit 50 --days 30 --out-dir .cache/eval/production_audit/batches/manual
python -m src.eval_platform audit judge --batch .cache/eval/production_audit/batches/manual/planner.yaml --batch .cache/eval/production_audit/batches/manual/extractor.yaml --batch .cache/eval/production_audit/batches/manual/variety_match.yaml --out-dir .cache/eval/production_audit/reviews
python -m src.eval_platform audit review-queue --review .cache/eval/production_audit/reviews/planner.review.yaml --review .cache/eval/production_audit/reviews/extractor.review.yaml --review .cache/eval/production_audit/reviews/variety_match.review.yaml --out-dir .cache/eval/production_audit/queues
python -m src.eval_platform audit export-csv --queue .cache/eval/production_audit/queues/planner.review.queue.yaml --out-dir .cache/eval/production_audit/csv
python -m src.eval_platform audit import-csv --csv .cache/eval/production_audit/csv/planner.review.queue.csv
python -m src.eval_platform audit promote --review .cache/eval/production_audit/reviews/planner.review.yaml --review .cache/eval/production_audit/reviews/extractor.review.yaml --review .cache/eval/production_audit/reviews/variety_match.review.yaml --out-dir .cache/eval/production_audit/promotions
```

One-shot latest production-audit cycle:

```bash
python -m src.eval_platform audit run-latest --limit 50 --days 30 --out-dir .cache/eval/production_audit/runs/latest
```

`run-latest` stops at `queues/`; it does not export CSV, promote review records, or import anything back into `expert`.

Production-audit sampling now uses a persisted cursor in `.state/eval/production_audit/sampling_state.json`.
That means repeated `sample` / `run-latest` calls continue from the last sampled `(created_at, id)` watermark instead of always re-reading the latest rows.
Use `--reset-cursor` to bootstrap again from the current date window.

Import promotion candidates and rerun expert gates:

```bash
python -m src.eval_platform promote \
  --promotion .cache/eval/production_audit/promotions/planner.review.planner.promotion.yaml \
  --rerun-profile expert_blocking_gate \
  --rerun-profile expert_regression_gate
```

By default, `promote` also removes matching cases from `src/eval_assets/production_audit/` after they are imported into `expert`, so the same sample does not stay in both lines. Use `--keep-production-audit` if you want to retain the audit copy.

Default `.cache/eval` layout:

- `.cache/eval/release_compare/`
  - compare JSON outputs
- `.cache/eval/production_audit/batches/`
  - `audit sample` outputs
- `.cache/eval/production_audit/runs/`
  - `audit run-latest` full one-shot runs
- `.cache/eval/production_audit/reviews/`
  - `audit judge` outputs
- `.cache/eval/production_audit/queues/`
  - `audit review-queue` outputs
- `.cache/eval/production_audit/csv/`
  - Excel-friendly CSV exports for human review
- `.cache/eval/production_audit/promotions/`
  - `audit promote` outputs

Asset layout:

- `src/eval_assets/expert/`: offline expert-maintained release datasets
- `src/eval_assets/production_audit/`: production-audit sample datasets and templates

PowerShell wrappers are also available:

- `scripts/run_eval_release_compare.ps1`
- `scripts/run_production_audit_cycle.ps1`
- `scripts/register_production_audit_task.ps1`
- `scripts/import_promotion_candidates.ps1`

`sample` also emits `*.context_dependent.yaml` batches for short follow-up turns such as `芜湖呢`.
Those files are marked `judge_only`: they carry a deidentified multi-turn `context_window` from the same session for AI judge review, but are not counted as deterministic single-turn replay cases.

## More Docs
- Deployment details: `doc/deploy.md`
- Technical details: `doc/technical-details.md`
- Test organization and scope boundary: `doc/tests.md`
