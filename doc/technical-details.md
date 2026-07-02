# Technical Details

## Contents

- [Overview](#overview)
- [Core Modules](#core-modules)
- [Request Flow](#request-flow)
- [Routing](#routing)
- [Tools And Workflow](#tools-and-workflow)
- [Config](#config)
- [Observability](#observability)
- [Tests](#tests)

## Overview

This document summarizes the main runtime structure of the project.

The system is organized into three main layers:

1. `chainlit_app.py`
   - UI entry
   - sends user requests to the backend
2. `src/api/server.py`
   - API entry
   - exposes the external HTTP interface
3. `src/agent/*`
   - request orchestration
   - routing, state handling, tool/workflow execution

High-level flow:

```text
Chainlit -> FastAPI -> RequestRouter -> Tool / Workflow -> Unified response
```

## Core Modules

- `src/api/server.py`
  - API entry and interaction logging
- `src/agent/router.py`
  - top-level request orchestration
- `src/agent/intent_router.py`
  - rules + fast intent + planner routing
- `src/agent/pending_manager.py`
  - follow-up state lifecycle
- `src/agent/session_context.py`
  - thread-aware contextual resume
- `src/agent/plan_executor.py`
  - tool/workflow execution and input validation
- `src/agent/followup.py`
  - shared follow-up contract
- `src/agent/field_updates.py`
  - shared field override helpers
- `src/agent/tools/*`
  - tool adapters
- `src/agent/workflows/*`
  - LangGraph workflow definitions
- `src/application/services/*`
  - domain-facing application services
- `src/infra/*`
  - config, caches, stores, LLM, DB/http adapters
- `src/schemas/models.py`
  - unified request/response schemas

## Request Flow

Runtime order:

1. receive request
2. load latest session state
3. check `pending`
4. build `session_context` candidate
5. run standalone intent routing
6. resolve thread ownership
7. execute tool or workflow
8. persist interaction and session state

The system is thread-aware, not only session-aware. See [dialogue-orchestration.md](./dialogue-orchestration.md).

## Routing

Routing is layered:

1. `pending`
   - resume field fill / option select / confirmation / clarification
2. `session_context`
   - continue current task thread when confidence is high
3. `IntentRouter`
   - rule route
   - fast intent
   - planner LLM
   - boundary normalization

Important behavior:

- complete standalone prompts should break out of stale context
- confirmation pending only accepts explicit confirmation replies
- ambiguous thread ownership can enter clarification pending

## Tools And Workflow

Main tools:

- `weather_lookup`
- `variety_lookup`
- `sowing_suitability_lookup`
- `plant_plan_list_active`
- `plant_plan_delete`
- `growth_stage_lookup`
- `memory_clear`

Main workflow:

- `crop_calendar_workflow`

Rules of thumb:

- use tools for single-task lookup or short recommendation
- use workflow for multi-step planting-plan generation

## Config

Primary config lives in:

- `.env`
- `.env.docker`
- `src/infra/config.py`

Important groups:

- LLM config
  - `LLM_MODEL`
  - `EXTRACTOR_MODEL`
  - `AUDIT_JUDGE_MODEL`
- DB / cache
  - `DATABASE_URL`
  - `AGRI_DB_URL`
  - `CACHE_DB_URL`
- business API
  - `BUSINESS_API_BASE_URL`
  - endpoint overrides when needed

## Observability

Useful logs:

- `.cache/logs/observability.log`
- `.cache/logs/api_errors.log`

Important stored interaction metadata:

- `request_id`
- `thread_id`
- `parent_interaction_id`
- `continuity_type`
- `continuity_source`
- `dialogue_act`
- `task_type`

These fields support debugging, audit, and bad-case analysis.

## Tests

Main test areas:

- `tests/router/`
- `tests/weather/`
- `tests/sowing/`
- `tests/variety/`
- `tests/workflow/`
- `tests/domain/`
- `tests/architecture/`

Model-governance and online audit are documented separately in [eval-governance.md](./eval-governance.md).
