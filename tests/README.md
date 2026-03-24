# Tests Guide

## Scope Boundary

This repository keeps `eval` and `tests` intentionally separate:

- `src/eval_platform/`
  - Answers: "Can this model or prompt version ship?"
  - Used for release gates, model comparison, and production-audit review.
  - Focuses on LLM-sensitive behavior and a small set of deterministic session-continuity checks that directly affect model rollout decisions.
- `tests/`
  - Answers: "Did this code change break the system?"
  - Used for service correctness, router state transitions, workflow continuity, formatting, and regression protection.
  - Should remain the main home for system behavior, dependency boundaries, and non-model logic.

In short:

- `eval` is for model quality governance.
- `tests` is for system correctness and regression coverage.

## Directory Layout

Tests are organized by domain instead of a flat `test_*.py` list:

- `tests/weather/`
- `tests/sowing/`
- `tests/variety/`
- `tests/workflow/`
- `tests/router/`
- `tests/domain/`
- `tests/architecture/`

Shared helpers stay at the top level:

- `tests/scenario_loader.py`: loads YAML scenarios from `tests/scenarios/`
- `tests/support.py`: common test router / LLM helpers

Scenario data is grouped under:

- `tests/scenarios/weather/`
- `tests/scenarios/sowing/`
- `tests/scenarios/variety/`
- `tests/scenarios/workflow/`

These two layers have different roles:

- `tests/<domain>/`
  - test executors and assertions
  - Python test files that call services, router logic, workflows, or scenario loaders
- `tests/scenarios/<domain>/`
  - scenario fixtures
  - YAML case data such as prompts, multi-turn steps, expected payloads, and expected messages

Examples:

- [tests/sowing/test_session.py](/f:/workspace/nlp-crop-calendar/tests/sowing/test_session.py)
  - contains the Python test logic for sowing session behavior
- [tests/scenarios/sowing/session.yaml](/f:/workspace/nlp-crop-calendar/tests/scenarios/sowing/session.yaml)
  - contains the replayable sowing session scenarios consumed by tests

In short:

- `tests/<domain>` = how to run and assert the test
- `tests/scenarios/<domain>` = what business scenarios are being tested

## Scenario-Driven Tests

Use `YAML + executor` for tests that are fundamentally user/business scenarios:

- multi-turn session reuse
- follow-up merging
- prompt normalization
- UI/detail formatting
- regression conversation playback

Recommended YAML split:

- `service.yaml`: service-layer scenarios
- `session.yaml`: session-context and multi-turn scenarios
- `ui.yaml`: frontend/render formatting scenarios
- `regression.yaml`: end-to-end regression playback when needed

In this repository, weather uses `service/session/ui`, while other domains currently use the subset they need.

## Recommended Test Categories

Within `tests/`, prefer these categories:

- `service`
  - Domain logic, payload assembly, provider integration behavior, and returned business data.
- `session`
  - Multi-turn continuity, pending follow-up resume, session-context reuse, and stateful domain behavior.
- `regression`
  - Higher-level end-to-end conversation playback for previously broken or high-risk flows.
- `ui`
  - Rendering, display formatting, and user-facing message shaping.
- `router`
  - Cross-domain routing, planner fallback, and executor state transitions.
- `architecture`
  - Dependency boundaries and import hygiene.
- `domain`
  - Pure domain logic and utility-level business rules.
- `infra`
  - Local infrastructure helpers and non-domain adapters.

If a case is primarily about shipping or comparing a model, prefer `src/eval_platform/`.
If a case is primarily about protecting product logic after code changes, prefer `tests/`.

## When To Keep Plain Python Tests

Keep tests as normal Python unit tests when they are mainly:

- low-level pure function checks
- payload builder and validation checks
- architecture/dependency boundary rules
- router/executor internal state transitions that are easier to read as code

## Common Commands

Run all tests:

```bash
python -m unittest
```

Run weather regression:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_weather_regression.ps1
```

Run by domain:

```bash
python -m unittest tests.weather.test_service tests.weather.test_session tests.weather.test_ui tests.weather.test_regression
python -m unittest tests.sowing.test_service tests.sowing.test_session
python -m unittest tests.variety.test_service tests.variety.test_session
python -m unittest tests.workflow.test_service tests.workflow.test_session
```

## Maintenance Rules

- Add new user/business scenarios to YAML first when the test is scenario-shaped.
- Keep executors generic; avoid hardcoding one-off scenarios into the Python test body.
- Prefer placing new test files under the matching domain directory.
- Update docs and scripts when module paths change.
