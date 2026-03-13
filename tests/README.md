# Tests Guide

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
