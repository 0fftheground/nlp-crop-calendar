# Tests Guide

## Contents

- [Boundary](#boundary)
- [Layout](#layout)
- [Common Commands](#common-commands)
- [Rules](#rules)

This document describes the scope and layout of the repository's system and regression tests.

## Boundary

`tests/` and `src/eval_platform/` serve different purposes:

- `src/eval_platform/`
  - model quality governance
  - release comparison
  - production audit
- `tests/`
  - system correctness
  - router/workflow behavior
  - regression protection

## Layout

Main test domains:

- `tests/weather/`
- `tests/sowing/`
- `tests/variety/`
- `tests/workflow/`
- `tests/router/`
- `tests/domain/`
- `tests/architecture/`

Scenario fixtures:

- `tests/scenarios/weather/`
- `tests/scenarios/sowing/`
- `tests/scenarios/variety/`
- `tests/scenarios/workflow/`

Helpers:

- `tests/scenario_loader.py`
- `tests/support.py`

## Common Commands

Run all tests:

```bash
python -m unittest
```

Run by domain:

```bash
python -m unittest tests.weather.test_service tests.weather.test_session tests.weather.test_ui
python -m unittest tests.sowing.test_service tests.sowing.test_session
python -m unittest tests.variety.test_service tests.variety.test_session
python -m unittest tests.workflow.test_service tests.workflow.test_session
```

Run weather regression script:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_weather_regression.ps1
```

## Rules

- put scenario-shaped cases in YAML first
- keep Python executors generic
- keep domain-specific tests under the matching directory
- use `tests/` for product/system behavior
- use `src/eval_platform/` for model-governance cases
