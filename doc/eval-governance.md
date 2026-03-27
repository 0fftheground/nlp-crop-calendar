# Eval Governance

## Contents

- [Overview](#overview)
- [Lines And Profiles](#lines-and-profiles)
- [When To Run What](#when-to-run-what)
- [Main Commands](#main-commands)
- [Production Audit Flow](#production-audit-flow)
- [Artifacts](#artifacts)
- [Operating Rules](#operating-rules)

## Overview

This project separates model governance from general product testing.

The evaluation system has two lines:

- `expert`
  - offline release gate
  - curated benchmark datasets
  - used for release decisions
- `production_audit`
  - sampled from real interactions
  - used for AI judge and human review
  - feeds confirmed failures back into `expert`

`src/eval_platform/` owns evaluation execution.  
`tests/` remains the system correctness and regression suite.

## Lines And Profiles

Profiles are defined in [src/eval_assets/governance.yaml](/f:/workspace/nlp-crop-calendar/src/eval_assets/governance.yaml).

- `expert_blocking_gate`
  - blocking cases only
- `expert_regression_gate`
  - blocking + regression cases
- `production_audit_review`
  - audit-only review set

Gate meanings:

- `blocking`
  - any failure blocks release
- `regression`
  - broader comparison set
- `audit`
  - monitoring and review only

## When To Run What

- use `run`
  - when validating one model on one profile
  - typical after prompt, code, or dataset changes
- use `compare`
  - when deciding whether candidate model can replace baseline
- use `audit`
  - when reviewing real production quality
- use `promote`
  - when confirmed audit failures should enter `expert`

## Main Commands

Offline gate:

```bash
python -m src.eval_platform run --profile expert_blocking_gate
python -m src.eval_platform run --profile expert_regression_gate
```

Baseline vs candidate:

```bash
python -m src.eval_platform compare ...
```

Production audit one-shot:

```bash
python -m src.eval_platform audit run-latest --days 30 --out-dir .cache/eval/production_audit/latest
```

Audit split steps:

```bash
python -m src.eval_platform audit sample ...
python -m src.eval_platform audit judge ...
python -m src.eval_platform audit review-queue ...
python -m src.eval_platform audit export-csv ...
python -m src.eval_platform audit import-csv ...
python -m src.eval_platform audit promote ...
python -m src.eval_platform promote --promotion ...
```

## Production Audit Flow

Recommended loop:

```text
sample -> judge -> review-queue -> export/import CSV -> audit promote -> promote
```

Meaning of each step:

- `sample`
  - build audit batches from production interactions
- `judge`
  - run AI judge over sampled batches
- `review-queue`
  - keep only records needing human review
- `export-csv` / `import-csv`
  - expert-friendly review layer
- `audit promote`
  - export confirmed bad cases as promotion payloads
- `promote`
  - import reviewed cases back into `expert`

## Artifacts

Main local outputs:

- `.cache/eval/release_compare/`
  - compare results
- `.cache/eval/production_audit/batches/`
  - standalone `audit sample`
- `.cache/eval/production_audit/runs/`
  - one-shot `audit run-latest`
- `.cache/eval/production_audit/reviews/`
  - standalone `audit judge`
- `.cache/eval/production_audit/queues/`
  - standalone `audit review-queue`
- `.cache/eval/production_audit/promotions/`
  - standalone `audit promote`
- `.state/eval/production_audit/sampling_state.json`
  - sampling cursor

## Operating Rules

- candidate must pass `expert_blocking_gate`
- candidate must not regress on `expert_regression_gate`
- production audit should run on a regular cadence
- confirmed production failures should be reviewed before promotion
- promoted audit samples should not remain duplicated in `production_audit`

Recommended automation boundary:

- safe to automate:
  - `audit run-latest`
- keep human in the loop for:
  - review
  - promotion into `expert`
