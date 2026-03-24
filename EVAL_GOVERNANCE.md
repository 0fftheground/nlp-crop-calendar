# Eval Governance

This repository uses two evaluation lines and three operational profiles.

## Evaluation Lines

- `expert`
  - Maintained by product/business owners.
  - Used for release gates and model promotion decisions.
  - Prefer deterministic graders over AI judge.
  - Includes both single-turn task datasets and deterministic session/follow-up datasets.
- `production_audit`
  - Sampled from deidentified real interactions.
  - Used for periodic AI-judge review and human spot checks.
  - Not a hard release gate by default.

## Boundary With `tests/`

`src/eval_platform` is not the general system test suite.

- `src/eval_platform`
  - Owns release gating, model comparison, and online-quality review.
  - Should stay small, explicit, and decision-oriented.
- `tests/`
  - Owns system correctness, workflow behavior, router state transitions, and regression protection.
  - Should remain broader than `eval`.

Some session behavior exists in `expert` eval on purpose, but only for the few continuity cases that can block a model rollout.

## Case Gates

- `blocking`
  - High-risk cases. Any failure blocks rollout or model switch.
- `regression`
  - Broader comparison set. Used after blocking passes.
- `audit`
  - Monitoring-only production samples for quality review.

## Profiles

Profiles are defined in [governance.yaml](/f:/workspace/nlp-crop-calendar/src/eval_assets/governance.yaml).

- `expert_blocking_gate`
  - Runs only `blocking` cases from the `expert` line.
  - Exit code is non-zero on any failure.
- `expert_regression_gate`
  - Runs `blocking` and `regression` cases from the `expert` line.
  - Blocks rollout if blocking failures exist or the aggregate pass rate falls below policy.
  - Also carries relative `latency` and `estimated token` regression budgets against the baseline model.
- `production_audit_review`
  - Runs `audit` cases from the `production_audit` line.
  - Exit code stays zero by design; results feed AI judge review and human spot checks.

## Operating Model

1. Candidate model must pass `expert_blocking_gate`.
2. Candidate model must not regress on `expert_regression_gate`.
3. Real interaction samples are reviewed on `production_audit_review`.
4. The audit loop runs `sample -> ai judge -> human spot check -> promote`.
5. New high-risk failures from audit are deidentified and promoted into the `expert` line.

## End-to-End Pipeline

This section is the practical "how the whole thing runs" view for teammates who need to operate or extend the evaluation process.

### Pipeline Overview

```text
                       ┌──────────────────────────────┐
                       │ 1. Expert datasets            │
                       │ src/eval_assets/expert/       │
                       └───────────────┬──────────────┘
                                       │
                                       v
                       ┌──────────────────────────────┐
                       │ 2. Offline run / compare      │
                       │ src.eval_platform run         │
                       │ src.eval_platform compare     │
                       └───────────────┬──────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    v                                     v
      ┌──────────────────────────────┐     ┌──────────────────────────────┐
      │ expert_blocking_gate         │     │ expert_regression_gate       │
      │ hard rollout blocker         │     │ broader baseline comparison  │
      └───────────────┬──────────────┘     └───────────────┬──────────────┘
                      │                                    │
                      └──────────────────┬─────────────────┘
                                         │
                                         v
                       ┌──────────────────────────────┐
                       │ 3. Release decision           │
                       │ candidate can / cannot ship   │
                       └───────────────┬──────────────┘
                                       │
                                       v
                ┌─────────────────────────────────────────────┐
                │ 4. Online production audit                  │
                │ sample -> judge -> review-queue -> promote  │
                └──────────────────────┬──────────────────────┘
                                       │
                                       v
                ┌─────────────────────────────────────────────┐
                │ 5. Confirmed bad samples re-enter expert    │
                │ src.eval_platform promote                   │
                └──────────────────────┬──────────────────────┘
                                       │
                                       v
                       ┌──────────────────────────────┐
                       │ 6. Next model comparison      │
                       │ uses the expanded expert set  │
                       └──────────────────────────────┘
```

### What Lives Where

- Eval logic: `src/eval_platform/`
- Eval assets: `src/eval_assets/`
- Offline release datasets: `src/eval_assets/expert/`
- Online audit datasets/templates: `src/eval_assets/production_audit/`
- Governance config: `src/eval_assets/governance.yaml`
- Local run artifacts: `.cache/eval/...`
- Production-audit sampling cursor: `.state/eval/production_audit/sampling_state.json`
- AI judge model config:
  - `AUDIT_JUDGE_MODEL`
  - if not set, audit judge falls back to `LLM_MODEL`

### Cache Artifact Layout

The default cache layout is intentionally separated by pipeline step so operators can tell where a file came from without inspecting the file contents.

- `.cache/eval/release_compare/`
  - generated by `compare`
  - contains baseline-vs-candidate JSON outputs
- `.cache/eval/production_audit/batches/`
  - generated by `audit sample`
  - contains raw sampled audit batches
- `.cache/eval/production_audit/runs/`
  - generated by `audit run-latest`
  - each timestamped run contains:
    - batch files at the run root
    - `reviews/`
    - `queues/`
- `.cache/eval/production_audit/reviews/`
  - generated by standalone `audit judge`
- `.cache/eval/production_audit/queues/`
  - generated by standalone `audit review-queue`
- `.cache/eval/production_audit/promotions/`
  - generated by standalone `audit promote`

If you still see flat folders such as `audit_batch_run` or `audit_review_run` directly under `.cache/eval/`, those are legacy manual outputs from earlier iterations. They are no longer the preferred default layout.

### Who Uses Which Path

- Product / business / LLM owner
  - Maintains `expert` datasets
  - Decides `blocking` vs `regression`
- Engineer changing model choice or prompt behavior
  - Runs `run` and `compare`
  - Uses results for rollout decisions
- Engineer / analyst on online quality review
  - Runs `audit`
  - Performs human spot checks
  - Promotes confirmed failures back into `expert`

## Operator Guide

This section is the quick-start operating manual for teammates. If someone only wants to know "what should I run, when, and why", start here.

### Command Selection Guide

- Use `run`
  - When you want to evaluate one model on one governance profile.
  - Typical use:
    - verify current default model still passes
    - check a prompt change against the existing gate
    - rerun the gate after adding expert cases
- Use `compare`
  - When you want to compare a candidate model against a baseline model before rollout.
  - Typical use:
    - decide whether a new `llm-model` can replace the current one
    - decide whether a new `extractor-model` can replace the current one
- Use `audit`
  - When you want to review real production traffic quality.
  - Typical use:
    - periodic online health review
    - AI-judge review of sampled real interactions
    - building a human spot-check queue
- Use `promote`
  - When you have already confirmed bad production samples and want them to become part of the offline expert gate.
  - Typical use:
    - import reviewed failures back into `expert`
    - rerun release gates after importing them

### When To Run What

- Before changing model configuration
  - Run `compare`
  - Reason: this is the actual baseline-vs-candidate decision step
- After changing prompt logic but before release
  - Run `run --profile expert_blocking_gate`
  - Then run `run --profile expert_regression_gate`
  - Reason: prompt changes may affect release behavior even if model stays the same
- After adding or editing expert datasets
  - Run `run`
  - Reason: the gate itself changed; rerun it immediately
- On a regular production-review cadence
  - Run `audit run-latest`
  - Then review the generated review and queue files
  - Reason: this is the online monitoring path
- After human reviewers confirm real failures
  - Run `audit promote`
  - Then run `promote`
  - Reason: confirmed production failures should feed the next release gate

### Common End-to-End Scenarios

#### Scenario 1: Evaluate a prompt or code change against the current model

Use this when the model stays the same, but router/extractor/workflow behavior may have changed.

1. Update code, prompt, or dataset.
2. Run:
   ```bash
   python -m src.eval_platform run --profile expert_blocking_gate
   python -m src.eval_platform run --profile expert_regression_gate
   ```
3. Read the terminal summary or JSON output if requested.
4. If either profile fails, do not promote the change.

#### Scenario 2: Compare a candidate model before switching production

Use this when you want to test a new baseline replacement.

1. Choose the baseline and candidate values.
2. Run:
   ```bash
   python -m src.eval_platform compare \
     --baseline-llm-model gpt-4.1-mini \
     --baseline-extractor-model gpt-4.1-mini \
     --candidate-llm-model gpt-5-mini \
     --candidate-extractor-model gpt-5-mini
   ```
3. Open the generated JSON file.
4. Review:
   - whether `passed=true`
   - which tasks were impacted
   - which regressions, if any, were found
5. Only switch the configured model if compare passes.

#### Scenario 3: Weekly or daily online audit review

Use this when you want to inspect real traffic quality over time.

1. Run:
   ```bash
   python -m src.eval_platform audit run-latest --limit 50 --days 30 --out-dir .cache/eval/production_audit/latest
   ```
2. Inspect:
   - `reviews/*.review.yaml`
   - `queues/*.review.queue.yaml`
3. Human reviewers update the review files.
4. Confirmed issues are marked `human_review.status: promote_to_expert`.
5. Export promotion files:
   ```bash
   python -m src.eval_platform audit promote --review .cache/eval/production_audit/latest/reviews/planner.review.yaml --review .cache/eval/production_audit/latest/reviews/extractor.review.yaml --review .cache/eval/production_audit/latest/reviews/variety_match.review.yaml --out-dir .cache/eval/promotion_candidates
   ```
6. Import them into expert:
   ```bash
   python -m src.eval_platform promote --promotion .cache/eval/promotion_candidates/planner.review.planner.promotion.yaml --rerun-profile expert_blocking_gate --rerun-profile expert_regression_gate
   ```

### Inputs And Outputs By Command

- `run`
  - Input:
    - governance profile
    - expert or production-audit dataset list
    - optional model overrides
  - Output:
    - terminal summary
    - optional JSON report
    - exit code driven by governance policy
- `compare`
  - Input:
    - baseline models
    - candidate models
    - expert profiles
  - Output:
    - concise terminal summary
    - JSON result at `.cache/eval/release_compare/latest.json` by default
    - exit code showing pass/fail
- `audit sample` / `audit run-latest`
  - Input:
    - real interactions
    - sampling cursor
  - Output:
    - YAML batch files under the chosen output directory
- `audit judge`
  - Input:
    - sampled batch files
  - Output:
    - `*.review.yaml`
- `audit review-queue`
  - Input:
    - review files
  - Output:
    - `*.queue.yaml`
- `audit promote`
  - Input:
    - human-reviewed review files
  - Output:
    - task-scoped promotion YAML files
- `promote`
  - Input:
    - promotion YAML files
  - Output:
    - updated expert datasets
    - cleaned production-audit datasets by default
    - optional rerun results

### Ownership Suggestions

- Model owner / LLM engineer
  - Runs `compare`
  - Interprets release decision JSON
- Feature engineer
  - Runs `run` after router/prompt/workflow changes
- Operations / analyst / QA
  - Runs `audit run-latest`
  - Performs human spot checks
- Product or business reviewer
  - Decides whether confirmed samples should become `blocking` or `regression`
- Repo maintainer
  - Runs `promote` and reruns expert gates after promotion

## Execution Flow

### A. Offline Expert Evaluation

This is the release-gating path. It answers: "Is the current model acceptable on the curated gold set?"

#### Inputs

- Governance profile from `src/eval_assets/governance.yaml`
- One or more expert datasets under `src/eval_assets/expert/`
- Optional model overrides via CLI

#### Command

```bash
python -m src.eval_platform run --profile expert_blocking_gate
python -m src.eval_platform run --profile expert_regression_gate
```

#### What Happens Internally

1. The CLI loads the requested profile from `src/eval_assets/governance.yaml`.
2. That profile resolves a list of dataset files and which gates to include.
3. Each case in each dataset is sent to the registered runner for its task.
4. The runner returns:
   - `actual`
   - lightweight metrics such as latency and estimated tokens
5. A deterministic grader compares `actual` with `expected`.
6. A dataset summary is produced.
7. A profile summary is produced from all dataset summaries.
8. Exit code is decided from policy.

#### Outputs

- Terminal summary
- Optional JSON if `--json-out` is passed
- Non-zero exit code for failing release-gate profiles

#### When To Use

- Before changing default model configuration
- Before releasing prompt changes that affect router/extractor behavior
- After importing promoted failures from production audit

### B. Baseline vs Candidate Comparison

This is the release-decision path. It answers: "Is the candidate model safe to replace the current baseline?"

#### Inputs

- Baseline and candidate model values
- Expert blocking and regression profiles
- Current governance thresholds

#### Command

```bash
python -m src.eval_platform compare \
  --baseline-llm-model gpt-4.1-mini \
  --baseline-extractor-model gpt-4.1-mini \
  --candidate-llm-model gpt-5-mini \
  --candidate-extractor-model gpt-5-mini
```

#### What Happens Internally

1. Compare resolves the effective baseline and candidate models.
2. It validates that provided model names exist and are accessible.
3. It detects which model dimension actually changed.
4. It filters the expert regression set down to the impacted tasks only.
5. It runs:
   - candidate blocking gate
   - baseline regression gate
   - candidate regression gate
6. It compares the baseline and candidate summaries on:
   - `pass_rate`
   - `avg_score`
   - `blocking_failed`
   - `avg_latency_ms`
   - `estimated_total_tokens`
7. It writes a JSON result file.
8. It prints only a short summary to the terminal.

#### Outputs

- Default JSON: `.cache/eval/release_compare/latest.json`
- Optional custom JSON path via `--json-out`
- Terminal summary showing:
  - pass/fail
  - impacted tasks
  - number of regressions
  - JSON result location

#### Important Operator Notes

- If only `llm-model` changed, compare only runs `planner` and `variety_match`.
- If only `extractor-model` changed, compare only runs `extractor` and `workflow_extract`.
- Deterministic continuity tasks such as `session_context` and `followup_resume` are not part of model-vs-model compare.
- If baseline and candidate resolve to the same effective models, compare exits early.

### C. Production Audit Sampling

This is the online-quality monitoring path. It answers: "What is happening on real traffic, and what new bad cases should be reviewed?"

#### Inputs

- Real rows from the configured `interactions` store
- Deidentification rules
- Current sampling cursor in `.state/eval/production_audit/sampling_state.json`

#### Command

```bash
python -m src.eval_platform audit sample --limit 50 --days 30 --out-dir .cache/eval/audit_batch
```

Or, for one-step execution:

```bash
python -m src.eval_platform audit run-latest --limit 50 --days 30 --out-dir .cache/eval/production_audit/latest
```

#### What Happens Internally

1. The sampler reads recent interactions from the configured store.
2. It continues from the last `(created_at, id)` watermark by default.
3. It deidentifies sensitive values.
4. It reconstructs task-shaped audit cases when possible.
5. It splits samples into:
   - standalone replayable cases
   - `context_dependent` judge-only cases
6. It writes YAML batch files to the requested output directory.

#### Outputs

- `planner.yaml`
- `extractor.yaml`
- `variety_match.yaml`
- optionally:
  - `planner.context_dependent.yaml`
  - `extractor.context_dependent.yaml`
  - `variety_match.context_dependent.yaml`

#### Important Operator Notes

- `context_dependent` files are not deterministic replay datasets.
- They exist so AI judge and human review can inspect short follow-up turns with a bounded `context_window`.
- Sampling is incremental by default; use `--reset-cursor` only when you intentionally want to rebuild from the current time window.

### D. AI Judge Review

This is the first screening pass on sampled production data.

#### Command

```bash
python -m src.eval_platform audit judge --batch .cache/eval/audit_batch/planner.yaml --batch .cache/eval/audit_batch/extractor.yaml --batch .cache/eval/audit_batch/variety_match.yaml --out-dir .cache/eval/audit_review
```

#### What Happens Internally

1. Each sampled record is paired with its observed output and expected structure.
2. AI judge reviews the record.
3. Review payloads are written as `*.review.yaml`.

#### Outputs

- `planner.review.yaml`
- `extractor.review.yaml`
- `variety_match.review.yaml`

These files are the source of truth for downstream human review and later promotion.

### E. Human Review Queue

This is the triage step. It narrows the review set to samples that are not safe to auto-pass.

#### Command

```bash
python -m src.eval_platform audit review-queue --review .cache/eval/audit_review/planner.review.yaml --review .cache/eval/audit_review/extractor.review.yaml --review .cache/eval/audit_review/variety_match.review.yaml --out-dir .cache/eval/audit_queue
```

#### What Happens Internally

1. Review records are scanned for:
   - AI-judge failures
   - low-confidence AI-judge passes
   - explicit `needs_human_review`
2. Only those records are copied into a queue file.

#### Outputs

- `planner.review.queue.yaml`
- `extractor.review.queue.yaml`
- `variety_match.review.queue.yaml`

#### Human Reviewer Responsibility

- Confirm whether the sample is actually wrong
- Correct `input` or `expected` if needed
- Set `human_review.status: promote_to_expert` when the case should become part of the offline gold set

### F. CSV / Excel Review Layer

This is the human-friendly review layer. YAML remains the system source of truth, but human reviewers do not need to edit YAML directly.

#### Recommended Flow

1. Build queue YAML files.
2. Export them to CSV.
3. Open the CSV in Excel or another spreadsheet tool.
4. Fill review columns such as:
   - `human_status`
   - `reviewer`
   - `target_gate`
   - `corrected_input_json`
   - `corrected_expected_json`
   - `notes`
5. Import the CSV back into the original review YAML.
6. Let `audit promote` export only the cases that were marked `promote_to_expert`.

#### Export Command

Export queue files for spreadsheet review:

```bash
python -m src.eval_platform audit export-csv --queue .cache/eval/production_audit/queues/planner.review.queue.yaml --out-dir .cache/eval/production_audit/csv
```

You can also export full review files:

```bash
python -m src.eval_platform audit export-csv --review .cache/eval/production_audit/reviews/planner.review.yaml --out-dir .cache/eval/production_audit/csv
```

#### Import Command

After the CSV is edited and saved:

```bash
python -m src.eval_platform audit import-csv --csv .cache/eval/production_audit/csv/planner.review.queue.csv
```

#### What Import Does

- Updates the original `review.yaml`
- Refreshes the adjacent `queue.yaml` if it exists
- Removes items from the queue once their `human_review.status` is no longer `pending`

#### Notes For Reviewers

- `corrected_input_json` and `corrected_expected_json` should stay valid JSON objects
- CSV files are written with UTF-8 BOM so Excel opens Chinese text more reliably
- `source_review_file` is included to allow safe round-trip import back into the correct review file

### G. Audit Promotion Export

This turns confirmed bad online samples into promotion payloads.

#### Command

```bash
python -m src.eval_platform audit promote --review .cache/eval/audit_review/planner.review.yaml --review .cache/eval/audit_review/extractor.review.yaml --review .cache/eval/audit_review/variety_match.review.yaml --out-dir .cache/eval/promotion_candidates
```

#### What Happens Internally

1. Review files are scanned.
2. Only records with `human_review.status: promote_to_expert` are kept.
3. Promotion payloads are grouped by task.
4. New YAML files are written to the promotion output directory.

#### Outputs

- task-specific promotion files such as:
  - `planner.review.planner.promotion.yaml`
  - `extractor.review.extractor.promotion.yaml`
  - `variety_match.review.variety_match.promotion.yaml`
- the source `review.yaml` is updated in place with promotion metadata such as `promotion_exported_at`
- if an adjacent `queue.yaml` exists, it is rebuilt so already-processed records drop out of the queue

### H. Promotion Import Back Into Expert

This closes the loop. It makes reviewed production failures part of the next offline release gate.

#### Command

```bash
python -m src.eval_platform promote --promotion .cache/eval/promotion_candidates/planner.review.planner.promotion.yaml --rerun-profile expert_blocking_gate --rerun-profile expert_regression_gate
```

#### What Happens Internally

1. Promotion payloads are loaded.
2. Their cases are upserted into `src/eval_assets/expert/<task>.yaml`.
3. Matching cases are removed from `src/eval_assets/production_audit/<task>.yaml` by default.
4. Optional rerun profiles are executed.
5. The rerun result becomes the updated release baseline.

#### Matching Rule For Production-Audit Cleanup

- Primary key: `source.interaction_id`
- Fallback: `id`
- Final fallback: normalized `input + expected`

#### Why This Cleanup Exists

- To avoid the same sample living in both the expert and production-audit lines
- To keep production audit focused on "still uncurated" real-traffic signals
- To keep expert focused on curated, reusable release-gate cases

## Scheduled Audit Feasibility

Yes. This pipeline can support a scheduled task for ongoing production-state review.

### Recommended Automation Boundary

- Safe to automate:
  - `python -m src.eval_platform audit run-latest ...`
  - storing the generated review and queue files
  - notifying reviewers that a new batch is ready
- Usually keep manual:
  - editing `human_review`
  - deciding whether a case should be promoted
  - importing promoted cases into `expert`

This split is safer because online audit is partly judgment-driven, while release-gate promotion changes your curated gold set.

### Recommended Scheduled Flow

1. A scheduled job runs `audit run-latest`.
2. The job writes results to a timestamped folder under `.cache/eval/production_audit/`.
   - default scheduled location: `.cache/eval/production_audit/scheduled/<timestamp>/`
3. The job optionally copies or uploads the generated review files somewhere visible to reviewers.
4. Human reviewers inspect and update the review files.
5. A human or controlled follow-up job runs:
   - `audit promote`
   - `promote`

### Good Trigger Frequencies

- Daily
  - if traffic is active and model behavior changes often
- Weekly
  - if traffic volume is moderate and manual review bandwidth is limited
- After major model or prompt changes
  - even if you already run a daily or weekly schedule

### Important Note

Automated audit is a good next step and fits the current design well.
What should not be fully automated by default is the final promotion into `expert`, because that changes the curated release-gate dataset and should stay reviewable.

### Provided Windows Scripts

- Run one scheduled-style audit cycle manually:
  - `scripts/run_production_audit_cycle.ps1`
  - default behavior:
    - uses `conda run -n llm-agent`
    - writes into `.cache/eval/production_audit/scheduled/<timestamp>/`
    - continues from `.state/eval/production_audit/sampling_state.json`
- Register a Windows Scheduled Task:
  - `scripts/register_production_audit_task.ps1`
  - this creates a Task Scheduler entry that executes `run_production_audit_cycle.ps1`

Example registration:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_production_audit_task.ps1 -TaskName NlpCropCalendarProductionAudit -Frequency Daily -At 09:00
```

Example weekly registration:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/register_production_audit_task.ps1 -TaskName NlpCropCalendarProductionAuditWeekly -Frequency Weekly -At 09:00 -DaysOfWeek Monday,Thursday
```

## Daily Operating Recipes

### Change the default model safely

1. Run `compare` between baseline and candidate.
2. Review the JSON result.
3. If passed, update the model config.
4. Run `expert_blocking_gate` once more after the config change if needed.

### Review online quality every cycle

1. Run `audit run-latest`.
2. Inspect review files and queue files.
3. Mark confirmed bad cases for promotion.
4. Export promotion payloads.
5. Import them into `expert`.
6. Rerun expert gates.

### Add a new release-gate case manually

1. Decide whether it belongs to `blocking` or `regression`.
2. Put it in the correct task dataset under `src/eval_assets/expert/`.
3. Add enough human-readable context fields such as `description`, `prior_prompt`, or `prior_pending_message`.
4. Run the relevant expert profile.

The eval runner now records lightweight execution metrics for each case and dataset:

- `avg_latency_ms`
- `p95_latency_ms`
- `estimated_input_tokens`
- `estimated_output_tokens`
- `estimated_total_tokens`

These token values are tokenizer-based estimates from the active model client. They are intended for model comparison and budget control, not exact billing reconciliation.

Expert release gates include:

- single-turn tasks: `planner`, `extractor`, `variety_match`
- workflow extraction task: `workflow_extract`
- session continuity tasks: `session_context`, `followup_resume`

## Commands

Run the blocking gate:

```bash
python -m src.eval_platform run --profile expert_blocking_gate
```

Run the broader expert regression set:

```bash
python -m src.eval_platform run --profile expert_regression_gate
```

Run production audit samples:

```bash
python -m src.eval_platform run --profile production_audit_review
```

Override candidate models during comparison:

```bash
python -m src.eval_platform run --profile expert_regression_gate --llm-model gpt-5-mini --extractor-model gpt-5-mini
```

Run baseline-vs-candidate release comparison:

```bash
python -m src.eval_platform compare --baseline-llm-model gpt-4.1-mini --baseline-extractor-model gpt-4.1-mini --candidate-llm-model gpt-5-mini --candidate-extractor-model gpt-5-mini --json-out .cache/eval/release-compare.json
```

If `--json-out` is omitted, compare writes to `.cache/eval/release_compare/latest.json`.
The terminal only prints a concise summary; the full comparison payload is stored in that JSON file.

`compare` automatically filters the regression set to only the tasks affected by the changed model dimension:

- `llm-model` changes: `planner`, `variety_match`
- `extractor-model` changes: `extractor`, `workflow_extract`
- deterministic continuity tasks are skipped for compare because they are not model-sensitive

If baseline and candidate resolve to the same effective models on both dimensions, `compare` exits without running dataset comparisons.

To compare only one model dimension, keep the other dimension explicitly identical on both sides.

LLM-only comparison:

```bash
python -m src.eval_platform compare --baseline-llm-model gpt-4.1-mini --candidate-llm-model gpt-5-mini --baseline-extractor-model gpt-4.1-mini --candidate-extractor-model gpt-4.1-mini
```

Extractor-only comparison:

```bash
python -m src.eval_platform compare --baseline-llm-model gpt-4.1-mini --candidate-llm-model gpt-4.1-mini --baseline-extractor-model gpt-4.1-mini --candidate-extractor-model gpt-5-mini
```

If one of these arguments is omitted, that side falls back to the current environment configuration, which makes the comparison less controlled.

If a provided model name does not exist or is not accessible on the configured OpenAI-compatible endpoint, compare stops during preflight validation before running any datasets.

`release_compare` now checks three things on the regression profile:

- quality regression: `pass_rate`, `avg_score`, `blocking_failed`
- latency regression: `avg_latency_ms`
- token regression: `estimated_total_tokens`

The default policy in [governance.yaml](/f:/workspace/nlp-crop-calendar/src/eval_assets/governance.yaml) uses:

- `max_latency_regression_ratio: 1.5`
- `max_total_tokens_regression_ratio: 1.5`

## Production Audit Closed Loop

Sample deidentified interactions from the configured interaction store:

```bash
python -m src.eval_platform audit sample --limit 50 --days 30 --out-dir .cache/eval/production_audit/batches/manual
```

Sampling is now incremental by default:

- the sampler persists a watermark file at `.state/eval/production_audit/sampling_state.json`
- each run continues from the last sampled `(created_at, id)` pair
- use `--reset-cursor` if you want to rebuild from the current date window

Run the latest production-audit batch end to end:

```bash
python -m src.eval_platform audit run-latest --limit 50 --days 30 --out-dir .cache/eval/production_audit/runs/latest
```

This command now emits two kinds of batch files:

- `planner.yaml` / `extractor.yaml` / `variety_match.yaml`
  - standalone, deterministic replay is enabled
- `planner.context_dependent.yaml` / `extractor.context_dependent.yaml` / `variety_match.context_dependent.yaml`
  - short follow-up samples that depend on session context
  - AI judge only; deterministic replay is intentionally skipped
  - each record carries a deidentified bounded `context_window` from the same session instead of just the immediately previous turn

Run AI judge on sampled audit batches:

```bash
python -m src.eval_platform audit judge --batch .cache/eval/production_audit/batches/manual/planner.yaml --batch .cache/eval/production_audit/batches/manual/extractor.yaml --batch .cache/eval/production_audit/batches/manual/variety_match.yaml --out-dir .cache/eval/production_audit/reviews
```

Generate a human review queue from judge output:

```bash
python -m src.eval_platform audit review-queue --review .cache/eval/production_audit/reviews/planner.review.yaml --review .cache/eval/production_audit/reviews/extractor.review.yaml --review .cache/eval/production_audit/reviews/variety_match.review.yaml --out-dir .cache/eval/production_audit/queues
```

After humans edit review files and mark `human_review.status: promote_to_expert`, export promotion candidates:

```bash
python -m src.eval_platform audit promote --review .cache/eval/production_audit/reviews/planner.review.yaml --review .cache/eval/production_audit/reviews/extractor.review.yaml --review .cache/eval/production_audit/reviews/variety_match.review.yaml --out-dir .cache/eval/production_audit/promotions
```

Import promotion candidates back into the expert datasets and rerun release gates:

```bash
python -m src.eval_platform promote --promotion .cache/eval/production_audit/promotions/planner.review.planner.promotion.yaml --rerun-profile expert_blocking_gate --rerun-profile expert_regression_gate
```

By default, `promote` prunes matching cases from `src/eval_assets/production_audit/` after import, using `source.interaction_id` as the primary match key and falling back to `id` or normalized `input + expected`. Pass `--keep-production-audit` to skip that cleanup.

The canonical module layout is `src/eval_platform` for logic and `src/eval_assets` for assets.
Offline release datasets live under `src/eval_assets/expert/`; production-audit assets live under `src/eval_assets/production_audit/`.
