# Dialogue Orchestration

## Contents

- [Overview](#overview)
- [Core Layers](#core-layers)
- [Runtime Flow](#runtime-flow)
- [Current Implementation](#current-implementation)
- [Adapter Registry](#adapter-registry)

## Overview

The dialogue manager decides whether a user turn continues an existing task or starts a new one.

It does this by separating conversation handling into session state, task threads, dialogue acts, and structured field updates.

The goal is to keep multi-turn behavior stable as tools and workflows grow over time.

This helps avoid:

- carrying the wrong context into a new task
- misclassifying confirmations or option selections as free-form input
- overwriting structured task state before thread ownership is clear
- relying on `session_id` alone to infer continuity

## Core Layers

The orchestration model has four layers:

1. `session`
2. `thread`
3. `dialogue_act`
4. `field_update`

### Session

`session` is the outer conversation container. It stores:

- pending follow-up state
- session context snapshots
- recent interaction lineage

`session_id` groups a conversation, but it is not the only continuity signal.

Current pending kinds:

- `field_fill`
  - waiting for missing structured fields
- `option_select`
  - waiting for a candidate or numbered choice
- `confirmation`
  - waiting for explicit confirm or cancel
- `clarification`
  - waiting for thread-ownership disambiguation

### Thread

`thread` is the unit of task continuity. One session may contain multiple independent task threads.

Each interaction carries:

- `thread_id`
- `parent_interaction_id`
- `task_type`
- `continuity_type`
- `continuity_source`

Field meanings:

- `thread_id`
  - stable identifier for the current task thread
  - new standalone requests normally start a new `thread_id`
- `parent_interaction_id`
  - direct parent interaction inside the current thread
  - `null` means the current turn starts a new branch
- `task_type`
  - semantic task label for the active thread
  - this is a string identifier, not a strict closed enum
  - current common values include:
    - `none`
    - `weather_lookup`
    - `variety_lookup`
    - `sowing_suitability_lookup`
    - `plant_plan_list_active`
    - `plant_plan_delete`
    - `growth_stage_lookup`
    - `crop_calendar_workflow`
    - `clarification`
  - adapter-normalized labels may also appear in contextual-resume paths:
    - `weather`
    - `variety`
    - `sowing`
    - `plan_list`
    - `plan_delete`
    - `growth_stage`
    - `crop_calendar`
- `continuity_type`
  - how the current turn was attached to the current thread
  - allowed values currently used by the runtime:
    - `standalone`
    - `pending_resume`
    - `session_context_resume`
- `continuity_source`
  - where the continuity signal came from
  - allowed values currently used by the runtime:
    - `none`
    - `pending`
    - `session_context`

### Dialogue Act

Once the thread is known, the system classifies the current turn as a dialogue act.

Supported acts:

- `start_new_task`
- `update_fields`
- `select_option`
- `confirm`
- `cancel`
- `continue_task`
- `empty_input`

Field meaning:

- `dialogue_act`
  - the user's action inside the current thread
  - allowed values currently used by the runtime:
    - `start_new_task`
    - `update_fields`
    - `select_option`
    - `confirm`
    - `cancel`
    - `continue_task`
    - `empty_input`

### Field Update

Only after thread and dialogue act are resolved should the system apply field updates.

This layer should prefer structured field extraction and field override, not full prompt reinterpretation.

## Runtime Flow

Recommended runtime flow:

1. check `pending`
2. try `session_context`
3. run standalone intent routing
4. resolve thread ownership
5. assign `dialogue_act`
6. apply field updates
7. execute tool or workflow

If thread ownership is ambiguous, the system should prefer clarification over forced context carry-over.

## Current Implementation

The current codebase already persists these orchestration fields:

- `thread_id`
- `parent_interaction_id`
- `task_type`
- `continuity_type`
- `continuity_source`
- `dialogue_act`

Current behavior:

- `RequestRouter` writes `dialogue_act` and `task_type` into the interaction context
- `interaction_store` persists orchestration fields in top-level interaction columns and raw payloads
- `pending` is typed:
  - `field_fill`
  - `option_select`
  - `confirmation`
  - `clarification`
- confirmation-style pending only resumes on explicit confirmation replies
- clarification-style pending can resume on replies such as:
  - `继续当前任务`
  - `开启新任务`
  - `1`
  - `2`
- audit records include orchestration metadata for bad-case analysis

## Adapter Registry

Session-context restoration uses a registry-based adapter layer instead of a central `if/elif` chain.

Each adapter declares:

- `kind`
- `name`
- `task_type`
- `updatable_fields`
- `extract_context(...)`
- `build_candidate(...)`

Current adapters cover:

- `weather_lookup`
- `variety_lookup`
- `sowing_suitability_lookup`
- `plant_plan_list_active`
- `plant_plan_delete`
- `growth_stage_lookup`
- `crop_calendar_workflow`

To add a new tool or workflow:

1. add a `SessionContextAdapter`
2. declare its `task_type`
3. declare its `updatable_fields`
4. implement `extract_context(...)`
5. implement `build_candidate(...)`
6. add at least one session-context regression test
