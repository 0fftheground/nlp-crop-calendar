# Dialogue Orchestration

## Contents

- [Goal](#goal)
- [Core Model](#core-model)
- [Session Layer](#session-layer)
- [Thread Layer](#thread-layer)
- [Dialogue Act Layer](#dialogue-act-layer)
- [Field Update Layer](#field-update-layer)
- [Resolution Order](#resolution-order)
- [Current Engineering Mapping](#current-engineering-mapping)
- [Adapter Registry](#adapter-registry)
- [Next Steps](#next-steps)

## Goal

This project should not treat conversation handling as a single “follow-up vs new topic” classifier.

The better model is:

1. determine the active task thread
2. determine the user’s dialogue act inside that thread
3. apply structured field updates only after the thread and act are clear

This prevents common failures such as:

- carrying weather context into a crop-calendar request
- treating `早稻呢` as a region
- treating a confirmation as a normal free-form message

## Core Model

The orchestration model has four layers:

1. `session`
2. `thread`
3. `dialogue_act`
4. `field_update`

The system should resolve them in that order.

## Session Layer

`session` is the outer conversation container.

It stores:

- recent interactions
- active thread pointers
- pending follow-up state
- session context snapshots

`session_id` groups a conversation, but it must not be used as the only signal for continuity.

## Thread Layer

`thread` is the unit of business continuity.

One session may contain multiple threads:

- `weather_lookup`
- `sowing_suitability_lookup`
- `crop_calendar_workflow`
- `growth_stage_lookup`

Each interaction should carry:

- `thread_id`
- `parent_interaction_id`
- `task_type`
- `continuity_type`
- `continuity_source`

Interpretation:

- `thread_id` identifies the current business thread
- `parent_interaction_id` points to the immediate previous interaction in that thread
- `task_type` identifies the active tool or workflow
- `continuity_type` explains whether the turn is standalone, pending resume, or session-context resume

## Dialogue Act Layer

Once the thread is known, the system should classify the user turn as a dialogue act.

Recommended acts:

- `start_new_task`
- `update_fields`
- `select_option`
- `confirm`
- `cancel`
- `continue_task`
- `empty_input`

Examples:

- `早稻呢` -> `update_fields`
- `第2个` -> `select_option`
- `是` -> `confirm`
- `不用了` -> `cancel`
- `我想建立一个在湖南常德种植的湘早籼24号的移栽方案` -> `start_new_task`

## Field Update Layer

Only after thread and dialogue act are resolved should the system apply field updates.

Examples:

- `早稻呢` -> `culti_type=早稻`
- `直播呢` -> `planting_method=direct_seeding`
- `长沙呢` -> `region_id=长沙`
- `5月1日` -> `sowing_date=当年-05-01`

This layer should prefer structured field extraction and field override, not full prompt reinterpretation.

## Resolution Order

Recommended runtime order:

1. check `pending`
2. check session-context candidate
3. run standalone planner
4. resolve thread ownership
5. assign `dialogue_act`
6. apply structured field updates
7. execute tool or workflow

If uncertain, the system should prefer clarification over forced context carry-over.

## Current Engineering Mapping

The current codebase already has these pieces:

- `session_id`
- `thread_id`
- `parent_interaction_id`
- `continuity_type`
- `continuity_source`

This document adds a clearer orchestration contract:

- `task_type`
- `dialogue_act`

Current implementation status:

- `RequestRouter` now writes `dialogue_act` and `task_type` into the interaction context
- `interaction_store` persists those fields in request/response summaries, raw payloads, and top-level interaction columns
- pending resume uses a first-pass dialogue-act inference:
  - option choice -> `select_option`
  - follow-up field fill -> `update_fields`
  - save confirmation yes/no -> `confirm` / `cancel`
- standalone planner resolutions are labeled `start_new_task`
- session-context short-circuit resolutions are labeled `update_fields`
- clarification-style conflict handling is now available when a brief prompt could belong to the current thread or a new task
- audit source records now carry orchestration metadata such as `dialogue_act`, `task_type`, `continuity_type`, and `continuity_source`

This is not the final orchestration system, but it gives the project explicit metadata for future debugging, audit, and thread-aware routing.

## Adapter Registry

The project now uses a registry-based session-context adapter layer instead of a long `if/elif` chain.

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

Why this matters:

- adding a new tool or workflow no longer requires editing a central routing `if/elif` ladder
- context extraction and contextual resume logic are declared together
- per-tool field-update semantics can stay close to the tool domain while still using one shared orchestration contract

Recommended onboarding steps for a new tool or workflow:

1. add a `SessionContextAdapter` entry
2. declare its `task_type`
3. declare its `updatable_fields`
4. implement `extract_context(...)`
5. implement `build_candidate(...)`
6. add one session-context regression test

## Next Steps

Recently completed follow-ups:

1. more planting-style contextual follow-up merging now uses shared field-update helpers instead of tool-local heuristics only
2. adapter metadata (`task_type`, `updatable_fields`, `evidence`) is exposed to contextual-candidate logging and tracing
3. `dialogue_act` and `task_type` are now included in audit source payloads and AI-judge context
4. complete standalone prompts now bypass session-context short-circuit through explicit thread-switch rules
5. ambiguous thread ownership now creates a dedicated clarification pending state instead of forcing the request into the previous thread

Remaining follow-ups worth doing later:

1. move more non-planting tool-specific field updates into the same shared helper layer
2. surface adapter metadata directly in UI/debug panels, not only logs and audit payloads
