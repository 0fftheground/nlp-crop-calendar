# Interfaces

## Contents

- [Weather Service](#weather-service)
- [Crop Calendar Service](#crop-calendar-service)
- [Platform Service](#platform-service)
- [Task Creation Endpoint](#task-creation-endpoint)

This document lists the main external services that this project integrates with at runtime.

## Weather Service

Base URL:

- `http://10.109.2.203:15637`

Main endpoints:

- `POST /suit_rili`
  - daily weather + agronomy suitability
- `POST /bozhong_syd`
  - sowing suitability

Typical inputs:

- `farm_id` or `region_id`
- date range
- cultivation / sowing metadata for sowing recommendation

## Crop Calendar Service

Base URL:

- `http://10.109.2.203:26405`

Main endpoints:

- `POST /cropCalender/previewCalender`
  - preview crop calendar plan
- `POST /cropCalender/plantPlan/activate`
  - activate / deactivate plan
- `POST /cropCalender/refreshCalenderByPlantPlan`
  - refresh by existing plan
- `POST /cropCalender/plantPlan/delete`
  - delete plan

Important note:

- region-mode preview does not create a plan
- farm-mode preview may return `plant_season_id`

## Platform Service

Base URL:

- `http://10.109.2.203:35168`

Main endpoints used by this project:

- `POST /planting-plan/search`
- `GET /planting-plan/active`
- `GET /planting-plan/{plan_id}`
- `GET /planting-plan/detail/{plan_id}`
- `GET /growth-stage/by-plan/{plan_id}`
- `GET /farm-work/recent-week/{farm_id}`

These endpoints support:

- active plan listing
- plan lookup and detail query
- growth-stage result lookup
- recent farm work summary

## Task Creation Endpoint

Endpoint:

- `POST /api/tasks/{plan_id}`

Purpose:

- unified task creation entry
- recommended tasks are written to the task record table only after completion
- other tasks are written to either the task record table or the extra task table based on `is_completed`
- `播种` and `移栽` are special cases: they update the planting plan date directly and trigger recalculation

Common request fields:

- `name`
  - task name
- `date`
  - task date
- `is_completed`
  - service-side derived field
  - `date <= today` -> `true`
  - `date > today` -> `false`
- `task_type`
  - service-side derived field
  - if `name` exactly matches `public.agri_code_dict.code_name` where `category=farmworks`, both `name` and `task_type` should use that matched value
  - if `name` is a custom value not found in `farmworks`, keep the user-provided `name` and set `task_type` to `其他`
- `detail` / `work_desc`
  - optional for completed tasks
  - supports JSON text or object
  - only contains `operator` and `work_desc`
  - `operator` defaults to the logged-in user when not explicitly provided

Recommended `detail` / `work_desc` structure:

```json
{
  "operator": "张三",
  "work_desc": "施肥N单质肥，亩施10 kg；无人机撒施叶面肥。"
}
```

Request example 1: completed recommended task

```json
{
  "name": "封闭除草",
  "date": "2026-03-19",
  "is_completed": true,
  "detail": "{\"operator\":\"张三\",\"work_desc\":\"完成封闭除草处理。\"}"
}
```

Response:

```json
{
  "status": "success",
  "target": "record"
}
```

Request example 2: incomplete extra task

```json
{
  "name": "追肥",
  "date": "2026-03-26",
  "is_completed": false
}
```

Response:

```json
{
  "status": "success",
  "target": "extra"
}
```

Request example 3: completed extra task

```json
{
  "name": "无人机补施叶面肥",
  "date": "2026-03-20",
  "is_completed": true,
  "detail": {
    "operator": "李四",
    "work_desc": "无人机喷施叶面肥，亩用量 80 ml。"
  }
}
```

Response:

```json
{
  "status": "success",
  "target": "record"
}
```

Request example 4: sowing

```json
{
  "name": "播种",
  "date": "2026-04-01",
  "is_completed": true,
  "detail": "{\"operator\":\"张三\",\"work_desc\":\"完成直播播种。\"}"
}
```

Response:

```json
{
  "status": "updated_plan",
  "field": "sowing_date"
}
```

Request example 5: transplanting

```json
{
  "name": "移栽",
  "date": "2026-04-18",
  "is_completed": true,
  "detail": "{\"operator\":\"张三\",\"work_desc\":\"完成机插移栽。\"}"
}
```

Response:

```json
{
  "status": "updated_plan",
  "field": "transp_date"
}
```

Validation rules:

- `is_completed` is derived from the task date instead of trusting user input
- recommended task + future date (`is_completed=false`) is not allowed and should return `400`
- other tasks derive `task_type` from `farmworks` dictionary matching; unmatched custom names should use `task_type=其他`
- completed tasks may omit `detail` / `work_desc`
- `播种` and `移栽` only support completed entries
- `移栽` is only allowed for non-direct-seeding plans
- duplicate entries under the same plan are not allowed for the same task on the same day
