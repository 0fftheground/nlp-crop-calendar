# Interfaces

## Contents

- [Weather Service](#weather-service)
- [Crop Calendar Service](#crop-calendar-service)
- [Platform Service](#platform-service)

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
