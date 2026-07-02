from __future__ import annotations

from dataclasses import dataclass

from .field_updates import extract_field_overrides


@dataclass(frozen=True)
class ExtractDecision:
    should_extract: bool
    reason: str
    extracted_fields: tuple[str, ...] = ()


def should_extract_for_route(*, action: str, name: str, prompt: str) -> ExtractDecision:
    route_action = str(action or "").strip()
    route_name = str(name or "").strip()
    text = str(prompt or "").strip()
    if not text:
        return ExtractDecision(False, "empty_prompt")
    if route_action not in {"tool", "workflow"} or not route_name:
        return ExtractDecision(False, "unsupported_route")

    if route_action == "tool" and route_name == "plant_task_create":
        return _should_extract_plan_task(text)
    if route_action == "workflow" and route_name == "crop_calendar_workflow":
        return _should_extract_crop_calendar(text)
    if route_action == "tool" and route_name == "sowing_suitability_lookup":
        return _should_extract_sowing(text)
    return ExtractDecision(False, "route_not_managed")


def _should_extract_plan_task(prompt: str) -> ExtractDecision:
    overrides = extract_field_overrides(
        prompt,
        ("plan_id", "plant_season_id", "name", "task_type", "date", "operator", "work_desc"),
    )
    normalized = _normalize_plan_task_fields(overrides)
    fields = tuple(sorted(normalized))
    if normalized:
        return ExtractDecision(True, "plan_task_has_any_field", fields)
    return ExtractDecision(False, "plan_task_low_field_coverage", fields)


def _should_extract_crop_calendar(prompt: str) -> ExtractDecision:
    overrides = extract_field_overrides(
        prompt,
        (
            "region_id",
            "variety",
            "culti_type",
            "planting_method",
            "sowing_date",
            "transplant_date",
            "crop",
        ),
    )
    strong_fields = {
        key
        for key in (
            "region_id",
            "variety",
            "culti_type",
            "planting_method",
            "sowing_date",
            "transplant_date",
        )
        if overrides.get(key) not in (None, "", [])
    }
    fields = tuple(sorted(strong_fields))
    if strong_fields:
        return ExtractDecision(True, "crop_calendar_has_any_field", fields)
    return ExtractDecision(False, "crop_calendar_low_field_coverage", fields)


def _should_extract_sowing(prompt: str) -> ExtractDecision:
    overrides = extract_field_overrides(
        prompt,
        ("variety", "region_id", "culti_type", "planting_method", "crop"),
    )
    core_fields = {
        key
        for key in ("variety", "region_id", "culti_type", "planting_method")
        if overrides.get(key) not in (None, "", [])
    }
    fields = tuple(sorted(core_fields))
    if not core_fields:
        return ExtractDecision(False, "sowing_no_core_fields", fields)
    return ExtractDecision(True, "sowing_has_any_field", fields)


def _normalize_plan_task_fields(overrides: dict[str, object]) -> set[str]:
    normalized: set[str] = set()
    for key, value in dict(overrides or {}).items():
        if value in (None, "", []):
            continue
        if key in {"plan_id", "plant_season_id"}:
            normalized.add("plan_id")
            continue
        normalized.add(str(key))
    return normalized
