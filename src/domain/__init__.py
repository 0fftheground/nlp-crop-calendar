from __future__ import annotations

from typing import Any


_SERVICE_EXPORTS = {
    "CropCalendarArtifacts",
    "MissingPlantingInfoError",
    "assemble_weather_series",
    "build_operation_plan",
    "derive_weather_range",
    "extract_planting_details",
    "fetch_weather",
    "generate_crop_calendar",
    "list_missing_required_fields",
    "merge_planting_answers",
    "normalize_and_validate_planting",
    "predict_growth_stage",
}

__all__ = sorted(_SERVICE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SERVICE_EXPORTS:
        from . import services as _services

        return getattr(_services, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
