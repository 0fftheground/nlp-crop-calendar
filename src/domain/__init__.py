from __future__ import annotations

from typing import Any


_SERVICE_EXPORTS = {
    "DEFAULT_CROP",
    "CROP_KEYWORDS",
    "CROP_REQUIRED_FIELDS",
    "METHOD_KEYWORDS",
    "MissingPlantingInfoError",
    "extract_planting_details",
    "list_missing_required_fields",
    "merge_planting_answers",
    "normalize_and_validate_planting",
}

__all__ = sorted(_SERVICE_EXPORTS)


def __getattr__(name: str) -> Any:
    if name in _SERVICE_EXPORTS:
        from . import services as _services

        return getattr(_services, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
