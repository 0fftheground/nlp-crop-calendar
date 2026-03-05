from __future__ import annotations

from typing import Callable, Dict, Optional

from ...domain.planting import (
    extract_planting_details as _extract_planting_details,
    normalize_and_validate_planting as _normalize_and_validate_planting,
)
from ...infra.variety_store import find_exact_variety_in_text
from ...observability.logging_utils import log_event
from ...schemas import PlantingDetailsDraft
from ...schemas import PlantingDetails


def _resolve_variety_candidates(prompt: str) -> list[str]:
    exact = find_exact_variety_in_text(prompt)
    return [exact] if exact else []


class PlantingService:
    def __init__(
        self,
        *,
        variety_resolver: Callable[[str], list[str]] = _resolve_variety_candidates,
    ) -> None:
        self._variety_resolver = variety_resolver

    def extract_planting_details(
        self,
        prompt: str,
        *,
        llm_extract: Optional[Callable[[str], Dict[str, object]]] = None,
    ) -> PlantingDetailsDraft:
        return _extract_planting_details(
            prompt,
            llm_extract=llm_extract,
            variety_resolver=self._variety_resolver,
        )

    def normalize_and_validate_planting(self, draft: object) -> PlantingDetails:
        planting = _normalize_and_validate_planting(draft)
        log_event(
            "normalized_planting",
            planting=planting.model_dump(mode="json"),
        )
        return planting


_DEFAULT_PLANTING_SERVICE = PlantingService()


def extract_planting_details(
    prompt: str,
    *,
    llm_extract: Optional[Callable[[str], Dict[str, object]]] = None,
) -> PlantingDetailsDraft:
    return _DEFAULT_PLANTING_SERVICE.extract_planting_details(
        prompt,
        llm_extract=llm_extract,
    )


def normalize_and_validate_planting(draft: object) -> PlantingDetails:
    return _DEFAULT_PLANTING_SERVICE.normalize_and_validate_planting(draft)
