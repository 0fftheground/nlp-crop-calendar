from __future__ import annotations

from typing import Callable, Dict, Optional

from ...domain.planting import extract_planting_details as _extract_planting_details
from ...infra.variety_store import find_exact_variety_in_text
from ...schemas import PlantingDetailsDraft


def _resolve_variety_candidates(prompt: str) -> list[str]:
    exact = find_exact_variety_in_text(prompt)
    return [exact] if exact else []


def extract_planting_details(
    prompt: str,
    *,
    llm_extract: Optional[Callable[[str], Dict[str, object]]] = None,
) -> PlantingDetailsDraft:
    return _extract_planting_details(
        prompt,
        llm_extract=llm_extract,
        variety_resolver=_resolve_variety_candidates,
    )
