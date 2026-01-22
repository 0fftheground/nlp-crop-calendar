from __future__ import annotations

import json
from typing import Optional

from ..schemas import PlantingDetails


def build_planting_cache_key(planting: PlantingDetails) -> str:
    payload = planting.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def parse_planting_cache_key(payload: str) -> Optional[str]:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    planting_payload = data.get("planting", data)
    if not isinstance(planting_payload, dict):
        return None
    try:
        planting = PlantingDetails.model_validate(planting_payload)
    except Exception:
        return None
    return build_planting_cache_key(planting)
