from __future__ import annotations

import json

from ..schemas import PlantingDetails


def build_planting_cache_key(planting: PlantingDetails) -> str:
    payload = planting.model_dump(mode="json", exclude_none=True)
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
