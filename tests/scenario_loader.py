from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "tests" / "scenarios"


def load_yaml_scenarios(name: str) -> Any:
    path = SCENARIO_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
