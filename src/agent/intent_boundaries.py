from __future__ import annotations

import re

def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()
