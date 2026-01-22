from __future__ import annotations

from typing import Optional


def normalize_provider(value: Optional[str]) -> str:
    return (value or "mock").lower()
