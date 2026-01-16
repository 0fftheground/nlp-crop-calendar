"""Simple file export store for CSV downloads."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4


_FILE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _export_root() -> Path:
    return Path(__file__).resolve().parents[2] / ".cache" / "exports"


def write_export(content: str, *, suffix: str = "csv") -> str:
    export_dir = _export_root()
    export_dir.mkdir(parents=True, exist_ok=True)
    file_id = uuid4().hex
    path = export_dir / f"{file_id}.{suffix}"
    path.write_text(content, encoding="utf-8")
    return file_id


def resolve_export_path(file_id: str, *, suffix: str = "csv") -> Path:
    if not _FILE_ID_RE.match(file_id or ""):
        raise ValueError("invalid export id")
    return _export_root() / f"{file_id}.{suffix}"
