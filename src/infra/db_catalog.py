from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import AppConfig


TABLE_KEY_VARIETY = "variety"

DEFAULT_DB_TABLES: dict[str, str] = {
    TABLE_KEY_VARIETY: "agri_rice_variety",
}

LEGACY_TABLE_FIELDS: dict[str, str] = {
    TABLE_KEY_VARIETY: "variety_db_table",
}


@dataclass(frozen=True)
class RegionLookupSource:
    table: str
    id_column: str
    name_column: str


DEFAULT_REGION_LOOKUP_SOURCES: tuple[RegionLookupSource, ...] = (
    RegionLookupSource("public.agri_region", "id", "name"),
    RegionLookupSource("agri_region_dict", "region_id", "region_name"),
    RegionLookupSource("region_dict", "id", "name"),
    RegionLookupSource("sys_region", "id", "name"),
)


def _clean_text(value: object) -> str:
    return str(value or "").strip()


def resolve_db_table(config: AppConfig, key: str) -> str:
    key_text = _clean_text(key).lower()
    if not key_text:
        raise ValueError("table key is required")

    legacy_field = LEGACY_TABLE_FIELDS.get(key_text)
    if legacy_field:
        legacy_value = _clean_text(getattr(config, legacy_field, None))
        if legacy_value:
            return legacy_value

    default_table = DEFAULT_DB_TABLES.get(key_text)
    if default_table:
        return default_table
    raise KeyError(f"Unknown db table key: {key_text}")


def _iter_region_candidates(
    raw_candidates: Iterable[Mapping[str, object]] | None,
) -> list[RegionLookupSource]:
    if not raw_candidates:
        return []
    items: list[RegionLookupSource] = []
    for raw in raw_candidates:
        table = _clean_text(raw.get("table"))
        id_column = _clean_text(raw.get("id_column"))
        name_column = _clean_text(raw.get("name_column"))
        if table and id_column and name_column:
            items.append(
                RegionLookupSource(
                    table=table,
                    id_column=id_column,
                    name_column=name_column,
                )
            )
    return items


def resolve_region_lookup_sources(config: AppConfig) -> list[RegionLookupSource]:
    seen: set[tuple[str, str, str]] = set()
    resolved: list[RegionLookupSource] = []

    def add(source: RegionLookupSource) -> None:
        key = (source.table, source.id_column, source.name_column)
        if key in seen:
            return
        seen.add(key)
        resolved.append(source)

    for source in _iter_region_candidates(config.db_region_lookup_candidates):
        add(source)

    legacy_table = _clean_text(getattr(config, "region_db_table", None))
    if legacy_table:
        add(
            RegionLookupSource(
                table=legacy_table,
                id_column=_clean_text(
                    getattr(config, "region_db_id_column", "region_id")
                )
                or "region_id",
                name_column=_clean_text(
                    getattr(config, "region_db_name_column", "region_name")
                )
                or "region_name",
            )
        )

    for source in DEFAULT_REGION_LOOKUP_SOURCES:
        add(source)
    return resolved
