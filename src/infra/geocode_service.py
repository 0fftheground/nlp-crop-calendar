from __future__ import annotations

from typing import Optional, Tuple

import httpx

from .config import get_config
from .geocode_cache import get_geocode_cached, set_geocode_cached


def _parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_geocode_location(location: object) -> Optional[Tuple[float, float]]:
    if location is None:
        return None
    parts = str(location).split(",")
    if len(parts) != 2:
        return None
    lon = _parse_float(parts[0])
    lat = _parse_float(parts[1])
    if lat is None or lon is None:
        return None
    return lat, lon


def _normalize_geocode_city(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return None
    text = str(value).strip()
    return text or None


def geocode_with_amap(address: str, *, raw_region: Optional[str] = None) -> Optional[dict]:
    if not address:
        return None
    cached = get_geocode_cached(address)
    if cached:
        if raw_region and not cached.get("input_region"):
            updated = dict(cached)
            updated["input_region"] = raw_region
            set_geocode_cached(address, None, updated)
            return updated
        return cached
    cfg = get_config()
    if not cfg.amap_api_key:
        return None
    url = cfg.amap_geocode_url or "https://restapi.amap.com/v3/geocode/geo"
    params = {"key": cfg.amap_api_key, "address": address.strip()}
    try:
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("status")) != "1":
        return None
    geocodes = payload.get("geocodes")
    if not isinstance(geocodes, list) or not geocodes:
        return None
    primary = geocodes[0] if isinstance(geocodes[0], dict) else None
    if not primary:
        return None
    location = _parse_geocode_location(primary.get("location"))
    if location is None:
        return None
    lat, lon = location
    result = {
        "address": address,
        "input_region": raw_region or address,
        "formatted_address": primary.get("formatted_address"),
        "lat": lat,
        "lon": lon,
        "province": primary.get("province"),
        "city": _normalize_geocode_city(primary.get("city")),
        "district": primary.get("district"),
        "level": primary.get("level"),
        "adcode": primary.get("adcode"),
    }
    set_geocode_cached(address, None, result)
    return result


def annotate_geocode_gdd_region(address: str, gdd_region: str) -> None:
    if not address or not gdd_region:
        return
    cached = get_geocode_cached(address)
    if not cached:
        return
    updated = dict(cached)
    updated["gdd_region"] = gdd_region
    set_geocode_cached(address, None, updated)
