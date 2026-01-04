"""In-memory cache for weather series data."""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import Optional
from uuid import uuid4

from ..schemas import WeatherSeries


_MAX_CACHE_SIZE = 128
_CACHE: "OrderedDict[str, WeatherSeries]" = OrderedDict()
_CACHE_LOCK = Lock()


def store_weather_series(series: WeatherSeries) -> str:
    """Store a weather series and return a cache reference key."""
    cache_key = uuid4().hex
    with _CACHE_LOCK:
        _CACHE[cache_key] = series
        _CACHE.move_to_end(cache_key)
        if len(_CACHE) > _MAX_CACHE_SIZE:
            _CACHE.popitem(last=False)
    return cache_key


def get_weather_series(cache_key: Optional[str]) -> Optional[WeatherSeries]:
    """Retrieve a cached weather series by key."""
    if not cache_key:
        return None
    with _CACHE_LOCK:
        series = _CACHE.get(cache_key)
        if series is None:
            return None
        _CACHE.move_to_end(cache_key)
        return series
