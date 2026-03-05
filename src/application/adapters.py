from __future__ import annotations

from typing import Optional, Sequence

import httpx

from ..infra.config import get_config
from ..infra.postgres import fetch_all, quote_identifier
from .ports import ConfigPort, HttpPort, HttpResponsePort, SqlPort


class DefaultConfigAdapter(ConfigPort):
    def get(self):
        return get_config()


class DefaultSqlAdapter(SqlPort):
    def fetch_all(
        self, url: str, sql: str, params: Sequence[object] | None = None
    ) -> list[dict]:
        return fetch_all(url, sql, tuple(params or ()))

    def quote_identifier(self, name: str) -> str:
        return quote_identifier(name)


class DefaultHttpResponseAdapter(HttpResponsePort):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def text(self) -> str:
        return self._response.text

    def json(self):
        return self._response.json()

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


class DefaultHttpAdapter(HttpPort):
    def get(
        self,
        url: str,
        *,
        params: Optional[dict[str, object]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> HttpResponsePort:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.get(url, params=params or {}, headers=headers or {})
        return DefaultHttpResponseAdapter(response)

    def post(
        self,
        url: str,
        *,
        json_payload: dict,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> HttpResponsePort:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(url, json=json_payload, headers=headers or {})
        return DefaultHttpResponseAdapter(response)


DEFAULT_CONFIG_ADAPTER = DefaultConfigAdapter()
DEFAULT_SQL_ADAPTER = DefaultSqlAdapter()
DEFAULT_HTTP_ADAPTER = DefaultHttpAdapter()
