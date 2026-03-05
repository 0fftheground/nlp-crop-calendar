from __future__ import annotations

from typing import Any, Optional, Protocol, Sequence


class ConfigPort(Protocol):
    def get(self) -> Any:
        """Return application configuration object."""


class SqlPort(Protocol):
    def fetch_all(
        self, url: str, sql: str, params: Sequence[object] | None = None
    ) -> list[dict]:
        """Execute a query and return rows."""

    def quote_identifier(self, name: str) -> str:
        """Quote a database identifier."""


class HttpResponsePort(Protocol):
    @property
    def status_code(self) -> int:
        ...

    @property
    def text(self) -> str:
        ...

    def json(self) -> Any:
        ...

    def raise_for_status(self) -> None:
        ...


class HttpPort(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Optional[dict[str, object]] = None,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> HttpResponsePort:
        """Send an HTTP GET request and return response."""

    def post(
        self,
        url: str,
        *,
        json_payload: dict,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0,
    ) -> HttpResponsePort:
        """Send an HTTP POST request and return response."""
