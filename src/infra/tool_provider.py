from __future__ import annotations

from typing import Dict, Optional

import httpx

from ..schemas import ToolInvocation


INTRANET_TIMEOUT = 10.0


def normalize_provider(value: Optional[str]) -> str:
    return (value or "mock").lower()


def build_intranet_headers(api_key: Optional[str]) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def invoke_intranet_tool(
    tool_name: str,
    prompt: str,
    api_url: Optional[str],
    api_key: Optional[str],
) -> ToolInvocation:
    if not api_url:
        return ToolInvocation(
            name=tool_name,
            message="intranet provider not configured",
            data={},
        )
    try:
        with httpx.Client(timeout=INTRANET_TIMEOUT, trust_env=False) as client:
            response = client.post(
                api_url,
                json={"prompt": prompt},
                headers=build_intranet_headers(api_key),
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        return ToolInvocation(
            name=tool_name,
            message=f"intranet request failed: {exc}",
            data={},
        )

    if isinstance(payload, dict):
        name = payload.get("name") or tool_name
        message = payload.get("message") or "intranet response received"
        data = payload.get("data")
        if data is None:
            data = {k: v for k, v in payload.items() if k not in {"name", "message"}}
        if not isinstance(data, dict):
            data = {"payload": data}
        return ToolInvocation(name=name, message=message, data=data)
    return ToolInvocation(
        name=tool_name,
        message="intranet response received",
        data={"payload": payload},
    )


def maybe_intranet_tool(
    tool_name: str,
    prompt: str,
    provider: str,
    api_url: Optional[str],
    api_key: Optional[str],
) -> Optional[ToolInvocation]:
    provider = normalize_provider(provider)
    if provider in {"mock", "local", "sqlite"}:
        return None
    if provider == "intranet":
        return invoke_intranet_tool(tool_name, prompt, api_url, api_key)
    return ToolInvocation(
        name=tool_name,
        message=f"unsupported provider: {provider}",
        data={},
    )
