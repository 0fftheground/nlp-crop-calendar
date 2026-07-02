from __future__ import annotations

import json
from typing import Dict, Optional

from ..adapters import DEFAULT_CONFIG_ADAPTER, DEFAULT_HTTP_ADAPTER
from ..ports import ConfigPort, HttpPort
from ...observability.logging_utils import log_event, summarize_text


_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER


def configure_plan_task_ports(
    *,
    config_port: Optional[ConfigPort] = None,
    http_port: Optional[HttpPort] = None,
) -> None:
    global _CONFIG_PORT, _HTTP_PORT
    if config_port is not None:
        _CONFIG_PORT = config_port
    if http_port is not None:
        _HTTP_PORT = http_port


def _cfg():
    return _CONFIG_PORT.get()


def _post_json(
    url: str,
    *,
    payload: dict,
    headers: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
):
    return _HTTP_PORT.post(
        url,
        json_payload=payload,
        headers=headers,
        timeout=timeout,
    )


def _build_api_headers(*, api_key: Optional[str] = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-KEY"] = token
    return headers


def _join_api_url(base_url: Optional[str], suffix: str) -> Optional[str]:
    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return None
    return f"{base}/{suffix.lstrip('/')}"


def _get_plan_task_create_api_url(plan_id: object) -> Optional[str]:
    cfg = _cfg()
    raw = str(getattr(cfg, "task_create_api_url", "") or "").strip()
    plan_id_text = str(plan_id).strip()
    if raw:
        if "{plan_id}" in raw:
            return raw.format(plan_id=plan_id_text)
        return f"{raw.rstrip('/')}/{plan_id_text}"
    return _join_api_url(
        getattr(cfg, "business_api_base_url", None),
        f"/api/tasks/{plan_id_text}",
    )


def _format_plan_task_http_error(
    *, status_code: Optional[object], response_text: str
) -> str:
    text = str(response_text or "").strip()
    if text:
        try:
            payload = json.loads(text)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = str(
                payload.get("detail") or payload.get("msg") or payload.get("message") or ""
            ).strip()
            if detail:
                return f"农事录入失败：{detail}"
        return f"农事录入失败：{text}"
    if status_code not in (None, ""):
        return f"农事录入失败（status={status_code}）。"
    return "农事录入失败。"


def create_or_record_plan_task(
    plan_id: object,
    payload: Dict[str, object],
) -> Dict[str, object]:
    cfg = _cfg()
    url = _get_plan_task_create_api_url(plan_id)
    if not url:
        raise RuntimeError("缺少农事录入接口地址。")
    request_body = dict(payload)
    log_event(
        "plan_task_create_api_request",
        url=url,
        payload=request_body,
    )
    try:
        response = _post_json(
            url,
            payload=request_body,
            headers=_build_api_headers(api_key=getattr(cfg, "business_api_key", None)),
            timeout=10.0,
        )
        response.raise_for_status()
    except Exception as exc:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status_code = getattr(resp, "status_code", None)
            response_text = summarize_text(getattr(resp, "text", str(exc)), limit=1200)
            log_event(
                "plan_task_create_api_http_error",
                url=url,
                payload=request_body,
                status_code=status_code,
                response_text=response_text,
            )
            raise RuntimeError(
                _format_plan_task_http_error(
                    status_code=status_code,
                    response_text=response_text,
                )
            ) from exc
        else:
            log_event(
                "plan_task_create_api_request_error",
                url=url,
                payload=request_body,
                error=str(exc),
            )
        raise RuntimeError("农事录入失败。") from exc
    try:
        raw = response.json()
    except Exception as exc:
        log_event(
            "plan_task_create_api_parse_error",
            url=url,
            payload=request_body,
            status_code=response.status_code,
            response_text=summarize_text(response.text or "", limit=1200),
        )
        raise RuntimeError("农事录入接口返回格式未识别。") from exc
    log_event(
        "plan_task_create_api_response",
        url=url,
        payload=request_body,
        status_code=response.status_code,
        response_summary=summarize_text(
            json.dumps(raw, ensure_ascii=False, default=str), limit=1200
        ),
    )
    if not isinstance(raw, dict):
        raise RuntimeError("农事录入接口返回格式未识别。")
    code = str(raw.get("code", "")).strip()
    if code and code not in {"0", "200"}:
        msg = str(raw.get("msg") or raw.get("message") or "农事录入失败。").strip()
        log_event(
            "plan_task_create_api_business_error",
            url=url,
            payload=request_body,
            code=code,
            msg=msg,
        )
        raise RuntimeError(msg)
    status = str(raw.get("status") or "").strip()
    if status.lower() in {"error", "failed", "fail"}:
        msg = str(raw.get("msg") or raw.get("message") or "农事录入失败。").strip()
        raise RuntimeError(msg)
    return raw
