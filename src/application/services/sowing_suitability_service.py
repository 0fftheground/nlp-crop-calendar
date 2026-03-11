from __future__ import annotations

import json
import re
from typing import Any, Dict, Mapping, Optional

from ..adapters import (
    DEFAULT_CONFIG_ADAPTER,
    DEFAULT_HTTP_ADAPTER,
    DEFAULT_SQL_ADAPTER,
)
from ..ports import ConfigPort, HttpPort, SqlPort
from ...agent.followup import build_tool_followup_invocation
from ...domain.planting import DEFAULT_CROP, extract_planting_details
from ...infra.db_catalog import TABLE_KEY_VARIETY, resolve_db_table
from ...infra.variety_store import extract_variety_tokens
from ...schemas.models import ToolInvocation
from .crop_calendar_service import (
    _coerce_region_id_value,
    _normalize_culti_type_code,
    _normalize_sowing_method_code,
    _resolve_code,
    _resolve_region_id_for_payload,
    configure_crop_calendar_ports,
)


_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER
_SQL_PORT: SqlPort = DEFAULT_SQL_ADAPTER


def configure_sowing_suitability_ports(
    *,
    config_port: Optional[ConfigPort] = None,
    http_port: Optional[HttpPort] = None,
    sql_port: Optional[SqlPort] = None,
) -> None:
    global _CONFIG_PORT, _HTTP_PORT, _SQL_PORT
    if config_port is not None:
        _CONFIG_PORT = config_port
    if http_port is not None:
        _HTTP_PORT = http_port
    if sql_port is not None:
        _SQL_PORT = sql_port
    configure_crop_calendar_ports(
        config_port=config_port,
        sql_port=sql_port,
        http_port=http_port,
    )


def _cfg():
    return _CONFIG_PORT.get()


def _post_json(
    url: str,
    *,
    payload: dict[str, object],
    headers: Optional[dict[str, str]] = None,
    timeout: float = 10.0,
):
    return _HTTP_PORT.post(
        url,
        json_payload=payload,
        headers=headers,
        timeout=timeout,
    )


def _fetch_all(url: str, sql: str, params: tuple[object, ...] = ()) -> list[dict]:
    return _SQL_PORT.fetch_all(url, sql, params)


def _qid(name: str) -> str:
    return _SQL_PORT.quote_identifier(name)


def _build_api_headers(*, api_key: Optional[str] = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = str(api_key or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-API-KEY"] = token
    return headers


def _require_db_url() -> str:
    cfg = _cfg()
    if not cfg.agri_db_url:
        raise RuntimeError("缺少 AGRI_DB_URL，无法读取品种数据。")
    return cfg.agri_db_url


def _get_sowing_suitability_api_url() -> Optional[str]:
    cfg = _cfg()
    raw = getattr(cfg, "sowing_suitability_api_url", None)
    if raw:
        return str(raw).strip()
    base = str(getattr(cfg, "business_api_base_url", None) or "").strip().rstrip("/")
    if base:
        return f"{base}/bozhong_syd"
    weather_url = str(getattr(cfg, "farm_weather_api_url", None) or "").strip()
    if not weather_url:
        return None
    if weather_url.endswith("/suit_rili"):
        return weather_url[: -len("/suit_rili")] + "/bozhong_syd"
    return weather_url.rstrip("/") + "/bozhong_syd"


def _load_prompt_payload(prompt: str) -> Optional[dict]:
    text = str(prompt or "").strip()
    if not text:
        return None
    candidate = text
    for _ in range(2):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, str):
            candidate = parsed
            continue
        return None
    return None


def _extract_query_text(prompt: str) -> str:
    payload = _load_prompt_payload(prompt)
    if not isinstance(payload, dict):
        return str(prompt or "").strip()
    for key in ("query", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    followup = payload.get("followup")
    if isinstance(followup, dict):
        value = followup.get("prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(prompt or "").strip()


def _extract_variety_hint(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in (
        r"(?:品种|种子|种的是|播的是)\s*[:：]?\s*([A-Za-z0-9\u4e00-\u9fff]{2,20})",
        r"([A-Za-z0-9\u4e00-\u9fff]{2,20}(?:号|优\d+|香\d+))",
    ):
        match = re.search(pattern, text)
        if match:
            value = str(match.group(1) or "").strip("，。；、,.!?！？ ")
            if value:
                return value
    tokens = extract_variety_tokens(text)
    for token in tokens:
        if any(ch.isdigit() for ch in token) or "号" in token:
            return token
    return None


def _build_query_from_prompt(prompt: str) -> Dict[str, object]:
    payload = _load_prompt_payload(prompt) or {}
    followup = payload.get("followup")
    draft: Dict[str, object] = {}
    followup_prompt = ""
    if isinstance(followup, Mapping):
        raw_draft = followup.get("draft")
        if isinstance(raw_draft, Mapping):
            draft.update(dict(raw_draft))
        value = followup.get("prompt")
        if isinstance(value, str) and value.strip():
            followup_prompt = value.strip()
    for key in (
        "variety",
        "culti_type",
        "planting_method",
        "region_id",
        "region",
        "farm_id",
        "crop",
    ):
        value = payload.get(key)
        if value is not None and value != "":
            draft[key] = value
    query_text = _extract_query_text(prompt)
    planting = extract_planting_details(query_text)
    if planting.culti_type and "culti_type" not in draft:
        draft["culti_type"] = planting.culti_type
    if planting.planting_method and "planting_method" not in draft:
        draft["planting_method"] = planting.planting_method
    if planting.region_id:
        if "region_id" not in draft and "region" not in draft:
            draft["region"] = planting.region_id
    if planting.crop and "crop" not in draft:
        draft["crop"] = planting.crop
    if followup_prompt:
        followup_planting = extract_planting_details(followup_prompt)
        if followup_planting.culti_type:
            draft["culti_type"] = followup_planting.culti_type
        if followup_planting.planting_method:
            draft["planting_method"] = followup_planting.planting_method
        if followup_planting.region_id and not draft.get("region_id"):
            draft["region_id"] = followup_planting.region_id
        if followup_planting.crop and not draft.get("crop"):
            draft["crop"] = followup_planting.crop
    variety = payload.get("variety")
    if not isinstance(variety, str) or not variety.strip():
        if isinstance(followup, Mapping):
            raw_draft = followup.get("draft")
            if isinstance(raw_draft, Mapping):
                raw_variety = raw_draft.get("variety")
                if isinstance(raw_variety, str) and raw_variety.strip():
                    variety = raw_variety.strip()
    if not isinstance(variety, str) or not variety.strip():
        variety = _extract_variety_hint(query_text)
    if isinstance(variety, str) and variety.strip():
        draft["variety"] = variety.strip()
    if "region_id" not in draft and "region" in draft:
        draft["region_id"] = draft.get("region")
    draft.setdefault("crop", DEFAULT_CROP)
    return draft


def _extract_contextual_region_hint(text: str) -> Optional[str]:
    prompt = str(text or "").strip()
    if not prompt:
        return None
    for pattern in (
        r"^在([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)?$",
        r"^([\u4e00-\u9fff]{2,20})(?:呢|吗|怎么样|如何|可以吗)$",
    ):
        match = re.search(pattern, prompt)
        if not match:
            continue
        region = str(match.group(1) or "").strip()
        region = re.sub(r"(呢|吗|呀|啊)$", "", region).strip()
        if region:
            return region
    return None


def build_contextual_sowing_query(
    prompt: str, context: Optional[Mapping[str, object]]
) -> Optional[dict[str, object]]:
    if not isinstance(context, Mapping):
        return None
    text = _extract_query_text(prompt)
    if not text:
        return None
    base = {
        key: value
        for key, value in dict(context).items()
        if key in {"variety", "culti_type", "planting_method", "region_id", "farm_id", "crop"}
        and value not in (None, "")
    }
    if not base:
        return None
    current = _build_query_from_prompt(json.dumps({"query": text}, ensure_ascii=False))
    overrides = {
        key: value
        for key, value in current.items()
        if key in {"variety", "culti_type", "planting_method", "region_id", "farm_id", "crop"}
        and value not in (None, "")
    }
    if "region_id" not in overrides:
        region_hint = _extract_contextual_region_hint(text)
        if region_hint:
            overrides["region_id"] = region_hint
            overrides.pop("farm_id", None)
    if not overrides:
        return None
    merged = dict(base)
    merged.update(overrides)
    merged["query"] = text
    return merged


def _build_followup(prompt: str, *, draft: Mapping[str, object], missing: list[str]) -> ToolInvocation:
    labels = {
        "variety": "品种名",
        "culti_type": "稻作类型",
        "planting_method": "播种方式",
    }
    fields = "、".join(labels.get(name, name) for name in missing)
    return build_tool_followup_invocation(
        name="sowing_suitability_lookup",
        message=f"请补充{fields}，我才能给出播期推荐。",
        missing_fields=missing,
        draft=dict(draft),
        query=prompt,
    )


def _fetch_variety_sub_type(variety_name: str) -> Optional[int]:
    if not variety_name:
        return None
    url = _require_db_url()
    table = resolve_db_table(_cfg(), TABLE_KEY_VARIETY)
    try:
        sql = (
            f"SELECT {_qid('sub_type')} AS sub_type FROM {_qid(table)} "
            f"WHERE {_qid('name')} = %s LIMIT 1"
        )
        rows = _fetch_all(url, sql, (variety_name,))
    except Exception:
        return None
    if not rows or not isinstance(rows[0], dict):
        return None
    raw = rows[0].get("sub_type")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(raw)
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return _resolve_code("sub_type", text)


def _normalize_crop_code(value: object) -> int:
    text = str(value or "").strip()
    if not text or text == DEFAULT_CROP:
        return 0
    return 0


def _build_request_payload(query: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    draft = dict(query)
    missing = [
        field
        for field in ("variety", "culti_type", "planting_method")
        if not str(draft.get(field) or "").strip()
    ]
    if missing:
        raise ValueError(json.dumps({"type": "followup", "missing": missing}, ensure_ascii=False))
    sowing_method = _normalize_sowing_method_code(draft.get("planting_method"))
    if sowing_method is None:
        raise RuntimeError("无法解析播种方式代码。")
    culti_type = _normalize_culti_type_code(draft.get("culti_type"))
    if culti_type is None:
        raise RuntimeError("无法解析稻作类型代码。")
    sub_type = _fetch_variety_sub_type(str(draft.get("variety") or "").strip())
    if sub_type is None:
        raise RuntimeError(f"未找到品种亚种类型: {draft.get('variety')}")
    region_raw = str(draft.get("region_id") or "").strip()
    resolved_region_id = _resolve_region_id_for_payload(region_raw)
    farm_id: Optional[int] = None
    if region_raw and resolved_region_id is None:
        raise RuntimeError(f"暂不支持该区域的播期推荐：{region_raw}")
    if resolved_region_id is None:
        raw_farm_id = draft.get("farm_id") or getattr(_cfg(), "default_farm_id", None)
        if not raw_farm_id:
            raise RuntimeError("缺少区域或 DEFAULT_FARM_ID，无法查询播期推荐。")
        try:
            farm_id = int(str(raw_farm_id).strip())
        except Exception as exc:
            raise RuntimeError(f"farm_id 非法: {raw_farm_id}") from exc
    request_payload: dict[str, object] = {
        "culti_type": culti_type,
        "sowing_method": sowing_method,
        "sub_type": sub_type,
        "crop": _normalize_crop_code(draft.get("crop")),
    }
    if resolved_region_id is not None:
        request_payload["region_id"] = _coerce_region_id_value(resolved_region_id)
    elif farm_id is not None:
        request_payload["farm_id"] = farm_id
    return request_payload, {
        "variety": draft.get("variety"),
        "culti_type": draft.get("culti_type"),
        "planting_method": draft.get("planting_method"),
        "region_id": region_raw or draft.get("region_id"),
        "farm_id": farm_id,
        "crop": draft.get("crop") or DEFAULT_CROP,
        "sub_type": sub_type,
    }


def lookup_sowing_suitability(prompt: str) -> ToolInvocation:
    text = str(prompt or "")
    draft = _build_query_from_prompt(text)
    missing = [
        field
        for field in ("variety", "culti_type", "planting_method")
        if not str(draft.get(field) or "").strip()
    ]
    if missing:
        return _build_followup(text, draft=draft, missing=missing)
    try:
        request_payload, resolved = _build_request_payload(draft)
    except ValueError as exc:
        try:
            payload = json.loads(str(exc))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("type") == "followup":
            missing = payload.get("missing") or []
            return _build_followup(text, draft=draft, missing=list(missing))
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message=str(exc),
            data={},
        )
    except RuntimeError as exc:
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message=str(exc),
            data={},
        )
    url = _get_sowing_suitability_api_url()
    if not url:
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message="缺少播期推荐接口地址。",
            data={},
        )
    try:
        response = _post_json(
            url,
            payload=request_payload,
            headers=_build_api_headers(
                api_key=getattr(_cfg(), "business_api_key", None)
            ),
            timeout=10.0,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message=f"查询播期推荐失败: {exc}",
            data={"request": request_payload, "resolved": resolved},
        )
    if not isinstance(payload, dict):
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message="播期推荐接口返回格式未识别。",
            data={"request": request_payload, "resolved": resolved},
        )
    code = str(payload.get("code", "")).strip()
    message = str(payload.get("message") or payload.get("msg") or "").strip()
    if code and code != "200":
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message=message or "播期推荐接口返回失败。",
            data={
                "request": request_payload,
                "resolved": resolved,
                "raw": payload,
            },
        )
    data = payload.get("data")
    result_data = data if isinstance(data, dict) else {}
    return ToolInvocation(
        name="sowing_suitability_lookup",
        message=message or "已获取播期推荐。",
        data={
            "request": request_payload,
            "resolved": resolved,
            "result": result_data,
        },
    )
