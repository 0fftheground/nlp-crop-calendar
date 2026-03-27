from __future__ import annotations

import json
import re
from typing import Dict, Mapping, Optional

from pydantic import BaseModel, Field

from ..adapters import (
    DEFAULT_CONFIG_ADAPTER,
    DEFAULT_HTTP_ADAPTER,
    DEFAULT_SQL_ADAPTER,
)
from ..ports import ConfigPort, HttpPort, SqlPort
from ...agent.followup import build_tool_followup_invocation, resolve_followup_choice
from ...agent.field_updates import (
    extract_planting_field_overrides,
    extract_region_followup_hint,
)
from ...agent.intent_boundaries import looks_like_sowing_query
from ...domain.planting import DEFAULT_CROP, extract_planting_details
from ...domain.region_text import build_region_text_variants, normalize_region_token
from ...infra.llm_extract import llm_structured_extract
from ...infra.db_catalog import TABLE_KEY_VARIETY, resolve_db_table
from ...infra.variety_db_schema import (
    VARIETY_PG_COLUMN_MAP,
    VARIETY_PG_NAME_COLUMN,
)
from ...infra.variety_store import (
    extract_variety_tokens,
    find_exact_variety_in_text,
    retrieve_variety_candidates,
)
from ...observability.logging_utils import log_event, summarize_text
from ...prompts.planting_extract import build_planting_extract_prompt
from ...schemas.models import ToolInvocation
from .crop_calendar_service import (
    _coerce_region_id_value,
    _normalize_culti_type_code,
    _normalize_sowing_method_code,
    _resolve_code,
    _resolve_region_id_for_payload,
    resolve_culti_type_label,
    configure_crop_calendar_ports,
)


_CONFIG_PORT: ConfigPort = DEFAULT_CONFIG_ADAPTER
_HTTP_PORT: HttpPort = DEFAULT_HTTP_ADAPTER
_SQL_PORT: SqlPort = DEFAULT_SQL_ADAPTER

_PROVINCE_CODE_MAP = {
    "11": "北京",
    "12": "天津",
    "13": "河北",
    "14": "山西",
    "15": "内蒙古",
    "21": "辽宁",
    "22": "吉林",
    "23": "黑龙江",
    "31": "上海",
    "32": "江苏",
    "33": "浙江",
    "34": "安徽",
    "35": "福建",
    "36": "江西",
    "37": "山东",
    "41": "河南",
    "42": "湖北",
    "43": "湖南",
    "44": "广东",
    "45": "广西",
    "46": "海南",
    "50": "重庆",
    "51": "四川",
    "52": "贵州",
    "53": "云南",
    "54": "西藏",
    "61": "陕西",
    "62": "甘肃",
    "63": "青海",
    "64": "宁夏",
    "65": "新疆",
    "71": "台湾",
    "81": "香港",
    "82": "澳门",
}


class _SowingPlantingExtract(BaseModel):
    region_id: Optional[str] = Field(default=None)
    crop: Optional[str] = None
    variety: Optional[str] = None
    culti_type: Optional[str] = None
    planting_method: Optional[str] = None


def _llm_extract_planting_for_sowing(prompt: str) -> Dict[str, object]:
    return llm_structured_extract(
        prompt,
        schema=_SowingPlantingExtract,
        system_prompt=build_planting_extract_prompt(
            "播期推荐场景下，若用户提供品种名、稻作类型、播种方式、区域，请尽量准确抽取。"
        ),
    )


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
    invalid_cues = ("我在", "种植", "播种", "移栽", "适合", "什么时候", "何时", "合适")
    for pattern in (
        r"(?:品种|种子|种的是|播的是)\s*[:：]?\s*([A-Za-z0-9\u4e00-\u9fff]{2,20})",
        r"([A-Za-z0-9\u4e00-\u9fff]{2,20}(?:号|优\d+|香\d+))",
        r"([A-Za-z\u4e00-\u9fff]{1,12}\d{1,6}(?:号)?)",
    ):
        match = re.search(pattern, text)
        if match:
            value = str(match.group(1) or "").strip("，。；、,.!?！？ ")
            if value and not any(token in value for token in invalid_cues):
                return value
    cue_index = -1
    cue_length = 0
    for cue in (
        "品种",
        "种子",
        "种的是",
        "播的是",
        "种植",
        "一季晚稻",
        "双季晚稻",
        "双季早稻",
        "一季稻",
        "单季稻",
        "再生稻",
        "早稻",
        "中稻",
        "晚稻",
    ):
        index = text.rfind(cue)
        if index > cue_index:
            cue_index = index
            cue_length = len(cue)
    if cue_index >= 0:
        tail = text[cue_index + cue_length :]
        for stop in ("，", ",", "。", "；", ";", "、", "适合", "合适", "什么时候", "何时", "吗", "呢", "移栽", "插秧", "直播", "播种"):
            if stop in tail:
                tail = tail.split(stop, 1)[0]
        tail = tail.strip("，。；、,.!?！？ ")
        if tail and (any(ch.isdigit() for ch in tail) or "号" in tail):
            return tail
    tokens = extract_variety_tokens(text)
    for token in tokens:
        if (any(ch.isdigit() for ch in token) or "号" in token) and not any(
            cue in token for cue in invalid_cues
        ):
            return token
    return None


def _resolve_followup_variety_candidate(payload: Mapping[str, object]) -> Optional[str]:
    followup = payload.get("followup")
    if not isinstance(followup, Mapping):
        return None
    answer = followup.get("prompt")
    if not isinstance(answer, str) or not answer.strip():
        return None
    draft = followup.get("draft")
    if not isinstance(draft, Mapping):
        return None
    raw_candidates = draft.get("candidates") or draft.get("variety_candidates")
    if not isinstance(raw_candidates, list):
        return None
    candidates = [str(item).strip() for item in raw_candidates if str(item).strip()]
    if not candidates:
        return None
    return resolve_followup_choice(answer.strip(), candidates)


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
    chosen_variety = _resolve_followup_variety_candidate(payload)
    if chosen_variety:
        draft["variety"] = chosen_variety
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
    planting = extract_planting_details(
        query_text, llm_extract=_llm_extract_planting_for_sowing
    )
    if planting.variety and "variety" not in draft:
        draft["variety"] = planting.variety
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
        followup_planting = extract_planting_details(
            followup_prompt, llm_extract=_llm_extract_planting_for_sowing
        )
        followup_variety = _extract_variety_hint(followup_prompt)
        if followup_planting.variety and not draft.get("variety"):
            draft["variety"] = followup_planting.variety
        if followup_planting.culti_type:
            draft["culti_type"] = followup_planting.culti_type
        if followup_planting.planting_method:
            draft["planting_method"] = followup_planting.planting_method
        if followup_planting.region_id and not draft.get("region_id"):
            draft["region_id"] = followup_planting.region_id
        if followup_planting.crop and not draft.get("crop"):
            draft["crop"] = followup_planting.crop
        if followup_variety and not draft.get("variety"):
            draft["variety"] = followup_variety
    if not str(draft.get("variety") or "").strip():
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
        if (not isinstance(variety, str) or not variety.strip()) and followup_prompt:
            variety = _extract_variety_hint(followup_prompt)
        if isinstance(variety, str) and variety.strip():
            draft["variety"] = variety.strip()
    if "region_id" not in draft and "region" in draft:
        draft["region_id"] = draft.get("region")
    draft.setdefault("crop", DEFAULT_CROP)
    return draft


def _extract_contextual_region_hint(text: str) -> Optional[str]:
    return extract_region_followup_hint(text, invalid_tokens=())


def _extract_contextual_sowing_overrides(text: str) -> dict[str, object]:
    overrides = extract_planting_field_overrides(
        text,
        include_variety=True,
        include_dates=False,
        include_crop=True,
        variety_matcher=find_exact_variety_in_text,
    )
    # Short follow-ups like "早稻呢" / "直播呢" should only override the explicitly
    # extracted field, not be reinterpreted as a region.
    if "region_id" not in overrides and not any(
        key in overrides for key in ("variety", "culti_type", "planting_method")
    ):
        region_hint = _extract_contextual_region_hint(text)
        if region_hint:
            overrides["region_id"] = region_hint
    return overrides


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
    overrides = _extract_contextual_sowing_overrides(text)
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    elif not looks_like_sowing_query(text):
        return None
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


def _build_variety_candidate_followup(
    prompt: str,
    *,
    draft: Mapping[str, object],
    candidates: list[str],
) -> ToolInvocation:
    options = [str(item).strip() for item in candidates if str(item).strip()]
    lines = ["未找到完全匹配的品种。你是不是想查询以下品种："]
    for idx, name in enumerate(options, start=1):
        lines.append(f"{idx}. {name}")
    lines.append("请回复序号或品种名称。")
    followup_draft = dict(draft)
    followup_draft["variety"] = None
    followup_draft["candidates"] = options
    return build_tool_followup_invocation(
        name="sowing_suitability_lookup",
        message="\n".join(lines),
        missing_fields=["variety"],
        draft=followup_draft,
        query=prompt,
        options=options,
        choice_hint=True,
        strict_options_only=True,
        source="candidate",
        extra={"candidates": options},
    )


def _fetch_variety_records(variety_name: str) -> list[dict[str, object]]:
    if not variety_name:
        return []
    url = _require_db_url()
    table = resolve_db_table(_cfg(), TABLE_KEY_VARIETY)
    try:
        sql = (
            f"SELECT "
            f"{_qid(VARIETY_PG_NAME_COLUMN)} AS name, "
            f"{_qid(VARIETY_PG_COLUMN_MAP['subspecies_type'])} AS sub_type, "
            f"{_qid(VARIETY_PG_COLUMN_MAP['rice_type'])} AS culti_type, "
            f"{_qid(VARIETY_PG_COLUMN_MAP['approval_region'])} AS approve_region, "
            f"{_qid(VARIETY_PG_COLUMN_MAP['suitable_region'])} AS suitable_region, "
            f"{_qid(VARIETY_PG_COLUMN_MAP['approval_year'])} AS approve_year "
            f"FROM {_qid(table)} "
            f"WHERE {_qid(VARIETY_PG_NAME_COLUMN)} = %s"
        )
        rows = _fetch_all(url, sql, (variety_name,))
    except Exception:
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _parse_int_like(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return None


def _parse_approve_year(value: object) -> int:
    text = str(value or "").strip()
    if text.isdigit() and len(text) == 4:
        return int(text)
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else 0


def _resolve_variety_record_by_region(
    records: list[dict[str, object]],
    region_text: str,
    resolved_region_id: object = None,
) -> Optional[dict[str, object]]:
    if not records:
        return None
    variants = build_region_text_variants(region_text)
    province = ""
    region_id_text = str(resolved_region_id or "").strip()
    region_id_match = re.match(r"^(\d{2})\d{4,10}$", region_id_text)
    if region_id_match:
        province = _PROVINCE_CODE_MAP.get(region_id_match.group(1), "")
    if province:
        variants.extend(build_region_text_variants(province))
    if not variants:
        return None
    best_record: Optional[dict[str, object]] = None
    best_score = -1
    for record in records:
        approval_region = normalize_region_token(record.get("approve_region"))
        suitable_region = normalize_region_token(record.get("suitable_region"))
        if not approval_region:
            approval_region = ""
        if not suitable_region:
            suitable_region = ""
        score = 0
        for variant in variants:
            normalized = normalize_region_token(variant)
            if not normalized:
                continue
            for candidate_region, weight_exact, weight_contains, weight_reverse in (
                (approval_region, 120, 100, 90),
                (suitable_region, 110, 95, 85),
            ):
                if not candidate_region:
                    continue
                if normalized == candidate_region:
                    score = max(score, weight_exact)
                elif normalized in candidate_region:
                    score = max(score, weight_contains)
                elif candidate_region in normalized:
                    score = max(score, weight_reverse)
        if score <= 0:
            continue
        if (
            score > best_score
            or (
                score == best_score
                and _parse_approve_year(record.get("approve_year"))
                > _parse_approve_year(best_record.get("approve_year") if best_record else None)
            )
        ):
            best_record = record
            best_score = score
    return best_record


def _resolve_variety_metadata(
    variety_name: str,
    region_text: str = "",
    resolved_region_id: object = None,
) -> tuple[Optional[dict[str, object]], bool]:
    records = _fetch_variety_records(variety_name)
    if not records:
        return None, False
    if region_text.strip():
        matched = _resolve_variety_record_by_region(records, region_text, resolved_region_id)
        return matched, True
    best_record = max(
        records,
        key=lambda item: _parse_approve_year(item.get("approve_year")),
    )
    return best_record, False


def _fetch_variety_sub_type(variety_name: str) -> Optional[int]:
    record, _ = _resolve_variety_metadata(variety_name)
    if not record:
        return None
    raw = record.get("sub_type")
    if raw is None:
        return None
    parsed = _parse_int_like(raw)
    if parsed is not None:
        return parsed
    text = str(raw).strip()
    if not text:
        return None
    return _resolve_code("sub_type", text)


def _normalize_crop_code(value: object) -> int:
    text = str(value or "").strip()
    if not text or text == DEFAULT_CROP:
        return 0
    return 0


def _build_request_payload(query: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    draft = dict(query)
    region_raw = str(draft.get("region_id") or "").strip()
    variety = str(draft.get("variety") or "").strip()
    resolved_region_id = _resolve_region_id_for_payload(region_raw)
    variety_record, matched_by_region = _resolve_variety_metadata(
        variety,
        region_raw,
        resolved_region_id,
    )
    if variety_record:
        if not str(draft.get("culti_type") or "").strip():
            variety_culti_type = variety_record.get("culti_type")
            if variety_culti_type not in (None, ""):
                draft["culti_type"] = variety_culti_type
    elif variety and region_raw and matched_by_region:
        raise RuntimeError(f"品种 {variety} 未在 {region_raw} 审定。")
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
    sub_type = None
    if variety_record:
        sub_type = _parse_int_like(variety_record.get("sub_type"))
        if sub_type is None:
            raw_sub_type = str(variety_record.get("sub_type") or "").strip()
            if raw_sub_type:
                sub_type = _resolve_code("sub_type", raw_sub_type)
    if sub_type is None:
        sub_type = _fetch_variety_sub_type(variety)
    if sub_type is None:
        raise RuntimeError(f"未找到品种亚种类型: {draft.get('variety')}")
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
        "culti_type": resolve_culti_type_label(draft.get("culti_type"))
        or draft.get("culti_type"),
        "planting_method": draft.get("planting_method"),
        "region_id": region_raw or draft.get("region_id"),
        "farm_id": farm_id,
        "crop": draft.get("crop") or DEFAULT_CROP,
        "sub_type": sub_type,
    }


def lookup_sowing_suitability(prompt: str) -> ToolInvocation:
    text = str(prompt or "")
    draft = _build_query_from_prompt(text)
    raw_variety = str(draft.get("variety") or "").strip()
    raw_region = str(draft.get("region_id") or "").strip()
    resolved_region_id = _resolve_region_id_for_payload(raw_region)
    variety_record, matched_by_region = _resolve_variety_metadata(
        raw_variety,
        raw_region,
        resolved_region_id,
    )
    if raw_variety and variety_record:
        if not str(draft.get("culti_type") or "").strip():
            variety_culti_type = variety_record.get("culti_type")
            if variety_culti_type not in (None, ""):
                draft["culti_type"] = variety_culti_type
    elif raw_variety and raw_region and matched_by_region:
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message=f"品种 {raw_variety} 未在 {raw_region} 审定。",
            data={"draft": draft},
        )
    if raw_variety and _fetch_variety_sub_type(raw_variety) is None and not variety_record:
        query_text = _extract_query_text(text)
        exact_variety = find_exact_variety_in_text(raw_variety) or find_exact_variety_in_text(
            query_text
        )
        if exact_variety:
            draft["variety"] = exact_variety
        else:
            candidates = retrieve_variety_candidates(raw_variety, limit=5)
            if not candidates and query_text and query_text != raw_variety:
                candidates = retrieve_variety_candidates(query_text, limit=5)
            if candidates:
                return _build_variety_candidate_followup(
                    text, draft=draft, candidates=candidates
                )
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
    request_body = dict(request_payload)
    log_event(
        "sowing_suitability_api_request",
        url=url,
        payload=request_body,
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
    except Exception as exc:
        resp = getattr(exc, "response", None)
        if resp is not None:
            log_event(
                "sowing_suitability_api_http_error",
                url=url,
                payload=request_body,
                status_code=getattr(resp, "status_code", None),
                response_text=summarize_text(getattr(resp, "text", str(exc)), limit=1200),
            )
        else:
            log_event(
                "sowing_suitability_api_request_error",
                url=url,
                payload=request_body,
                error=str(exc),
            )
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message=f"查询播期推荐失败: {exc}",
            data={"request": request_payload, "resolved": resolved},
        )
    try:
        payload = response.json()
    except Exception:
        log_event(
            "sowing_suitability_api_parse_error",
            url=url,
            payload=request_body,
            status_code=response.status_code,
            response_text=summarize_text(response.text or "", limit=1200),
        )
        return ToolInvocation(
            name="sowing_suitability_lookup",
            message="播期推荐接口返回格式未识别。",
            data={"request": request_payload, "resolved": resolved},
        )
    log_event(
        "sowing_suitability_api_response",
        url=url,
        payload=request_body,
        status_code=response.status_code,
        response_summary=summarize_text(
            json.dumps(payload, ensure_ascii=False, default=str), limit=1200
        ),
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
        log_event(
            "sowing_suitability_api_business_error",
            url=url,
            payload=request_body,
            code=code,
            msg=message or None,
            response_summary=summarize_text(
                json.dumps(payload, ensure_ascii=False, default=str), limit=1200
            ),
        )
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
