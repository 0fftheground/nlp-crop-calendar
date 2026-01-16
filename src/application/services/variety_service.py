from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel

from ...domain.planting import DEFAULT_CROP
from ...infra.config import get_config
from ...infra.llm import get_chat_model
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...infra.variety_choice_store import VarietyChoice, get_variety_choice_store
from ...infra.variety_store import extract_variety_tokens, retrieve_variety_candidates
from ...observability.logging_utils import log_event
from ...prompts.variety_match import (
    VARIETY_MATCH_SYSTEM_PROMPT,
    VARIETY_NAME_PICKER_SYSTEM_PROMPT,
)
from ...schemas.models import ToolInvocation


VARIETY_DB_TABLE = "variety_approvals"
VARIETY_DB_FIELD_LABELS = {
    "variety_name": "品种名称",
    "approval_year": "审定年份",
    "approval_region": "审定区域",
    "suitable_region": "适种地区",
    "rice_type": "稻作类型",
    "subspecies_type": "亚种类型",
    "maturity": "熟期",
    "control_variety": "对照品种",
    "days_vs_control": "比对照长(天)",
}
_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_FOLLOWUP_INDEX_RE = re.compile(r"^第?\s*(\d+)\s*(?:个|条|项)?$")
_FOLLOWUP_CN_MAP = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_FOLLOWUP_ALL_TOKENS = {
    "全部",
    "全都",
    "所有",
    "都要",
    "都要看",
    "全要",
    "全部区域",
    "全部信息",
}
_CHOICE_CANCEL_TOKENS = {
    "更换",
    "重新选择",
    "换一个",
    "换个",
    "取消默认",
    "不要这个",
}


class VarietyMatchDecision(BaseModel):
    index: int
    reason: Optional[str] = None


class VarietyNameDecision(BaseModel):
    index: int
    reason: Optional[str] = None
    confidence: Optional[float] = None


def _extract_user_id(prompt: str) -> Optional[str]:
    payload = _load_prompt_payload(prompt.strip())
    if not isinstance(payload, dict):
        return None
    for key in ("user_id", "userId"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_query_source(prompt: str) -> str:
    payload = _load_prompt_payload(prompt.strip())
    if isinstance(payload, dict):
        for key in ("query", "prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        followup = payload.get("followup")
        if isinstance(followup, dict):
            draft = followup.get("draft")
            if isinstance(draft, dict):
                for key in ("query", "prompt"):
                    value = draft.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
    return prompt


def _extract_followup_answer(prompt: str) -> str:
    payload = _load_prompt_payload(prompt.strip())
    if isinstance(payload, dict):
        followup = payload.get("followup")
        if isinstance(followup, dict):
            answer = followup.get("prompt")
            if isinstance(answer, str) and answer.strip():
                return answer.strip()
    return prompt.strip()


def _make_choice_key(text: str) -> str:
    normalized = _decode_unicode_escapes(text or "").strip()
    tokens = extract_variety_tokens(normalized)
    if tokens:
        return " ".join(tokens)
    return normalized


def _get_choice_from_store(prompt: str) -> Optional[VarietyChoice]:
    user_id = _extract_user_id(prompt)
    if not user_id:
        return None
    query_source = _extract_query_source(prompt)
    if not query_source:
        return None
    key = _make_choice_key(query_source)
    if not key:
        return None
    try:
        store = get_variety_choice_store()
        return store.get(user_id, key)
    except Exception:
        return None


def _store_choice(
    prompt: str, variety: str, region_choice: Optional[str]
) -> None:
    user_id = _extract_user_id(prompt)
    if not user_id:
        return None
    query_source = _extract_query_source(prompt)
    if not query_source:
        return None
    key = _make_choice_key(query_source)
    if not key or not variety:
        return None
    try:
        store = get_variety_choice_store()
        store.set(user_id, key, variety, region_choice)
    except Exception:
        return None


def _clear_choice(prompt: str) -> None:
    user_id = _extract_user_id(prompt)
    if not user_id:
        return None
    query_source = _extract_query_source(prompt)
    if not query_source:
        return None
    key = _make_choice_key(query_source)
    if not key:
        return None
    try:
        store = get_variety_choice_store()
        store.delete(user_id, key)
    except Exception:
        return None


def _is_cancel_choice(prompt: str) -> bool:
    answer = _extract_followup_answer(prompt)
    if not answer:
        return False
    for token in _CHOICE_CANCEL_TOKENS:
        if answer == token or answer.startswith(token):
            return True
    return False


def _llm_choose_variety_name(
    prompt: str, candidates: List[str]
) -> Optional[VarietyNameDecision]:
    if not prompt or not candidates:
        return None
    try:
        llm = get_chat_model()
    except Exception:
        return None
    payload = {"prompt": prompt, "candidates": candidates}
    try:
        chooser = llm.with_structured_output(VarietyNameDecision)
        result = chooser.invoke(
            [
                ("system", VARIETY_NAME_PICKER_SYSTEM_PROMPT),
                ("human", json.dumps(payload, ensure_ascii=True, default=str)),
            ]
        )
        decision = (
            result
            if isinstance(result, VarietyNameDecision)
            else VarietyNameDecision.model_validate(result)
        )
    except Exception:
        return None
    if decision.index < 0 or decision.index >= len(candidates):
        return None
    return decision


def _extract_variety(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    normalized_prompt = _normalize_variety_prompt(prompt) or prompt
    candidates = retrieve_variety_candidates(
        normalized_prompt, limit=10, threshold=0.5, semantic=True
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    decision = _llm_choose_variety_name(normalized_prompt, candidates)
    if decision:
        selected = candidates[decision.index]
        log_event(
            "variety_name_llm_choice",
            selected=selected,
            reason=decision.reason,
            confidence=decision.confidence,
        )
        return selected
    return candidates[0]


def _infer_crop_and_variety(prompt: str) -> Tuple[str, str]:
    crop_keywords = ["水稻", "小麦", "玉米", "大豆", "油菜", "棉花", "花生"]
    crop = next((item for item in crop_keywords if item in prompt), DEFAULT_CROP)
    variety = _extract_variety(prompt)
    if not variety:
        variety = "美香占2号" if crop == "水稻" else f"{crop}示例品种"
    return crop, variety


def _get_variety_db_path() -> Optional[Path]:
    cfg = get_config()
    if cfg.variety_db_path:
        return Path(cfg.variety_db_path)
    return Path(__file__).resolve().parents[3] / "resources" / "rice_variety_approvals.sqlite3"


def _normalize_variety_prompt(prompt: str) -> str:
    if not prompt:
        return ""
    text = prompt.strip()
    payload = _load_prompt_payload(text)
    if not isinstance(payload, dict):
        return _decode_unicode_escapes(text)
    candidates = []
    for key in ("variety", "query", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    followup = payload.get("followup")
    if isinstance(followup, dict):
        selected = _resolve_followup_candidate(payload)
        if selected:
            candidates.append(selected)
        else:
            value = followup.get("prompt")
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
        draft = followup.get("draft")
        if isinstance(draft, dict):
            for key in ("variety", "crop", "query", "prompt"):
                value = draft.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
    planting = payload.get("planting")
    if isinstance(planting, dict):
        for key in ("variety", "crop"):
            value = planting.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    if candidates:
        return " ".join(candidates)
    return _decode_unicode_escapes(text)


def _decode_unicode_escapes(text: str) -> str:
    if "\\u" not in text:
        return text

    def repl(match: re.Match[str]) -> str:
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)

    return _UNICODE_ESCAPE_RE.sub(repl, text)


def _load_prompt_payload(prompt: str) -> Optional[dict]:
    candidate = prompt
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


def _parse_followup_index(text: str) -> Optional[int]:
    if not text:
        return None
    value = text.strip()
    match = _FOLLOWUP_INDEX_RE.match(value)
    if match:
        return int(match.group(1))
    cn_match = re.match(r"^第?\s*([一二三四五六七八九十])", value)
    if cn_match:
        return _FOLLOWUP_CN_MAP.get(cn_match.group(1))
    return None


def _resolve_followup_candidate(payload: dict) -> Optional[str]:
    followup = payload.get("followup")
    if not isinstance(followup, dict):
        return None
    answer = followup.get("prompt")
    if not isinstance(answer, str):
        return None
    draft = followup.get("draft")
    if not isinstance(draft, dict):
        return None
    candidates = draft.get("candidates") or draft.get("variety_candidates")
    if not isinstance(candidates, list):
        return None
    answer = answer.strip()
    if not answer:
        return None
    index = _parse_followup_index(answer)
    if index is not None and 1 <= index <= len(candidates):
        chosen = candidates[index - 1]
        return str(chosen).strip() if chosen else None
    for cand in candidates:
        if isinstance(cand, str) and cand == answer:
            return cand
    for cand in candidates:
        if not isinstance(cand, str):
            continue
        if answer in cand or cand in answer:
            return cand
    return None


def _resolve_followup_region(
    payload: dict, region_candidates: List[str]
) -> Optional[str]:
    followup = payload.get("followup")
    if not isinstance(followup, dict):
        return None
    answer = followup.get("prompt")
    if not isinstance(answer, str):
        return None
    answer = answer.strip()
    if not answer:
        return None
    if answer in _FOLLOWUP_ALL_TOKENS:
        return "__all__"
    index = _parse_followup_index(answer)
    if index is not None and 1 <= index <= len(region_candidates):
        return region_candidates[index - 1]
    for candidate in region_candidates:
        if candidate == answer:
            return candidate
    for candidate in region_candidates:
        if answer in candidate or candidate in answer:
            return candidate
    return None


def _extract_region_candidates(records: List[Dict[str, object]]) -> List[str]:
    regions: List[str] = []
    seen = set()
    for record in records:
        value = str(record.get("审定区域") or "")
        if not value:
            continue
        for token in re.split(r"[，,、/\\s]+", value):
            token = token.strip()
            if token and token not in seen:
                seen.add(token)
                regions.append(token)
    return regions


def _build_region_followup(
    prompt: str,
    *,
    variety: str,
    region_candidates: List[str],
) -> ToolInvocation:
    options = "\n".join(
        f"{idx + 1}. {name}" for idx, name in enumerate(region_candidates)
    )
    message = (
        f"品种 {variety} 在多个审定区域有记录，请选择要查看的区域：\n"
        f"{options}\n"
        "回复序号/区域名称，或回复“全部”查看所有区域。"
    )
    return ToolInvocation(
        name="variety_lookup",
        message=message,
        data={
            "query": prompt,
            "variety": variety,
            "region_candidates": region_candidates,
            "missing_fields": ["approval_region"],
            "draft": {
                "variety": variety,
                "region_candidates": region_candidates,
                "query": prompt,
            },
            "followup_count": 0,
            "source": "candidate",
        },
    )


def _filter_records_by_region(
    records: List[Dict[str, object]], region: str
) -> List[Dict[str, object]]:
    if not region:
        return records
    filtered: List[Dict[str, object]] = []
    for record in records:
        value = (
            record.get("审定区域")
            or record.get("approval_region")
            or ""
        )
        if region in str(value):
            filtered.append(record)
    return filtered


def _query_variety_db_by_name(
    conn: sqlite3.Connection, name: str, limit: int
) -> List[sqlite3.Row]:
    rows = conn.execute(
        f"SELECT * FROM {VARIETY_DB_TABLE} WHERE variety_name = ?",
        (name,),
    ).fetchall()
    if rows:
        return rows
    like_prefix = f"{name}%"
    rows = conn.execute(
        f"SELECT * FROM {VARIETY_DB_TABLE} WHERE variety_name LIKE ? LIMIT ?",
        (like_prefix, limit),
    ).fetchall()
    if rows:
        return rows
    like_any = f"%{name}%"
    return conn.execute(
        f"SELECT * FROM {VARIETY_DB_TABLE} WHERE variety_name LIKE ? LIMIT ?",
        (like_any, limit),
    ).fetchall()


def _query_variety_db_by_prompt(
    conn: sqlite3.Connection, prompt: str, limit: int
) -> List[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM {VARIETY_DB_TABLE} "
        "WHERE ? LIKE '%' || variety_name || '%' "
        "ORDER BY LENGTH(variety_name) DESC LIMIT ?",
        (prompt, limit),
    ).fetchall()


def _query_variety_db_by_fuzzy_tokens(
    conn: sqlite3.Connection, tokens: List[str], limit: int
) -> List[sqlite3.Row]:
    if not tokens:
        return []
    for token in tokens:
        rows = conn.execute(
            f"SELECT * FROM {VARIETY_DB_TABLE} WHERE variety_name LIKE ? LIMIT ?",
            (f"%{token}%", limit),
        ).fetchall()
        if rows:
            return rows
    return []


def _rows_to_variety_records(rows: List[sqlite3.Row]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in rows:
        record: Dict[str, object] = {}
        for field in row.keys():
            label = VARIETY_DB_FIELD_LABELS.get(field, field)
            record[label] = row[field]
        records.append(record)
    return records


def _rows_to_variety_raw_records(
    rows: List[sqlite3.Row],
) -> List[Dict[str, object]]:
    return [dict(row) for row in rows]


def _lookup_variety_records(
    prompt: str,
    *,
    limit: int = 5,
    confirmed_candidate: Optional[str] = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    path = _get_variety_db_path()
    if not path or not path.exists():
        return [], []
    prompt_text = prompt or ""
    normalized_prompt = _normalize_variety_prompt(prompt_text) or prompt_text
    if confirmed_candidate is None:
        confirmed_candidate = _extract_confirmed_candidate(prompt_text)
    variety_name = confirmed_candidate or _extract_variety(normalized_prompt)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows: List[sqlite3.Row] = []
            if variety_name:
                rows = _query_variety_db_by_name(conn, variety_name, limit)
            if not rows and normalized_prompt:
                rows = _query_variety_db_by_prompt(conn, normalized_prompt, limit)
            if not rows and normalized_prompt:
                tokens = extract_variety_tokens(normalized_prompt)
                rows = _query_variety_db_by_fuzzy_tokens(conn, tokens, limit)
    except Exception:
        return [], []
    return _rows_to_variety_records(rows), _rows_to_variety_raw_records(rows)


def _extract_confirmed_candidate(prompt: str) -> Optional[str]:
    payload = _load_prompt_payload(prompt.strip())
    if not isinstance(payload, dict):
        return None
    return _resolve_followup_candidate(payload)


def _is_region_followup(payload: Optional[dict]) -> bool:
    if not isinstance(payload, dict):
        return False
    followup = payload.get("followup")
    if not isinstance(followup, dict):
        return False
    draft = followup.get("draft")
    if not isinstance(draft, dict):
        return False
    return bool(draft.get("region_candidates"))


def _is_exact_variety_match(variety: str, prompt: str) -> bool:
    if not variety or not prompt:
        return False
    normalized_prompt = _normalize_variety_prompt(prompt) or prompt
    return variety in normalized_prompt


def _format_variety_record(record: Dict[str, object]) -> str:
    lines: List[str] = []
    for key, value in record.items():
        if value is None or value == "":
            continue
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _escape_table_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ").strip()


def _format_variety_records_table(
    records: List[Dict[str, object]]
) -> str:
    if not records:
        return ""
    columns = list(VARIETY_DB_FIELD_LABELS.values())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for record in records:
        row = [
            _escape_table_cell(record.get(col, ""))
            for col in columns
        ]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join([header, separator, *rows])


def _normalize_region_token(value: str) -> str:
    return re.sub(r"(省|市|州|盟|地区|区|县)$", "", value or "").strip()


def _extract_region_tokens(
    prompt: str, records: List[Dict[str, object]]
) -> List[str]:
    tokens: List[str] = []
    if prompt:
        tokens.extend(
            re.findall(r"[\u4e00-\u9fff]{2,8}(?:省|市|州|盟|地区|区|县)", prompt)
        )
    approval_regions = {
        str(record.get("审定区域") or "").strip()
        for record in records
        if record.get("审定区域")
    }
    if prompt:
        for region in approval_regions:
            if region and region in prompt:
                tokens.append(region)
    unique: List[str] = []
    seen = set()
    for token in tokens:
        if token and token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


def _parse_approval_year(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() and len(text) == 4:
        return int(text)
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else None


def _score_record_by_region(
    record: Dict[str, object], region_tokens: List[str]
) -> int:
    approval_region = str(record.get("审定区域") or "")
    suitable_region = str(record.get("适种地区") or "")
    best = 0
    for token in region_tokens:
        normalized = _normalize_region_token(token)
        if token and token in suitable_region:
            best = max(best, 100)
        elif normalized and normalized in suitable_region:
            best = max(best, 90)
        if token and token in approval_region:
            best = max(best, 80)
        elif normalized and normalized in approval_region:
            best = max(best, 70)
    return best


def _pick_latest_year_record(
    records: List[Dict[str, object]], indices: Optional[List[int]] = None
) -> int:
    best_index = (indices[0] if indices else 0)
    best_year = -1
    for idx, record in enumerate(records):
        if indices and idx not in indices:
            continue
        year = _parse_approval_year(record.get("审定年份")) or 0
        if year > best_year:
            best_year = year
            best_index = idx
            continue
        if year == best_year:
            region = str(record.get("审定区域") or "")
            current = str(records[best_index].get("审定区域") or "")
            if region == "国审" and current != "国审":
                best_index = idx
    return best_index


def _llm_choose_variety_record(
    prompt: str,
    candidates: List[Dict[str, object]],
    region_tokens: List[str],
) -> Optional[VarietyMatchDecision]:
    try:
        llm = get_chat_model()
    except Exception:
        return None
    system_prompt = VARIETY_MATCH_SYSTEM_PROMPT
    payload = {
        "prompt": prompt,
        "region_tokens": region_tokens,
        "candidates": candidates,
    }
    try:
        chooser = llm.with_structured_output(VarietyMatchDecision)
        result = chooser.invoke(
            [
                ("system", system_prompt),
                ("human", json.dumps(payload, ensure_ascii=True, default=str)),
            ]
        )
        decision = (
            result
            if isinstance(result, VarietyMatchDecision)
            else VarietyMatchDecision.model_validate(result)
        )
    except Exception:
        return None
    if decision.index < 0 or decision.index >= len(candidates):
        return None
    return decision


def _select_best_variety_record(
    prompt: str, records: List[Dict[str, object]]
) -> Tuple[Dict[str, object], str, int]:
    region_tokens = _extract_region_tokens(prompt, records)
    if region_tokens:
        scored = [
            (_score_record_by_region(record, region_tokens), idx)
            for idx, record in enumerate(records)
        ]
        max_score = max(score for score, _ in scored)
        best_indices = [idx for score, idx in scored if score == max_score]
        if max_score > 0 and len(best_indices) == 1:
            idx = best_indices[0]
            return records[idx], "规则匹配", idx
        if max_score > 0 and len(best_indices) > 1:
            candidates = [
                {
                    "index": i,
                    "审定区域": records[i].get("审定区域"),
                    "适种地区": records[i].get("适种地区"),
                    "审定年份": records[i].get("审定年份"),
                    "稻作类型": records[i].get("稻作类型"),
                    "亚种类型": records[i].get("亚种类型"),
                    "熟期": records[i].get("熟期"),
                    "对照品种": records[i].get("对照品种"),
                }
                for i in best_indices
            ]
            decision = _llm_choose_variety_record(
                prompt, candidates, region_tokens
            )
            if decision:
                selected = records[decision.index]
                reason = decision.reason or "LLM 匹配"
                log_event(
                    "variety_match_llm_choice",
                    selected_index=decision.index,
                    reason=reason,
                )
                return selected, reason, decision.index
            fallback = _pick_latest_year_record(records, best_indices)
            return records[fallback], "年份优先", fallback
    fallback = _pick_latest_year_record(records)
    return records[fallback], "年份优先", fallback


def lookup_variety(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.variety_provider)
    intranet = maybe_intranet_tool(
        "variety_lookup",
        prompt,
        provider,
        cfg.variety_api_url,
        cfg.variety_api_key,
    )
    if intranet:
        return intranet
    query_source = _extract_query_source(prompt)
    cancel_choice = _is_cancel_choice(prompt)
    if cancel_choice:
        _clear_choice(prompt)
        followup = _build_variety_followup(prompt)
        if followup:
            return followup
    stored_choice = None if cancel_choice else _get_choice_from_store(prompt)
    confirmed_candidate = _extract_confirmed_candidate(prompt)
    used_stored_choice = False
    if confirmed_candidate is None and stored_choice:
        confirmed_candidate = stored_choice.variety
        used_stored_choice = True
    records, raw_records = _lookup_variety_records(
        prompt, confirmed_candidate=confirmed_candidate
    )
    if records:
        payload_data = _load_prompt_payload(prompt.strip())
        is_region_followup = _is_region_followup(payload_data)
        variety = (
            records[0].get("品种名称")
            or _extract_variety(prompt)
            or "未知"
        )
        if (
            confirmed_candidate is None
            and not is_region_followup
            and not _is_exact_variety_match(variety, prompt)
        ):
            followup = _build_variety_followup(prompt)
            if followup:
                return followup
        region_candidates = _extract_region_candidates(records)
        region_choice = None
        region_confirmed = False
        if is_region_followup and isinstance(payload_data, dict):
            region_choice = _resolve_followup_region(
                payload_data, region_candidates
            )
            region_confirmed = region_choice is not None
        if region_choice is None and stored_choice:
            stored_region = stored_choice.region_choice
            if stored_region == "__all__":
                region_choice = stored_region
                region_confirmed = True
            elif stored_region in region_candidates:
                region_choice = stored_region
                region_confirmed = True
        region_tokens = _extract_region_tokens(prompt, records)
        if (
            region_choice is None
            and not region_tokens
            and len(region_candidates) > 1
        ):
            return _build_region_followup(
                prompt, variety=variety, region_candidates=region_candidates
            )
        selected_records = records
        selected_raw_records = raw_records
        selection_prompt = prompt
        if region_choice and region_choice != "__all__":
            filtered = _filter_records_by_region(records, region_choice)
            filtered_raw = _filter_records_by_region(
                raw_records, region_choice
            )
            if filtered:
                selected_records = filtered
                selected_raw_records = filtered_raw
                if region_choice not in selection_prompt:
                    selection_prompt = (
                        f"{selection_prompt} {region_choice}"
                    )
            else:
                region_choice = None
        selected, reason, selected_index = _select_best_variety_record(
            selection_prompt, selected_records
        )
        variety = selected.get("品种名称") or _extract_variety(prompt) or "未知"
        if (
            confirmed_candidate is None
            and not _is_exact_variety_match(variety, prompt)
        ):
            followup = _build_variety_followup(prompt)
            if followup:
                return followup
        approval_regions = sorted(
            {
                r.get("审定区域")
                for r in selected_records
                if r.get("审定区域")
            }
        )
        region_note = f"（{len(approval_regions)}个区域）" if approval_regions else ""
        raw_selected = (
            selected_raw_records[selected_index]
            if 0 <= selected_index < len(selected_raw_records)
            else None
        )
        if region_choice == "__all__":
            detail = _format_variety_records_table(selected_records)
        else:
            detail = _format_variety_record(selected)
        payload = {
            "query": query_source,
            "crop": DEFAULT_CROP,
            "variety": variety,
            "selected": selected,
            "raw_selected": raw_selected,
            "selection_reason": reason,
            "matches": selected_records,
            "raw_matches": selected_raw_records,
            "region_candidates": region_candidates,
            "region_choice": region_choice,
            "source": "sqlite",
        }
        if detail and region_choice == "__all__":
            message = (
                f"已返回品种 {variety} 的全部审定信息"
                f"{region_note}。\n{detail}"
            )
        elif detail:
            message = (
                f"已返回品种 {variety} 的审定信息{region_note}。\n{detail}"
            )
        else:
            message = f"已返回品种 {variety} 的审定信息{region_note}。"
        if used_stored_choice:
            message = (
                f"{message}\n已默认使用上次选择：{variety}。"
                "如需更换，请回复“更换”。"
            )
            payload["choice_hint"] = True
            payload["options"] = ["更换", "重新选择"]
        if variety and variety != "未知":
            _store_choice(
                prompt,
                variety,
                region_choice if region_confirmed else None,
            )
        return ToolInvocation(
            name="variety_lookup",
            message=message,
            data=payload,
        )
    followup = _build_variety_followup(prompt)
    if followup:
        return followup
    crop, variety = _infer_crop_and_variety(prompt)
    payload = {
        "query": prompt,
        "crop": crop,
        "variety": variety,
        "growth_duration_days": 120,
        "traits": {
            "yield_level": "中高产",
            "lodging_resistance": "中等",
            "disease_resistance": ["稻瘟病", "纹枯病"],
        },
        "source": "mock",
    }
    return ToolInvocation(
        name="variety_lookup",
        message=f"已返回 {crop} 品种 {variety} 的模拟信息。",
        data=payload,
    )


def _build_variety_followup(prompt: str) -> Optional[ToolInvocation]:
    normalized_prompt = _normalize_variety_prompt(prompt) or prompt
    candidates = retrieve_variety_candidates(
        normalized_prompt, limit=5, threshold=0.5, semantic=True
    )
    if not candidates:
        return None
    options = "\n".join(
        f"{idx + 1}. {name}" for idx, name in enumerate(candidates)
    )
    message = (
        "未找到完全匹配的品种。你是不是想查询以下品种：\n"
        f"{options}\n"
        "请回复序号或品种名称。"
    )
    return ToolInvocation(
        name="variety_lookup",
        message=message,
        data={
            "query": normalized_prompt,
            "candidates": candidates,
            "missing_fields": ["variety"],
            "draft": {"candidates": candidates},
            "followup_count": 0,
            "source": "candidate",
        },
    )
