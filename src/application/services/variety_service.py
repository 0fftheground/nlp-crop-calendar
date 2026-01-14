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
from ...infra.variety_store import retrieve_variety_candidates
from ...observability.logging_utils import log_event
from ...prompts.variety_match import VARIETY_MATCH_SYSTEM_PROMPT
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
VARIETY_QUERY_STOPWORDS = {
    "水稻",
    "小麦",
    "玉米",
    "大豆",
    "油菜",
    "棉花",
    "花生",
    "品种",
    "品系",
    "抗性",
    "特性",
    "生育期",
    "熟期",
    "审定",
    "信息",
    "查询",
    "查",
    "帮我",
    "请",
    "一下",
    "是什么",
    "多长",
    "多少",
    "的",
    "和",
    "与",
    "及",
}
VARIETY_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]{2,}")


class VarietyMatchDecision(BaseModel):
    index: int
    reason: Optional[str] = None


def _extract_variety(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    candidates = retrieve_variety_candidates(prompt, limit=1)
    return candidates[0] if candidates else None


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
    try:
        payload = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt
    if not isinstance(payload, dict):
        return prompt
    candidates = []
    for key in ("variety", "query", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
    followup = payload.get("followup")
    if isinstance(followup, dict):
        value = followup.get("prompt")
        if isinstance(value, str) and value.strip():
            candidates.append(value.strip())
        draft = followup.get("draft")
        if isinstance(draft, dict):
            for key in ("variety", "crop"):
                value = draft.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())
    planting = payload.get("planting")
    if isinstance(planting, dict):
        for key in ("variety", "crop"):
            value = planting.get(key)
            if isinstance(value, str) and value.strip():
                candidates.append(value.strip())
    return " ".join(candidates) if candidates else prompt


def _extract_variety_tokens(prompt: str) -> List[str]:
    if not prompt:
        return []
    text = prompt
    for word in VARIETY_QUERY_STOPWORDS:
        text = text.replace(word, " ")
    candidates = VARIETY_TOKEN_RE.findall(text)
    tokens: List[str] = []
    seen = set()
    for token in candidates:
        token = token.strip()
        if not token or token in VARIETY_QUERY_STOPWORDS:
            continue
        if token.isdigit():
            continue
        if len(token) > 20:
            continue
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    tokens.sort(key=len, reverse=True)
    return tokens


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
    prompt: str, *, limit: int = 5
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    path = _get_variety_db_path()
    if not path or not path.exists():
        return [], []
    prompt_text = prompt or ""
    normalized_prompt = _normalize_variety_prompt(prompt_text) or prompt_text
    variety_name = _extract_variety(normalized_prompt)
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            rows: List[sqlite3.Row] = []
            if variety_name:
                rows = _query_variety_db_by_name(conn, variety_name, limit)
            if not rows and normalized_prompt:
                rows = _query_variety_db_by_prompt(conn, normalized_prompt, limit)
            if not rows and normalized_prompt:
                tokens = _extract_variety_tokens(normalized_prompt)
                rows = _query_variety_db_by_fuzzy_tokens(conn, tokens, limit)
    except Exception:
        return [], []
    return _rows_to_variety_records(rows), _rows_to_variety_raw_records(rows)


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
    records, raw_records = _lookup_variety_records(prompt)
    if records:
        selected, reason, selected_index = _select_best_variety_record(
            prompt, records
        )
        variety = selected.get("品种名称") or _extract_variety(prompt) or "未知"
        approval_regions = sorted(
            {r.get("审定区域") for r in records if r.get("审定区域")}
        )
        region_note = f"（{len(approval_regions)}个区域）" if approval_regions else ""
        raw_selected = (
            raw_records[selected_index]
            if 0 <= selected_index < len(raw_records)
            else None
        )
        payload = {
            "query": prompt,
            "crop": DEFAULT_CROP,
            "variety": variety,
            "selected": selected,
            "raw_selected": raw_selected,
            "selection_reason": reason,
            "matches": records,
            "raw_matches": raw_records,
            "source": "sqlite",
        }
        message = f"已返回品种 {variety} 的审定信息{region_note}。"
        return ToolInvocation(
            name="variety_lookup",
            message=message,
            data=payload,
        )
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
