from __future__ import annotations

import csv
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from ..infra.config import get_config
from ..infra.llm import get_audit_judge_model
from ..infra.postgres import fetch_all
from ..prompts.audit_judge import PRODUCTION_AUDIT_JUDGE_SYSTEM_PROMPT
from .audit_models import (
    AuditJudgeDecision,
    AuditReviewRecord,
    HumanReviewRecord,
    ProductionAuditBatch,
)
from .graders import grade_case
from .runners import TASK_RUNNERS

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_LONG_DIGIT_RE = re.compile(r"\b\d{8,}\b")
_BRIEF_FOLLOWUP_SUFFIX_RE = re.compile(r"(呢|吗|呀|啊|怎么样|如何|行吗|可以吗)$")
_REGION_ONLY_RE = re.compile(r"^(?:那|那就|那改成|改成|换成)?([\u4e00-\u9fff]{2,20})(?:呢|吗|呀|啊)?$")
_STANDALONE_QUERY_TOKENS = (
    "天气",
    "气象",
    "播种",
    "播期",
    "什么时候",
    "何时",
    "适合",
    "审定",
    "品种",
    "计划",
    "方案",
    "生育期",
    "查询",
    "查",
    "帮我",
    "帮忙",
    "怎么",
    "如何",
)
_CONTEXT_WINDOW_LIMIT = 5
_REVIEW_CSV_FIELDS = [
    "source_review_file",
    "case_id",
    "task",
    "gate",
    "prompt",
    "context_summary",
    "expected_json",
    "observed_output_json",
    "ai_verdict",
    "ai_risk",
    "ai_confidence",
    "ai_rationale",
    "ai_findings_json",
    "human_status",
    "reviewer",
    "target_gate",
    "corrected_input_json",
    "corrected_expected_json",
    "notes",
    "resolved_at",
    "promotion_exported_at",
    "promotion_file",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def yaml_dump(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def yaml_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping payload in {path}")
    return payload


def review_csv_fields() -> List[str]:
    return list(_REVIEW_CSV_FIELDS)


def _sanitize_text(text: Any) -> Any:
    if not isinstance(text, str):
        return text
    value = _EMAIL_RE.sub("<email>", text)
    value = _UUID_RE.sub("<uuid>", value)
    value = _LONG_DIGIT_RE.sub("<number>", value)
    return value


def _json_text(value: Any) -> str:
    if value in (None, "", {}, []):
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_json_text(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return {}
    return json.loads(text)


def _context_summary(source: Dict[str, Any]) -> str:
    context_window = list(source.get("context_window") or [])
    if not context_window:
        return ""
    parts: List[str] = []
    for item in context_window:
        prompt = str(dict(item).get("prompt") or "").strip()
        if prompt:
            parts.append(prompt)
    return " | ".join(parts)


def export_review_records_to_csv(
    payload: Dict[str, Any],
    *,
    out_path: Path,
    source_review_file: str,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=review_csv_fields())
        writer.writeheader()
        for record in payload.get("records") or []:
            ai_judge = dict(record.get("ai_judge") or {})
            human_review = dict(record.get("human_review") or {})
            writer.writerow(
                {
                    "source_review_file": source_review_file,
                    "case_id": str(record.get("id") or ""),
                    "task": str(record.get("task") or ""),
                    "gate": str(record.get("gate") or ""),
                    "prompt": str(dict(record.get("input") or {}).get("prompt") or ""),
                    "context_summary": _context_summary(dict(record.get("source") or {})),
                    "expected_json": _json_text(record.get("expected") or {}),
                    "observed_output_json": _json_text(record.get("observed_output") or {}),
                    "ai_verdict": str(ai_judge.get("verdict") or ""),
                    "ai_risk": str(ai_judge.get("risk") or ""),
                    "ai_confidence": str(ai_judge.get("confidence") or ""),
                    "ai_rationale": str(ai_judge.get("rationale") or ""),
                    "ai_findings_json": _json_text(ai_judge.get("findings") or []),
                    "human_status": str(human_review.get("status") or ""),
                    "reviewer": str(human_review.get("reviewer") or ""),
                    "target_gate": str(human_review.get("target_gate") or ""),
                    "corrected_input_json": _json_text(human_review.get("corrected_input") or {}),
                    "corrected_expected_json": _json_text(
                        human_review.get("corrected_expected") or {}
                    ),
                    "notes": str(human_review.get("notes") or ""),
                    "resolved_at": str(human_review.get("resolved_at") or ""),
                    "promotion_exported_at": str(
                        human_review.get("promotion_exported_at") or ""
                    ),
                    "promotion_file": str(human_review.get("promotion_file") or ""),
                }
            )
    return out_path


def _update_human_review_from_csv_row(
    human_review: Dict[str, Any],
    row: Dict[str, str],
) -> Dict[str, Any]:
    updated = dict(human_review)
    status = str(row.get("human_status") or "").strip()
    if status:
        updated["status"] = status
    reviewer = str(row.get("reviewer") or "")
    notes = str(row.get("notes") or "")
    target_gate = str(row.get("target_gate") or "").strip()
    resolved_at = str(row.get("resolved_at") or "").strip()
    promotion_exported_at = str(row.get("promotion_exported_at") or "").strip()
    promotion_file = str(row.get("promotion_file") or "").strip()

    updated["reviewer"] = reviewer or None
    updated["notes"] = notes
    updated["target_gate"] = target_gate or None
    updated["resolved_at"] = resolved_at or None
    updated["promotion_exported_at"] = promotion_exported_at or None
    updated["promotion_file"] = promotion_file or None

    corrected_input_text = str(row.get("corrected_input_json") or "").strip()
    corrected_expected_text = str(row.get("corrected_expected_json") or "").strip()
    if corrected_input_text:
        updated["corrected_input"] = _parse_json_text(corrected_input_text)
    else:
        updated["corrected_input"] = {}
    if corrected_expected_text:
        updated["corrected_expected"] = _parse_json_text(corrected_expected_text)
    else:
        updated["corrected_expected"] = {}
    validated = HumanReviewRecord.model_validate(updated)
    return validated.model_dump(mode="json", exclude_none=True)


def import_review_csv_rows(csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            source_review_file = str(row.get("source_review_file") or "").strip()
            case_id = str(row.get("case_id") or "").strip()
            if not source_review_file or not case_id:
                continue
            grouped.setdefault(source_review_file, []).append(
                {str(key): str(value or "") for key, value in row.items()}
            )
    return grouped


def deidentify_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if str(key) in {"user_id", "session_id", "trace_id", "raw_ref"}:
                continue
            cleaned[str(key)] = deidentify_value(item)
        return cleaned
    if isinstance(value, list):
        return [deidentify_value(item) for item in value]
    return _sanitize_text(value)


def _normalize_json(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _load_raw_payload(
    request_json: Dict[str, Any], response_json: Dict[str, Any]
) -> Dict[str, Any]:
    raw = request_json.get("raw")
    if isinstance(raw, dict):
        return raw
    raw = response_json.get("raw")
    if isinstance(raw, dict):
        return raw
    for ref in (request_json.get("raw_ref"), response_json.get("raw_ref")):
        if isinstance(ref, str) and ref:
            path = Path(ref)
            if path.exists():
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    return parsed
    return {}


def _load_postgres_interactions(
    limit: int,
    days: int,
    *,
    after_created_at: Optional[int] = None,
    after_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cfg = get_config()
    if not cfg.cache_db_url:
        raise RuntimeError("缺少 CACHE_DB_URL，无法读取 interactions。")
    if after_created_at is not None:
        sql = (
            "SELECT id, created_at, session_id, mode, prompt, request_json, response_json "
            "FROM interactions WHERE (created_at > %s OR (created_at = %s AND id > %s)) "
            "ORDER BY created_at ASC, id ASC LIMIT %s"
        )
        return fetch_all(
            cfg.cache_db_url,
            sql,
            (after_created_at, after_created_at, after_id or 0, limit),
        )
    cutoff = int(
        (datetime.now(timezone.utc) - timedelta(days=max(0, days))).timestamp()
    )
    sql = (
        "SELECT id, created_at, session_id, mode, prompt, request_json, response_json "
        "FROM interactions WHERE created_at >= %s "
        "ORDER BY created_at ASC, id ASC LIMIT %s"
    )
    return fetch_all(cfg.cache_db_url, sql, (cutoff, limit))


def _load_sqlite_interactions(
    limit: int,
    days: int,
    *,
    after_created_at: Optional[int] = None,
    after_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cfg = get_config()
    if cfg.interaction_store_path:
        path = Path(cfg.interaction_store_path)
    else:
        path = Path(__file__).resolve().parents[2] / ".cache" / "interactions.sqlite3"
    rows: List[Dict[str, Any]] = []
    with sqlite3.connect(path) as conn:
        cur = conn.cursor()
        if after_created_at is not None:
            cur.execute(
                "SELECT id, created_at, session_id, mode, prompt, request_json, response_json "
                "FROM interactions WHERE (created_at > ? OR (created_at = ? AND id > ?)) "
                "ORDER BY created_at ASC, id ASC LIMIT ?",
                (after_created_at, after_created_at, after_id or 0, limit),
            )
        else:
            cutoff = int(
                (datetime.now(timezone.utc) - timedelta(days=max(0, days))).timestamp()
            )
            cur.execute(
                "SELECT id, created_at, session_id, mode, prompt, request_json, response_json "
                "FROM interactions WHERE created_at >= ? "
                "ORDER BY created_at ASC LIMIT ?",
                (cutoff, limit),
            )
        for row in cur.fetchall():
            rows.append(
                {
                    "id": row[0],
                    "created_at": row[1],
                    "session_id": row[2],
                    "mode": row[3],
                    "prompt": row[4],
                    "request_json": row[5],
                    "response_json": row[6],
                }
            )
    return rows


def load_interactions(
    limit: int = 50,
    days: int = 30,
    *,
    after_created_at: Optional[int] = None,
    after_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cfg = get_config()
    store = (cfg.interaction_store or "").lower()
    if store == "postgres":
        return _load_postgres_interactions(
            limit,
            days,
            after_created_at=after_created_at,
            after_id=after_id,
        )
    if store == "sqlite":
        return _load_sqlite_interactions(
            limit,
            days,
            after_created_at=after_created_at,
            after_id=after_id,
        )
    raise RuntimeError(f"Unsupported interaction store for sampling: {store}")


def build_sampling_watermark(rows: Iterable[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    last_created_at: Optional[int] = None
    last_id: Optional[int] = None
    for row in rows:
        try:
            created_at = int(row.get("created_at") or 0)
            row_id = int(row.get("id") or 0)
        except Exception:
            continue
        if last_created_at is None or (created_at, row_id) > (last_created_at, last_id or 0):
            last_created_at = created_at
            last_id = row_id
    if last_created_at is None or last_id is None:
        return None
    return {
        "last_created_at": last_created_at,
        "last_id": last_id,
        "updated_at": utc_now_iso(),
    }


def _planner_case_from_interaction(
    row: Dict[str, Any], raw: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    prompt = str(row.get("prompt") or "").strip()
    response = dict(raw.get("response") or {})
    mode = str(response.get("mode") or row.get("mode") or "").strip()
    if not prompt or mode not in {"tool", "none", "workflow"}:
        return None
    expected: Dict[str, Any] = {"action": mode}
    observed_output: Dict[str, Any] = {"mode": mode}
    if mode == "tool":
        tool = dict(response.get("tool") or {})
        tool_name = str(tool.get("name") or "").strip()
        if not tool_name:
            return None
        expected["name"] = tool_name
        observed_output["tool_name"] = tool_name
        observed_output["message"] = _sanitize_text(tool.get("message") or "")
    elif mode == "none":
        plan = dict(response.get("plan") or {})
        observed_output["message"] = _sanitize_text(plan.get("message") or "")
    elif mode == "workflow":
        plan = dict(response.get("plan") or {})
        workflow_name = _extract_workflow_name(plan)
        if not workflow_name:
            return None
        expected["name"] = workflow_name
        observed_output["workflow_name"] = workflow_name
        observed_output["message"] = _sanitize_text(plan.get("message") or "")
        workflow_state = dict(dict(plan.get("data") or {}).get("workflow_state") or {})
        if workflow_state:
            observed_output["workflow_state"] = deidentify_value(workflow_state)
    return {
        "id": f"interaction_{row['id']}_planner",
        "gate": "audit",
        "source": {
            "interaction_id": row["id"],
            "created_at": row.get("created_at"),
            "provenance": "deidentified_interactions_table",
        },
        "input": {"prompt": _sanitize_text(prompt)},
        "expected": expected,
        "observed_output": deidentify_value(observed_output),
    }


def _extract_workflow_name(plan: Dict[str, Any]) -> Optional[str]:
    data = dict(plan.get("data") or {})
    workflow_name = str(data.get("workflow_name") or "").strip()
    if workflow_name:
        return workflow_name
    workflow = data.get("workflow")
    if isinstance(workflow, dict):
        candidate = str(workflow.get("workflow_name") or "").strip()
        if candidate:
            return candidate
        if workflow.get("plan_id") not in (None, "") or workflow.get("plan_filters"):
            return "growth_stage_query_workflow"
    if any(
        key in data
        for key in ("planting", "plant_season_id", "resolved_region_id", "save_response")
    ):
        return "crop_calendar_workflow"
    if plan.get("growth_stage") not in (None, ""):
        return "growth_stage_query_workflow"
    return None


def _extract_resolved_fields(tool_data: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(tool_data.get("resolved") or {})
    draft = dict(tool_data.get("draft") or {})
    fields: Dict[str, Any] = {}
    for source in (resolved, draft):
        for key in (
            "crop",
            "variety",
            "region_id",
            "culti_type",
            "planting_method",
            "sowing_date",
            "transplant_date",
        ):
            value = source.get(key)
            if value not in (None, "", []):
                fields[key] = value
    return deidentify_value(fields)


def _extractor_case_from_interaction(
    row: Dict[str, Any], raw: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    prompt = str(row.get("prompt") or "").strip()
    response = dict(raw.get("response") or {})
    tool = dict(response.get("tool") or {})
    if str(tool.get("name") or "").strip() != "sowing_suitability_lookup":
        return None
    expected = _extract_resolved_fields(dict(tool.get("data") or {}))
    if not prompt or not expected:
        return None
    return {
        "id": f"interaction_{row['id']}_extractor",
        "gate": "audit",
        "source": {
            "interaction_id": row["id"],
            "created_at": row.get("created_at"),
            "provenance": "deidentified_interactions_table",
        },
        "input": {"prompt": _sanitize_text(prompt)},
        "expected": expected,
        "observed_output": {
            "tool_name": "sowing_suitability_lookup",
            "resolved_fields": expected,
        },
    }


def _candidate_index_from_selected(
    candidates: List[Dict[str, Any]], selected: Dict[str, Any]
) -> Optional[int]:
    if not candidates or not selected:
        return None
    for idx, item in enumerate(candidates):
        if item == selected:
            return idx
    selected_name = selected.get("variety_name") or selected.get("品种名称")
    selected_region = selected.get("approval_region") or selected.get("审定区域")
    for idx, item in enumerate(candidates):
        if (item.get("variety_name") or item.get("品种名称")) == selected_name and (
            item.get("approval_region") or item.get("审定区域")
        ) == selected_region:
            return idx
    return None


def _variety_match_case_from_interaction(
    row: Dict[str, Any], raw: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    response = dict(raw.get("response") or {})
    tool = dict(response.get("tool") or {})
    if str(tool.get("name") or "").strip() != "variety_lookup":
        return None
    data = dict(tool.get("data") or {})
    raw_matches = list(data.get("raw_matches") or [])
    matches = list(data.get("matches") or [])
    candidates = raw_matches or matches
    if len(candidates) < 2:
        return None
    selected = dict(data.get("raw_selected") or data.get("selected") or {})
    selected_index = _candidate_index_from_selected(candidates, selected)
    if selected_index is None:
        return None
    region_choice = data.get("region_choice")
    query_text = _sanitize_text(data.get("query") or row.get("prompt") or "")
    prompt_text = query_text
    region_tokens: List[str] = []
    if isinstance(region_choice, str) and region_choice.strip():
        region_tokens = [region_choice.strip()]
        if region_choice not in prompt_text:
            prompt_text = f"{prompt_text} {region_choice}".strip()
    formatted_candidates: List[Dict[str, Any]] = []
    for idx, item in enumerate(candidates):
        payload = dict(deidentify_value(item))
        payload["index"] = idx
        formatted_candidates.append(payload)
    return {
        "id": f"interaction_{row['id']}_variety_match",
        "gate": "audit",
        "source": {
            "interaction_id": row["id"],
            "created_at": row.get("created_at"),
            "provenance": "deidentified_interactions_table",
        },
        "input": {
            "prompt": _sanitize_text(prompt_text),
            "region_tokens": deidentify_value(region_tokens),
            "candidates": formatted_candidates,
        },
        "expected": {"index": selected_index},
        "observed_output": {
            "selection_reason": _sanitize_text(data.get("selection_reason") or ""),
            "region_choice": _sanitize_text(region_choice),
            "selected": deidentify_value(selected),
        },
    }


def _summarize_interaction_context(
    row: Dict[str, Any], raw: Dict[str, Any]
) -> Dict[str, Any]:
    response = dict(raw.get("response") or {})
    summary: Dict[str, Any] = {
        "interaction_id": row.get("id"),
        "created_at": row.get("created_at"),
        "prompt": _sanitize_text(row.get("prompt") or ""),
        "mode": _sanitize_text(response.get("mode") or row.get("mode") or ""),
    }
    tool = dict(response.get("tool") or {})
    if tool:
        summary["tool_name"] = _sanitize_text(tool.get("name") or "")
        summary["message"] = _sanitize_text(tool.get("message") or "")
    plan = dict(response.get("plan") or {})
    if plan and not summary.get("message"):
        summary["message"] = _sanitize_text(plan.get("message") or "")
    workflow_name = _extract_workflow_name(plan)
    if workflow_name:
        summary["workflow_name"] = workflow_name
        workflow_state = dict(dict(plan.get("data") or {}).get("workflow_state") or {})
        if workflow_state:
            summary["workflow_state"] = deidentify_value(workflow_state)
    return deidentify_value(summary)


def _is_context_dependent_followup(
    prompt: str, previous_entry: Optional[Dict[str, Any]]
) -> bool:
    text = str(prompt or "").strip()
    if not text or previous_entry is None:
        return False
    normalized = re.sub(r"\s+", "", text)
    if len(text) <= 16 and _REGION_ONLY_RE.fullmatch(normalized):
        return True
    if (
        len(text) <= 16
        and _BRIEF_FOLLOWUP_SUFFIX_RE.search(text)
        and not any(token in text for token in _STANDALONE_QUERY_TOKENS)
    ):
        return True
    if (
        len(text) <= 16
        and any(sep in text for sep in ("，", ",", "、", "/"))
        and not any(token in text for token in _STANDALONE_QUERY_TOKENS)
    ):
        return True
    return False


def build_production_audit_batches(
    rows: Iterable[Dict[str, Any]],
    *,
    store_name: str,
) -> Dict[str, Dict[str, Any]]:
    generated_at = utc_now_iso()
    batches = {
        "planner": ProductionAuditBatch(
            task="planner",
            generated_at=generated_at,
            source_store=store_name,
        ).model_dump(mode="json"),
        "planner.context_dependent": ProductionAuditBatch(
            task="planner",
            generated_at=generated_at,
            source_store=store_name,
            sampling_scope="context_dependent",
            replay_mode="judge_only",
        ).model_dump(mode="json"),
        "extractor": ProductionAuditBatch(
            task="extractor",
            generated_at=generated_at,
            source_store=store_name,
        ).model_dump(mode="json"),
        "extractor.context_dependent": ProductionAuditBatch(
            task="extractor",
            generated_at=generated_at,
            source_store=store_name,
            sampling_scope="context_dependent",
            replay_mode="judge_only",
        ).model_dump(mode="json"),
        "variety_match": ProductionAuditBatch(
            task="variety_match",
            generated_at=generated_at,
            source_store=store_name,
        ).model_dump(mode="json"),
        "variety_match.context_dependent": ProductionAuditBatch(
            task="variety_match",
            generated_at=generated_at,
            source_store=store_name,
            sampling_scope="context_dependent",
            replay_mode="judge_only",
        ).model_dump(mode="json"),
    }
    seen_ids = {key: set() for key in batches}
    prepared_rows: List[Dict[str, Any]] = []
    for row in rows:
        request_json = _normalize_json(row.get("request_json"))
        response_json = _normalize_json(row.get("response_json"))
        raw = _load_raw_payload(request_json, response_json)
        if not raw:
            continue
        prepared_rows.append(
            {
                "row": row,
                "request_json": request_json,
                "response_json": response_json,
                "raw": raw,
            }
        )
    prepared_rows.sort(
        key=lambda item: (int(item["row"].get("created_at") or 0), int(item["row"].get("id") or 0))
    )
    prior_window_by_session: Dict[str, List[Dict[str, Any]]] = {}
    prior_window_by_row_id: Dict[int, List[Dict[str, Any]]] = {}
    for item in prepared_rows:
        row = item["row"]
        row_id = int(row.get("id") or 0)
        session_id = str(row.get("session_id") or "").strip()
        if session_id and session_id in prior_window_by_session:
            prior_window_by_row_id[row_id] = list(prior_window_by_session[session_id])
        if session_id:
            history = list(prior_window_by_session.get(session_id) or [])
            history.append(item)
            prior_window_by_session[session_id] = history[-_CONTEXT_WINDOW_LIMIT:]

    for item in prepared_rows:
        row = item["row"]
        raw = item["raw"]
        for task, builder in (
            ("planner", _planner_case_from_interaction),
            ("extractor", _extractor_case_from_interaction),
            ("variety_match", _variety_match_case_from_interaction),
        ):
            case = builder(row, raw)
            if not case:
                continue
            context_entries = prior_window_by_row_id.get(int(row.get("id") or 0), [])
            previous_entry = context_entries[-1] if context_entries else None
            is_context_dependent = _is_context_dependent_followup(
                str(row.get("prompt") or ""), previous_entry
            )
            bucket_key = f"{task}.context_dependent" if is_context_dependent else task
            if case["id"] in seen_ids[bucket_key]:
                continue
            seen_ids[bucket_key].add(case["id"])
            source = dict(case.get("source") or {})
            source["sampling_scope"] = (
                "context_dependent" if is_context_dependent else "standalone"
            )
            is_workflow_case = (
                str(dict(case.get("expected") or {}).get("action") or "").strip()
                == "workflow"
            )
            if (is_context_dependent or is_workflow_case) and context_entries:
                source["context_window"] = [
                    _summarize_interaction_context(entry["row"], entry["raw"])
                    for entry in context_entries
                ]
                if is_workflow_case and not is_context_dependent:
                    source["context_window_kind"] = "workflow_thread"
            case["source"] = source
            batches[bucket_key]["cases"].append(case)
    return batches


def save_production_audit_batches(
    batches: Dict[str, Dict[str, Any]], out_dir: Path
) -> List[Path]:
    paths: List[Path] = []
    for task_key, payload in batches.items():
        path = out_dir / f"{task_key}.yaml"
        yaml_dump(payload, path)
        paths.append(path)
    return paths


def _run_ai_judge(case: Dict[str, Any]) -> AuditJudgeDecision:
    llm = get_audit_judge_model()
    judge = llm.with_structured_output(AuditJudgeDecision)
    payload = {
        "task": case.get("task"),
        "input": case.get("input"),
        "expected": case.get("expected"),
        "observed_output": case.get("observed_output"),
        "rule_grade": case.get("rule_grade"),
        "source": case.get("source"),
    }
    result = judge.invoke(
        [
            ("system", PRODUCTION_AUDIT_JUDGE_SYSTEM_PROMPT),
            ("human", json.dumps(payload, ensure_ascii=False, default=str)),
        ]
    )
    return (
        result
        if isinstance(result, AuditJudgeDecision)
        else AuditJudgeDecision.model_validate(result)
    )


def build_review_records_from_batch(batch_path: Path) -> Dict[str, Any]:
    payload = yaml_load(batch_path)
    task = str(payload.get("task") or "").strip()
    runner = TASK_RUNNERS.get(task)
    if runner is None:
        raise ValueError(f"Unsupported audit batch task: {task}")
    replay_mode = str(payload.get("replay_mode") or "standalone_replay").strip()
    records: List[Dict[str, Any]] = []
    for case in payload.get("cases") or []:
        expected = dict(case.get("expected") or {})
        if replay_mode == "judge_only":
            rule_grade = {
                "available": False,
                "reason": "context_dependent_judge_only",
                "passed": None,
                "checked_fields": 0,
                "matched_fields": 0,
                "mismatches": [],
                "score": 0.0,
            }
        else:
            actual = runner(case)
            rule_grade = grade_case(expected, actual)
        record = AuditReviewRecord(
            id=str(case.get("id") or "unknown"),
            task=task,  # type: ignore[arg-type]
            gate=str(case.get("gate") or "audit"),
            input=dict(case.get("input") or {}),
            expected=expected,
            observed_output=dict(case.get("observed_output") or {}),
            source=dict(case.get("source") or {}),
            rule_grade=rule_grade,
        )
        record.ai_judge = _run_ai_judge(
            {
                "task": task,
                "input": record.input,
                "expected": record.expected,
                "observed_output": record.observed_output,
                "rule_grade": rule_grade,
                "source": record.source,
            }
        )
        records.append(record.model_dump(mode="json", exclude_none=True))
    return {
        "task": task,
        "line": payload.get("line") or "production_audit",
        "sampling_scope": payload.get("sampling_scope") or "standalone",
        "replay_mode": replay_mode,
        "generated_at": utc_now_iso(),
        "source_batch": str(batch_path),
        "records": records,
    }


def build_human_review_queue(
    review_payload: Dict[str, Any], *, max_confidence_auto_pass: float = 0.9
) -> Dict[str, Any]:
    queue: List[Dict[str, Any]] = []
    for record in review_payload.get("records") or []:
        human_review = HumanReviewRecord.model_validate(record.get("human_review") or {})
        if human_review.status != "pending":
            continue
        ai_judge = dict(record.get("ai_judge") or {})
        verdict = str(ai_judge.get("verdict") or "")
        confidence = float(ai_judge.get("confidence") or 0.0)
        if verdict != "pass" or confidence < max_confidence_auto_pass:
            queue.append(
                {
                    "id": record.get("id"),
                    "task": record.get("task"),
                    "source": dict(record.get("source") or {}),
                    "input": dict(record.get("input") or {}),
                    "expected": dict(record.get("expected") or {}),
                    "observed_output": dict(record.get("observed_output") or {}),
                    "ai_judge": ai_judge,
                    "human_review": human_review.model_dump(mode="json", exclude_none=True),
                }
            )
    return {
        "line": "production_audit",
        "generated_at": utc_now_iso(),
        "source_review": review_payload.get("source_batch"),
        "records": queue,
    }


def build_promotion_candidates(review_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for record in review_payload.get("records") or []:
        human = HumanReviewRecord.model_validate(record.get("human_review") or {})
        if human.status != "promote_to_expert":
            continue
        task = str(record.get("task") or "").strip()
        if task not in {"planner", "extractor", "variety_match"}:
            continue
        target = grouped.setdefault(
            task,
            {
                "task": task,
                "line": "expert",
                "owner": "production_audit_promotion",
                "generated_at": utc_now_iso(),
                "cases": [],
            },
        )
        expected = (
            dict(human.corrected_expected)
            if human.corrected_expected
            else dict(record.get("expected") or {})
        )
        input_payload = (
            dict(human.corrected_input)
            if human.corrected_input
            else dict(record.get("input") or {})
        )
        ai_judge = dict(record.get("ai_judge") or {})
        gate = human.target_gate or ai_judge.get("suggested_gate") or "regression"
        if gate not in {"blocking", "regression"}:
            gate = "regression"
        target["cases"].append(
            {
                "id": f"promoted_{record.get('id')}",
                "gate": gate,
                "source": {
                    **dict(record.get("source") or {}),
                    "promotion_source": "production_audit_review",
                },
                "input": input_payload,
                "expected": expected,
                "notes": human.notes,
            }
        )
    return grouped
