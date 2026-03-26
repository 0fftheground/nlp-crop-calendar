import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

import chainlit as cl
import httpx
import uuid

from src.infra.config import get_config
from src.observability.logging_utils import summarize_text

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
try:
    BACKEND_TIMEOUT_SECONDS = float(os.getenv("BACKEND_TIMEOUT_SECONDS", "90"))
except ValueError:
    BACKEND_TIMEOUT_SECONDS = 90.0
_AUTH_USERS_ENV = "CHAINLIT_AUTH_USERS"
_AUTH_USERNAME_ENV = "CHAINLIT_AUTH_USERNAME"
_AUTH_PASSWORD_ENV = "CHAINLIT_AUTH_PASSWORD"
_SESSION_ID_KEY = "session_id"
_CLIENT_ID_KEY = "client_id"
_USER_ID_KEY = "user_id"
_PROJECT_ROOT = Path(__file__).resolve().parent
_LOG_DIR = _PROJECT_ROOT / ".cache" / "logs"
_API_ERROR_LOG_PATH = _LOG_DIR / "api_errors.log"
_OBS_ERROR_LOG_PATH = _LOG_DIR / "observability.log"


def _build_capability_guide() -> str:
    return """欢迎使用农事助手。

支持的 Tool：

- `weather_lookup`：查询天气与农事适宜度
- `variety_lookup`：查询水稻品种基础信息
- `sowing_suitability_lookup`：查询播期推荐
- `plant_plan_list_active`：查询当前启用的种植计划
- `plant_plan_delete`：删除指定种植计划
- `growth_stage_lookup`：根据已有种植计划查询生育期结果

支持的 Workflow：

- `crop_calendar_workflow`：生成完整种植计划与农事方案

你可以直接这样问：

- 种植计划：`帮我做一份水稻种植计划`
- 生育期查询：`查询计划id=123的生育期预测结果`
- 天气与适宜度：`查询长沙2025-01-01到2025-01-03的天气`
- 播期推荐：`查询美香占2号、一季晚稻、直播在长沙的播期推荐`
- 品种信息：`南粳9108的生育期和适宜种植区域是什么？`
- 计划管理：`列出当前启用的种植计划` / `删除种植计划 plant_season_id=123`

提问时尽量带上：`地区 / 品种 / 稻作类型 / 播种方式 / 日期或计划ID`
如需保存计划，默认使用系统配置的 `DEFAULT_FARM_ID`。
"""


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


def _get_recent_week_farm_work_api_url(farm_id: object) -> Optional[str]:
    cfg = get_config()
    raw = str(getattr(cfg, "recent_week_farm_work_api_url", "") or "").strip()
    if raw:
        if "{farm_id}" in raw:
            return raw.format(farm_id=farm_id)
        return f"{raw.rstrip('/')}/{farm_id}"
    return _join_api_url(
        getattr(cfg, "business_api_base_url", None),
        f"/farm-work/recent-week/{farm_id}",
    )


def _extract_recent_farm_work_items(plan: object) -> list[tuple[str, str]]:
    if not isinstance(plan, dict):
        return []
    raw_items = None
    for key in (
        "farm_works",
        "farmWorks",
        "works",
        "tasks",
        "items",
        "list",
        "farmworks",
    ):
        value = plan.get(key)
        if value is not None:
            raw_items = value
            break
    if isinstance(raw_items, dict):
        normalized: list[tuple[str, str]] = []
        for name, work_date in raw_items.items():
            name_text = str(name).strip()
            date_text = str(work_date).strip()
            if name_text or date_text:
                normalized.append((date_text, name_text))
        return normalized
    if not isinstance(raw_items, list):
        return []
    normalized = []
    for item in raw_items:
        if isinstance(item, dict):
            name = ""
            for key in ("name", "title", "task_name", "taskName", "work_name", "workName"):
                value = item.get(key)
                if str(value or "").strip():
                    name = str(value).strip()
                    break
            work_date = ""
            for key in ("date", "work_date", "task_date", "taskDate", "day"):
                value = item.get(key)
                if str(value or "").strip():
                    work_date = str(value).strip()
                    break
            if name or work_date:
                normalized.append((work_date, name))
            continue
        text = str(item).strip()
        if text:
            normalized.append(("", text))
    return normalized


def _format_recent_farm_work_summary(payload: object, *, farm_id: object) -> str:
    farm_id_text = str(farm_id).strip()
    today = date.today()
    end_day = today + timedelta(days=6)
    date_range_text = f"{today.isoformat()} 至 {end_day.isoformat()}"
    if not isinstance(payload, dict):
        return (
            f"默认农场（farm_id={farm_id_text}）未来 7 天农事"
            f"（{date_range_text}）暂时无法加载。"
        )
    code = str(payload.get("code", "")).strip()
    data = payload.get("data")
    plans = data.get("plans") if isinstance(data, dict) else None
    if code == "204" or not isinstance(plans, list) or not plans:
        return (
            f"默认农场（farm_id={farm_id_text}）未来 7 天农事"
            f"（{date_range_text}）暂无安排。"
        )
    lines = [
        f"默认农场（farm_id={farm_id_text}）未来 7 天农事（{date_range_text}）："
    ]
    shown = 0
    for plan in plans:
        if not isinstance(plan, dict):
            continue
        items = _extract_recent_farm_work_items(plan)
        if not items:
            continue
        plan_name = ""
        for key in (
            "plan_name",
            "name",
            "planTitle",
            "title",
            "plan",
            "crop_name",
            "crop",
        ):
            value = plan.get(key)
            if str(value or "").strip():
                plan_name = str(value).strip()
                break
        if not plan_name:
            plan_id = str(plan.get("plan_id") or plan.get("id") or "").strip()
            plan_name = f"计划 {plan_id}" if plan_id else "未命名计划"
        preview = []
        for work_date, name in items[:4]:
            if work_date and name:
                preview.append(f"{work_date} {name}")
            elif work_date:
                preview.append(work_date)
            elif name:
                preview.append(name)
        if len(items) > 4:
            preview.append(f"等 {len(items)} 项")
        if preview:
            lines.append(f"- {plan_name}：{'；'.join(preview)}")
            shown += 1
        if shown >= 5:
            remaining = max(0, len(plans) - shown)
            if remaining:
                lines.append(f"- 其余 {remaining} 个计划已省略")
            break
    if shown == 0:
        return (
            f"默认农场（farm_id={farm_id_text}）未来 7 天农事"
            f"（{date_range_text}）暂无安排。"
        )
    return "\n".join(lines)


def _append_recent_farm_work_error_log(message: str) -> None:
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _API_ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


def _append_recent_farm_work_observability_log(**fields: object) -> None:
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat()
    payload = {"event": "recent_farm_work_summary_error", "trace_id": "unknown", **fields}
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        with _OBS_ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} - {json.dumps(payload, ensure_ascii=False, default=str)}\n")
    except Exception:
        pass


async def _fetch_recent_farm_work_summary() -> str:
    cfg = get_config()
    farm_id = str(getattr(cfg, "default_farm_id", None) or "").strip()
    if not farm_id:
        return ""
    url = _get_recent_week_farm_work_api_url(farm_id)
    if not url:
        return ""
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(BACKEND_TIMEOUT_SECONDS),
            trust_env=_trust_env_for_backend(url),
        ) as client:
            response = await client.get(
                url,
                headers=_build_api_headers(
                    api_key=getattr(cfg, "business_api_key", None)
                ),
            )
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        status_code = None
        response_text = ""
        if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
            status_code = exc.response.status_code
            response_text = summarize_text(exc.response.text, limit=300)
        _append_recent_farm_work_observability_log(
            farm_id=farm_id,
            url=url,
            error_type=type(exc).__name__,
            error=str(exc),
            status_code=status_code,
            response_text=response_text,
        )
        _append_recent_farm_work_error_log(
            (
                f"recent_farm_work_summary_error farm_id={farm_id} url={url} "
                f"error_type={type(exc).__name__} status_code={status_code} "
                f"error={exc} response_text={response_text}"
            )
        )
        return _format_recent_farm_work_summary(None, farm_id=farm_id)
    return _format_recent_farm_work_summary(payload, farm_id=farm_id)


def _is_capability_help_prompt(prompt: str) -> bool:
    text = (prompt or "").strip().lower()
    if not text:
        return False
    exact_hits = {
        "help",
        "功能",
        "示例",
        "例子",
        "帮助",
        "你能提供哪些功能",
        "你能做什么",
        "你能干什么",
        "支持哪些功能",
        "支持什么功能",
    }
    if text in exact_hits:
        return True
    fuzzy_hits = (
        "能提供哪些功能",
        "支持哪些功能",
        "有哪些功能",
        "可以做什么",
        "可用功能",
        "示例提问",
    )
    return any(item in text for item in fuzzy_hits)


def _format_weather_tool_details(
    tool_name: str, data: object, base_message: str = ""
) -> str:
    if tool_name not in {"weather_lookup", "growth_weather_lookup"}:
        return ""
    if not isinstance(data, dict):
        return ""
    lines: list[str] = []
    region = data.get("region")
    start_date = data.get("start_date")
    end_date = data.get("end_date")
    points = data.get("points")
    if region or start_date or end_date:
        meta = []
        if region:
            meta.append(f"区域：{region}")
        if start_date and end_date:
            meta.append(f"时间：{start_date} 至 {end_date}")
        elif start_date:
            meta.append(f"开始：{start_date}")
        elif end_date:
            meta.append(f"结束：{end_date}")
        if meta:
            lines.append("，".join(meta))
    advice_labels = {
        "sf_ws": "施肥",
        "lm_ws": "炼苗",
        "yz_ws": "移栽",
        "fd_ws": "翻地",
        "dy_ws": "打药",
        "sg_ws": "收割",
        "zd_ws": "整地",
    }
    requested_operations = data.get("requested_operations")
    requested_label_set = None
    if isinstance(requested_operations, list):
        requested_label_set = {
            str(item).strip()
            for item in requested_operations
            if str(item).strip()
        }

    def _format_score(score: object) -> str:
        if isinstance(score, int):
            return str(score)
        if isinstance(score, float):
            return f"{score:.2f}".rstrip("0").rstrip(".")
        return str(score)

    def _detect_reason_factor_keys(reason_text: str) -> list[str]:
        text = str(reason_text or "").strip()
        if not text:
            return []
        factor_patterns = (
            ("wind_speed", ("风速", "风力", "风大", "大风", "风较大", "风偏大")),
            ("precipitation", ("降水", "降雨", "下雨", "有雨", "雨", "雨量")),
            ("humidity", ("湿度", "潮湿", "湿润", "湿", "墒情")),
            (
                "temperature",
                ("温度", "气温", "高温", "低温", "升温", "降温", "温差", "冷", "热"),
            ),
        )
        matches: list[str] = []
        for key, patterns in factor_patterns:
            if any(pattern in text for pattern in patterns):
                matches.append(key)
        return matches

    def _format_reason_factors(item: dict, reason: object) -> str:
        reason_text = str(reason or "").strip()
        negative_markers = ("不适合", "不宜", "不建议", "较大", "偏大", "过大", "偏高", "偏低")
        if not reason_text or not any(marker in reason_text for marker in negative_markers):
            return ""
        factor_keys = _detect_reason_factor_keys(reason_text)
        if not factor_keys:
            return ""
        parts: list[str] = []
        if "wind_speed" in factor_keys and item.get("wind_speed") is not None:
            parts.append(f"风速 {_format_score(item.get('wind_speed'))}m/s")
        if "precipitation" in factor_keys and item.get("precipitation") is not None:
            parts.append(f"降水 {_format_score(item.get('precipitation'))}mm")
        if "humidity" in factor_keys and item.get("humidity") is not None:
            parts.append(f"湿度 {_format_score(item.get('humidity'))}%")
        if "temperature" in factor_keys:
            tmax = item.get("temperature_max")
            tmin = item.get("temperature_min")
            tavg = item.get("temperature")
            if tmax is not None and tmin is not None:
                parts.append(
                    f"气温 {_format_score(tmin)}至{_format_score(tmax)}°C"
                )
            elif tavg is not None:
                parts.append(f"气温 {_format_score(tavg)}°C")
        return "；".join(parts)

    def _format_operation_advice(
        label: str, score: object, reason: object, item: dict
    ) -> Optional[str]:
        if score is None and not reason:
            return None
        advice = f"{label}适宜度"
        if score is not None:
            advice = f"{advice} {_format_score(score)}"
        if reason:
            factor_text = _format_reason_factors(item, reason)
            if factor_text:
                advice = f"{advice}（{reason}；{factor_text}）"
            else:
                advice = f"{advice}（{reason}）"
        return advice

    if isinstance(points, list):
        lines.append(f"天数：{len(points)}")
        preview = points[:7]
        if preview:
            lines.append("逐日预览：")
        for item in preview:
            if not isinstance(item, dict):
                continue
            ts = str(item.get("timestamp") or "")[:10]
            tmax = item.get("temperature_max")
            tmin = item.get("temperature_min")
            tavg = item.get("temperature")
            rain = item.get("precipitation")
            segs = []
            if tmax is not None and tmin is not None:
                segs.append(f"{tmin}至{tmax}°C")
            elif tavg is not None:
                segs.append(f"{tavg}°C")
            if not segs and item.get("condition"):
                segs.append(str(item.get("condition")))
            advice_parts = []
            for score_key, label in advice_labels.items():
                if requested_label_set is not None and label not in requested_label_set:
                    continue
                score = item.get(score_key)
                reason = item.get(score_key.replace("_ws", "_reason"))
                advice = _format_operation_advice(label, score, reason, item)
                if advice:
                    advice_parts.append(advice)
            line = f"- {ts}" if ts else "- "
            if segs:
                line += " " + "，".join(segs)
            if advice_parts:
                line += "；" + "；".join(advice_parts)
            lines.append(line.rstrip())
        if len(points) > len(preview):
            lines.append(f"... 其余 {len(points) - len(preview)} 天已省略")
    detail_text = "\n".join([line for line in lines if line])
    if not detail_text:
        return ""
    if base_message and detail_text in base_message:
        return ""
    return detail_text


def _format_sowing_suitability_details(
    tool_name: str, data: object, base_message: str = ""
) -> str:
    if tool_name != "sowing_suitability_lookup":
        return ""
    if not isinstance(data, dict):
        return ""
    result = data.get("result")
    resolved = data.get("resolved")
    if not isinstance(result, dict):
        return ""
    lines: list[str] = []
    method_labels = {
        "direct_seeding": "直播",
        "transplanting": "移栽",
    }

    def _clean_dates(values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    def _format_date_range(values: list[str], *, full_threshold: int = 7) -> str:
        if not values:
            return ""
        if len(values) <= full_threshold:
            return "、".join(values)
        if len(values) == 1:
            return values[0]
        return f"{values[0]} 至 {values[-1]}（共{len(values)}天）"

    if isinstance(resolved, dict):
        meta = []
        for label, key in (
            ("品种", "variety"),
            ("稻作类型", "culti_type"),
            ("播种方式", "planting_method"),
            ("区域", "region_id"),
        ):
            value = resolved.get(key)
            if value:
                if key == "planting_method":
                    value = method_labels.get(str(value), str(value))
                meta.append(f"{label}：{value}")
        if meta:
            lines.append("，".join(meta))
    valid_dates = _clean_dates(result.get("suitDate"))
    invalid_dates = _clean_dates(result.get("unsuitDate"))
    reasons = _clean_dates(result.get("unsuitReasons"))
    if valid_dates:
        lines.append(f"推荐播期：{_format_date_range(valid_dates)}")
    if invalid_dates:
        lines.append(f"不推荐日期：{_format_date_range(invalid_dates, full_threshold=10)}")
    if reasons:
        deduped_reasons = list(dict.fromkeys(reasons))
        lines.append(f"原因：{'；'.join(deduped_reasons[:3])}")
    detail_text = "\n".join([line for line in lines if line])
    if not detail_text:
        return ""
    if base_message and detail_text in base_message:
        return ""
    return detail_text


def _format_tool_response_message(tool: dict) -> str:
    tool_name = str(tool.get("name") or "")
    message_text = str(tool.get("message") or "").strip()
    data = tool.get("data")
    content = f"工具 `{tool_name}` 已执行："
    if message_text:
        content += f"\n{message_text}"
    detail_text = _format_weather_tool_details(tool_name, data, base_message=message_text)
    if not detail_text:
        detail_text = _format_sowing_suitability_details(
            tool_name, data, base_message=message_text
        )
    if detail_text:
        content += f"\n\n{detail_text}"
    return content


def _trust_env_for_backend(url: str) -> bool:
    host = urlparse(url).hostname
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    return True


def _load_auth_users() -> dict[str, str]:
    users: dict[str, str] = {}
    raw = (os.getenv(_AUTH_USERS_ENV) or "").strip()
    if raw:
        for item in raw.split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            username, password = item.split(":", 1)
            username = username.strip()
            password = password.strip()
            if username and password:
                users[username] = password
    if users:
        return users
    username = (os.getenv(_AUTH_USERNAME_ENV) or "").strip()
    password = (os.getenv(_AUTH_PASSWORD_ENV) or "").strip()
    if username and password:
        users[username] = password
    return users


def _resolve_user_identifier() -> Optional[str]:
    try:
        user = cl.user_session.get("user")
    except Exception:
        user = None
    identifier = getattr(user, "identifier", None) if user else None
    if identifier:
        return str(identifier)
    ctx = getattr(cl, "context", None)
    if ctx:
        ctx_user = getattr(ctx, "user", None)
        identifier = getattr(ctx_user, "identifier", None) if ctx_user else None
        if identifier:
            return str(identifier)
        session = getattr(ctx, "session", None)
        session_user = getattr(session, "user", None) if session else None
        identifier = getattr(session_user, "identifier", None) if session_user else None
        if identifier:
            return str(identifier)
    return None


def _resolve_thread_id() -> Optional[str]:
    ctx = getattr(cl, "context", None)
    if not ctx:
        return None
    session = getattr(ctx, "session", None)
    if not session:
        return None
    for attr in ("thread_id", "id"):
        value = getattr(session, attr, None)
        if value:
            return str(value)
    return None


def _resolve_thread_id_from_resume(thread: object) -> Optional[str]:
    if isinstance(thread, dict):
        return thread.get("id") or thread.get("thread_id")
    for attr in ("id", "thread_id"):
        value = getattr(thread, attr, None)
        if value:
            return str(value)
    return None


async def _clear_status_message(status_msg: Optional[cl.Message]) -> None:
    if not status_msg:
        return
    try:
        await status_msg.remove()
        return
    except Exception:
        try:
            await status_msg.update(content="")
        except Exception:
            return


def _ensure_session_ids(thread_id: Optional[str] = None) -> tuple[str, str]:
    user_id = cl.user_session.get(_USER_ID_KEY)
    if not user_id:
        user_id = _resolve_user_identifier()
    if not user_id:
        user_id = cl.user_session.get(_CLIENT_ID_KEY)
    if not user_id:
        user_id = str(uuid.uuid4())
    cl.user_session.set(_USER_ID_KEY, user_id)
    if not cl.user_session.get(_CLIENT_ID_KEY):
        cl.user_session.set(_CLIENT_ID_KEY, user_id)
    if thread_id:
        cl.user_session.set(_SESSION_ID_KEY, thread_id)
    session_id = cl.user_session.get(_SESSION_ID_KEY)
    if not session_id:
        session_id = _resolve_thread_id()
    if not session_id:
        session_id = str(uuid.uuid4())
    cl.user_session.set(_SESSION_ID_KEY, session_id)
    return session_id, user_id


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    users = _load_auth_users()
    if users:
        expected = users.get(username)
        if expected and expected == password:
            return cl.User(identifier=username)
        return None
    if username.strip() and password.strip():
        return cl.User(identifier=username.strip())
    return None


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    _ensure_session_ids()
    await cl.Message(content=_build_capability_guide()).send()
    recent_work_summary = await _fetch_recent_farm_work_summary()
    if recent_work_summary:
        await cl.Message(content=recent_work_summary).send()


@cl.on_chat_resume
async def resume(thread: object):
    thread_id = _resolve_thread_id_from_resume(thread)
    _ensure_session_ids(thread_id=thread_id)


@cl.on_message
async def on_message(message: cl.Message):
    prompt = message.content.strip()
    if not prompt:
        await cl.Message(content="请输入有效的问题。").send()
        return
    if _is_capability_help_prompt(prompt):
        await cl.Message(content=_build_capability_guide()).send()
        return

    status_msg = await cl.Message(content="正在分析，请稍候...").send()
    try:
        session_id, user_id = _ensure_session_ids()
        async with httpx.AsyncClient(
            base_url=BACKEND_URL,
            timeout=httpx.Timeout(BACKEND_TIMEOUT_SECONDS),
            trust_env=_trust_env_for_backend(BACKEND_URL),
        ) as client:
            response = await client.post(
                "/api/v1/handle",
                json={
                    "prompt": prompt,
                    "session_id": session_id,
                    "user_id": user_id,
                },
            )
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        await _clear_status_message(status_msg)
        await cl.Message(content=f"请求失败: {exc}").send()
        return

    await _clear_status_message(status_msg)
    mode = data.get("mode")
    if mode == "tool" and data.get("tool"):
        tool = data["tool"]
        content = _format_tool_response_message(tool)
        await cl.Message(content=content).send()
        return
    if mode == "none":
        plan = data.get("plan") or {}
        content = plan.get("message", "") or "暂时无法识别与农事相关的需求。"
        await cl.Message(content=content).send()
        return

    plan = data.get("plan") or {}
    content = plan.get("message", "")
    trace = "\n".join(plan.get("trace", []))
    await cl.Message(content=content or "未生成计划。").send()
    if trace and ("还需要补充" in content or "需要补充" in content):
        await cl.Message(content="已进入 workflow，需要补充字段", author="debug").send()
