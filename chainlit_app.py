import os
from urllib.parse import urlparse
from typing import Optional

import chainlit as cl
import httpx
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
_AUTH_USERS_ENV = "CHAINLIT_AUTH_USERS"
_AUTH_USERNAME_ENV = "CHAINLIT_AUTH_USERNAME"
_AUTH_PASSWORD_ENV = "CHAINLIT_AUTH_PASSWORD"
_SESSION_ID_KEY = "session_id"
_CLIENT_ID_KEY = "client_id"
_USER_ID_KEY = "user_id"


def _build_capability_guide() -> str:
    return """欢迎使用农事助手。

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
            meta.append(f"时间：{start_date} ~ {end_date}")
        elif start_date:
            meta.append(f"开始：{start_date}")
        elif end_date:
            meta.append(f"结束：{end_date}")
        if meta:
            lines.append("，".join(meta))
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
                segs.append(f"{tmin}~{tmax}°C")
            elif tavg is not None:
                segs.append(f"{tavg}°C")
            if rain is not None:
                segs.append(f"降水 {rain}mm")
            if not segs and item.get("condition"):
                segs.append(str(item.get("condition")))
            line = f"- {ts}" if ts else "- "
            if segs:
                line += " " + "，".join(segs)
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
        return f"{values[0]} ~ {values[-1]}（共{len(values)}天）"

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
            timeout=30,
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
