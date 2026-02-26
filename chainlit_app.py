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
    return """欢迎来到农事助手！

我目前支持两类能力：`workflow`（多步骤流程）和 `tool`（单点查询/操作）。

当前限制：
- 创建/保存种植计划仅支持默认农场（`DEFAULT_FARM_ID`），暂不支持前端指定农场。

可直接参考下面示例提问：

## Workflow（复杂任务）

1. 完整种植计划（`crop_calendar_workflow`）
- 示例：`请基于默认农场，按水稻品种南粳9108、5月20日插秧生成完整农事计划。`
- 示例：`帮我给默认农场做一份水稻种植计划。`（信息不足时我会继续追问）

2. 生育期查询（`growth_stage_query_workflow`）
- 示例：`查询计划id=123的生育期预测结果。`
- 示例：`帮我查一下默认农场里南粳9108相关计划的生育期。`（可能会让你选择计划）

## Tool（单点查询/操作）

1. 天气查询（`weather_lookup`）
- 示例：`查询默认农场2026-03-01到2026-03-07的天气。`

2. 品种信息查询（`variety_lookup`）
- 示例：`南粳9108的生育期、抗性和适宜种植区域是什么？`

3. 查看启用计划（`plant_plan_list_active`）
- 示例：`列出默认农场当前启用的种植计划。`

4. 删除种植计划（`plant_plan_delete`）
- 示例：`删除种植计划 plant_season_id=123`

5. 清除历史经验（`memory_clear`）
- 示例：`清除历史经验记录`

提问建议：
- 尽量提供：`地区 / 作物 / 品种 / 播种或插秧日期 / 种植方式`
- 查询类请求尽量给出：`计划ID` 或明确的时间范围
- 如需生成/保存计划，默认写入系统配置的默认农场
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
        message_text = tool.get("message") or ""
        content = f"工具 `{tool.get('name')}` 已执行：\n{message_text}"
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
