import os
import re
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
_DOWNLOAD_PATTERN = re.compile(r"(^|\s)(/api/v1/download/[0-9a-f]{32})")

def _trust_env_for_backend(url: str) -> bool:
    host = urlparse(url).hostname
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    return True


def _expand_download_links(message: str) -> str:
    if not message:
        return message
    base_url = (BACKEND_URL or "").rstrip("/")
    if not base_url:
        return message

    def _replace(match: re.Match) -> str:
        return f"{match.group(1)}{base_url}{match.group(2)}"

    return _DOWNLOAD_PATTERN.sub(_replace, message)


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
    await cl.Message(content="欢迎来到农事助手！").send()


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

    await cl.Message(content="正在分析，请稍候...").send()
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
        await cl.Message(content=f"请求失败: {exc}").send()
        return

    mode = data.get("mode")
    if mode == "tool" and data.get("tool"):
        tool = data["tool"]
        message_text = _expand_download_links(tool.get("message") or "")
        content = f"工具 `{tool.get('name')}` 已执行：\n{message_text}"
        await cl.Message(content=content).send()
        return
    if mode == "none":
        plan = data.get("plan") or {}
        content = plan.get("message", "") or "暂时无法识别与农事相关的需求。"
        await cl.Message(content=content).send()
        return

    plan = data.get("plan") or {}
    content = _expand_download_links(plan.get("message", ""))
    trace = "\n".join(plan.get("trace", []))
    await cl.Message(content=content or "未生成计划。").send()
    if trace and ("还需要补充" in content or "需要补充" in content):
        await cl.Message(content="已进入 workflow，需要补充字段", author="debug").send()
