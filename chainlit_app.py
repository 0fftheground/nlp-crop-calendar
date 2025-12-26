import os
from urllib.parse import urlparse

import chainlit as cl
import httpx
import uuid

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

def _trust_env_for_backend(url: str) -> bool:
    host = urlparse(url).hostname
    if host in {"localhost", "127.0.0.1", "::1"}:
        return False
    return True


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    if not cl.user_session.get("session_id"):
        cl.user_session.set("session_id", str(uuid.uuid4()))
    await cl.Message(content="👩‍🌾 欢迎来到农事助手！").send()


@cl.on_message
async def on_message(message: cl.Message):
    prompt = message.content.strip()
    if not prompt:
        await cl.Message(content="请输入有效的问题。").send()
        return

    await cl.Message(content="正在分析，请稍候...").send()
    try:
        async with httpx.AsyncClient(
            base_url=BACKEND_URL,
            timeout=30,
            trust_env=_trust_env_for_backend(BACKEND_URL),
        ) as client:
            response = await client.post("/api/v1/handle", json={"prompt": prompt, "session_id": cl.user_session.get("session_id")})
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        await cl.Message(content=f"请求失败: {exc}").send()
        return

    mode = data.get("mode")
    if mode == "tool" and data.get("tool"):
        tool = data["tool"]
        content = f"🛠️ 工具 `{tool.get('name')}` 已执行：\n{tool.get('message')}"
        await cl.Message(content=content).send()
        if tool.get("data"):
            await cl.Message(content=f"附加数据:\n{tool['data']}", author="debug").send()
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
