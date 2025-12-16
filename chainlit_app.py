import os

import chainlit as cl
import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@cl.on_chat_start
async def start():
    cl.user_session.set("history", [])
    await cl.Message(content="👩‍🌾 欢迎来到农事助手！").send()


@cl.on_message
async def on_message(message: cl.Message):
    prompt = message.content.strip()
    if not prompt:
        await cl.Message(content="请输入有效的问题。").send()
        return

    await cl.Message(content="正在分析，请稍候...").send()
    try:
        async with httpx.AsyncClient(base_url=BACKEND_URL, timeout=30) as client:
            response = await client.post("/api/v1/handle", json={"prompt": prompt})
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
            await cl.Message(content=f"附加数据:\n{tool['data']}", author="debug", indent=1).send()
        return

    plan = data.get("plan") or {}
    content = plan.get("message", "")
    trace = "\n".join(plan.get("trace", []))
    await cl.Message(content=content or "未生成计划。").send()
    if trace:
        await cl.Message(content=f"调试信息:\n{trace}", author="debug", indent=1).send()
