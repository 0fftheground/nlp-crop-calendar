import json

from langchain_core.messages import HumanMessage, SystemMessage

from ..data.llm import get_chat_model
from ..tools.registry import list_tool_specs


class ToolSelector:
    """Use LLM to decide whether to run a tool or workflow."""

    def __init__(self):
        self.llm = get_chat_model()

    def decide(self, prompt: str) -> dict:
        specs = list_tool_specs()
        system_prompt = self._build_prompt(specs)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        payload = self._parse_json(getattr(response, "content", str(response)))
        if payload.get("action") not in {"tool", "workflow"}:
            payload["action"] = "workflow"
            payload["tool"] = None
        return payload

    def _build_prompt(self, specs) -> str:
        tools_text = "\n".join(
            [f"- {tool['name']}: {tool['description']}" for tool in specs]
        )
        return (
            "你是农事助手的任务分发器。根据用户的自然语言输入，决定以下两种方式之一：\n"
            "1) tool: 针对常见的快速查询（如下所列工具）。\n"
            "2) workflow: 复杂的农事规划，需要调用 LangGraph 工作流。\n"
            f"可用工具:\n{tools_text}\n\n"
            "请输出 JSON:\n"
            '{"action":"tool|workflow","tool":"tool_name or null","reason":"简短理由"}\n'
            "只返回 JSON，不要额外文字。"
        )

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"action": "workflow", "tool": None, "reason": "parse_error"}
