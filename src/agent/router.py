from typing import Optional

from ..schemas.models import (
    HandleResponse,
    ToolInvocation,
    UserRequest,
    WorkflowResponse,
)
from ..tools.registry import execute_tool
from .tool_selector import ToolSelector
from .workflows.crop_graph import build_graph


class RequestRouter:
    """Decide whether to run a standalone tool or the LangGraph workflow."""

    def __init__(self):
        self.graph = build_graph()
        self.selector = ToolSelector()

    def handle(self, request: UserRequest) -> HandleResponse:
        tool_result = self._try_tool(request.prompt)
        if tool_result:
            return HandleResponse(mode="tool", tool=tool_result)
        plan = self._run_workflow(request)
        return HandleResponse(mode="workflow", plan=plan)

    def _try_tool(self, prompt: str) -> Optional[ToolInvocation]:
        decision = self.selector.decide(prompt)
        if decision.get("action") == "tool" and decision.get("tool"):
            result = execute_tool(decision["tool"], prompt)
            if result:
                result.data["decision_reason"] = decision.get("reason")
                return result
        return None

    def _run_workflow(self, request: UserRequest) -> WorkflowResponse:
        initial_state = {"user_prompt": request.prompt, "trace": []}
        state = self.graph.invoke(initial_state)
        return WorkflowResponse(
            query=state["query"],
            recommendations=state.get("recommendations", []),
            message=state.get("message", ""),
            trace=state.get("trace", []),
        )
