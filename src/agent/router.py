import contextvars
import json
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool as lc_tool

from ..domain.services import MissingPlantingInfoError
from ..infra.llm import get_chat_model
from ..schemas.models import HandleResponse, ToolInvocation, UserRequest, WorkflowResponse
from ..tools.registry import build_agent_tools, execute_tool
from .workflows.crop_graph import (
    build_graph as build_crop_graph,
    build_growth_stage_graph,
)


class RequestRouter:
    """Use a LangChain agent to route tool calls and craft responses."""

    _session_id_ctx = contextvars.ContextVar("session_id", default="default")
    CROP_WORKFLOW_TOOL = "crop_calendar_workflow"
    GROWTH_WORKFLOW_TOOL = "growth_stage_workflow"

    def __init__(self):
        self.crop_graph = None
        self.growth_graph = None
        self.tools = build_agent_tools() + [
            self._build_crop_workflow_tool(),
            self._build_growth_stage_workflow_tool(),
        ]
        self.agent = self._build_agent()
        self._pending_followups: Dict[str, dict] = {}

    def handle(self, request: UserRequest) -> HandleResponse:
        session_id = request.session_id or "default"
        token = self._session_id_ctx.set(session_id)
        try:
            # If the session is mid follow-up, skip routing and resume directly.
            pending = self._pending_followups.get(session_id)
            if pending:
                if pending.get("mode") == "tool":
                    tool_payload = self._run_tool_followup(
                        request.prompt, pending, session_id
                    )
                    return HandleResponse(mode="tool", tool=tool_payload)
                workflow_name = pending.get("workflow_name")
                if workflow_name == self.GROWTH_WORKFLOW_TOOL:
                    plan = self._run_growth_stage_workflow(request.prompt)
                else:
                    plan = self._run_crop_calendar_workflow(request.prompt)
                return HandleResponse(mode="workflow", plan=plan)
            # No pending state: let the LLM agent decide tool vs workflow.
            result = self.agent.invoke({"input": request.prompt})
        finally:
            self._session_id_ctx.reset(token)
        steps = result.get("intermediate_steps", [])
        trace = self._format_trace(steps)
        # Prefer workflow payload if a workflow tool was invoked.
        workflow_payload = self._extract_workflow_payload(steps)
        if workflow_payload:
            plan = WorkflowResponse(**workflow_payload)
            plan.trace = trace
            return HandleResponse(mode="workflow", plan=plan)

        # Otherwise handle standalone tool results (may require follow-up).
        tool_payload = self._extract_tool_payload(steps)
        if tool_payload:
            self._update_tool_followup_state(session_id, tool_payload)
            return HandleResponse(mode="tool", tool=tool_payload)

        # If no tools were called, return the agent's direct response.
        if not steps:
            message = result.get("output", "") or "未识别到与农事相关的需求。"
            plan = WorkflowResponse(message=message, trace=trace)
            return HandleResponse(mode="none", plan=plan)

        # Fallback: agent responded but no tool/workflow payload detected.
        plan = WorkflowResponse(message=result.get("output", ""), trace=trace)
        return HandleResponse(mode="none", plan=plan)

    def _build_agent(self) -> AgentExecutor:
        llm = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是农事助手。根据用户意图决定是否调用工具或工作流："
                    "简单查询用工具；生育期预测或补充其所需种植信息时，"
                    "必须调用 growth_stage_workflow；需要全流程/多环节方案或"
                    "补充完整种植计划信息时，必须调用 crop_calendar_workflow；"
                    "若无关则直接回答。",
                ),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            return_intermediate_steps=True,
        )

    def _build_crop_workflow_tool(self):
        def _workflow(prompt: str) -> str:
            """
            完整种植计划工作流（抽取→追问→并行工具→推荐）。
            适用：用户要全流程/多环节方案，或在补充作物/品种/播种方式/播期等关键信息时。与种植无关不要调用
            输入：用户原始问题或补充描述；输出：WorkflowResponse 的 JSON 字符串。
            """
            try:
                plan = self._run_crop_calendar_workflow(prompt)
            except MissingPlantingInfoError as exc:
                plan = WorkflowResponse(message=str(exc))
            except Exception as exc:
                plan = WorkflowResponse(message=f"workflow 执行失败: {exc}")
            return json.dumps(
                plan.model_dump(mode="json"), ensure_ascii=True, default=str
            )

        return lc_tool(self.CROP_WORKFLOW_TOOL)(_workflow)

    def _build_growth_stage_workflow_tool(self):
        def _workflow(prompt: str) -> str:
            """
            生育期预测工作流（信息抽取-气象数据-生育期计算）。
            当问题涉及生育期预测或用户在补充相关种植要素时使用。
            """
            try:
                plan = self._run_growth_stage_workflow(prompt)
            except MissingPlantingInfoError as exc:
                plan = WorkflowResponse(message=str(exc))
            except Exception as exc:
                plan = WorkflowResponse(message=f"workflow 执行失败: {exc}")
            return json.dumps(
                plan.model_dump(mode="json"), ensure_ascii=True, default=str
            )

        return lc_tool(self.GROWTH_WORKFLOW_TOOL)(_workflow)

    def _run_crop_calendar_workflow(self, prompt: str) -> WorkflowResponse:
        if self.crop_graph is None:
            self.crop_graph = build_crop_graph()
        return self._run_graph(prompt, self.crop_graph, self.CROP_WORKFLOW_TOOL)

    def _run_growth_stage_workflow(self, prompt: str) -> WorkflowResponse:
        if self.growth_graph is None:
            self.growth_graph = build_growth_stage_graph()
        return self._run_graph(prompt, self.growth_graph, self.GROWTH_WORKFLOW_TOOL)

    def _run_graph(self, prompt: str, graph, workflow_name: str) -> WorkflowResponse:
        session_id = self._session_id_ctx.get()
        initial_state = {"user_prompt": prompt, "trace": []}
        pending = self._pending_followups.get(session_id)
        if pending and pending.get("workflow_name") == workflow_name:
            initial_state.update(
                {
                    "planting_draft": pending.get("planting_draft"),
                    "missing_fields": pending.get("missing_fields"),
                    "followup_count": pending.get("followup_count", 0),
                }
            )
        state = graph.invoke(initial_state)
        self._update_workflow_followup_state(session_id, state, workflow_name)
        return WorkflowResponse(
            query=state.get("query"),
            recommendations=state.get("recommendations", []),
            message=state.get("message", ""),
            trace=state.get("trace", []),
            data=state.get("data", {}),
        )

    def _run_tool_followup(
        self, prompt: str, pending: dict, session_id: str
    ) -> ToolInvocation:
        tool_name = pending.get("tool_name")
        if not tool_name:
            self._pending_followups.pop(session_id, None)
            return ToolInvocation(
                name="unknown_tool",
                message="tool followup missing tool_name",
                data={},
            )
        followup_payload = {
            "followup": {
                "prompt": prompt,
                "draft": pending.get("draft") or {},
                "missing_fields": pending.get("missing_fields") or [],
                "followup_count": pending.get("followup_count", 0),
            }
        }
        followup_prompt = json.dumps(
            followup_payload, ensure_ascii=True, default=str
        )
        result = execute_tool(tool_name, followup_prompt)
        if not result:
            self._pending_followups.pop(session_id, None)
            return ToolInvocation(
                name=tool_name,
                message="tool not found",
                data={},
            )
        self._update_tool_followup_state(session_id, result)
        return result

    def _update_workflow_followup_state(
        self, session_id: str, state: dict, workflow_name: str
    ) -> None:
        missing = state.get("missing_fields") or []
        draft = state.get("planting_draft")
        if missing and draft is not None:
            self._pending_followups[session_id] = {
                "mode": "workflow",
                "workflow_name": workflow_name,
                "planting_draft": draft,
                "missing_fields": missing,
                "followup_count": state.get("followup_count", 0),
            }
        else:
            pending = self._pending_followups.get(session_id)
            if (
                pending
                and pending.get("mode") == "workflow"
                and pending.get("workflow_name") == workflow_name
            ):
                self._pending_followups.pop(session_id, None)

    def _update_tool_followup_state(
        self, session_id: str, tool_payload: ToolInvocation
    ) -> None:
        data = tool_payload.data or {}
        missing = data.get("missing_fields") or []
        draft = data.get("draft")
        if missing and isinstance(draft, dict):
            followup_count = data.get("followup_count", 0)
            self._pending_followups[session_id] = {
                "mode": "tool",
                "tool_name": tool_payload.name,
                "draft": draft,
                "missing_fields": missing,
                "followup_count": followup_count,
            }
        else:
            pending = self._pending_followups.get(session_id)
            if pending and pending.get("mode") == "tool":
                self._pending_followups.pop(session_id, None)

    def _format_trace(self, steps: List[object]) -> List[str]:
        trace: List[str] = []
        for action, observation in steps:
            trace.append(f"tool={action.tool} input={action.tool_input}")
            trace.append(f"observation={observation}")
        return trace

    def _extract_workflow_payload(self, steps: List[object]) -> Optional[dict]:
        for action, observation in reversed(steps):
            if action.tool not in {
                self.CROP_WORKFLOW_TOOL,
                self.GROWTH_WORKFLOW_TOOL,
            }:
                continue
            payload = self._safe_json(observation)
            if isinstance(payload, dict):
                return payload
        return None

    def _extract_tool_payload(self, steps: List[object]) -> Optional[ToolInvocation]:
        for action, observation in reversed(steps):
            if action.tool in {
                self.CROP_WORKFLOW_TOOL,
                self.GROWTH_WORKFLOW_TOOL,
            }:
                continue
            payload = self._safe_json(observation)
            if isinstance(payload, dict) and payload.get("name"):
                return ToolInvocation(**payload)
        return None

    @staticmethod
    def _safe_json(text: object) -> Optional[dict]:
        if not isinstance(text, str):
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
