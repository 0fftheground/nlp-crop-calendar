import contextvars
import json
from typing import List, Optional

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool as lc_tool

from ..domain.services import MissingPlantingInfoError
from ..infra.llm import get_chat_model
from ..infra.pending_store import build_pending_followup_store
from ..infra.preference_store import build_preference_store
from ..observability.logging_utils import log_event, summarize_text
from ..schemas.models import (
    HandleResponse,
    PlantingDetailsDraft,
    PlantingDetails,
    ToolInvocation,
    UserRequest,
    WorkflowResponse,
)
from .tools.registry import build_agent_tools, execute_tool
from .intent_rules import ControlIntentRouter, classify_intent
from .workflows.registry import (
    CROP_WORKFLOW_NAME,
    GROWTH_WORKFLOW_NAME,
    get_workflow_spec,
    list_workflow_specs,
)


class RequestRouter:
    """Use a LangChain agent to route tool calls and craft responses."""

    _session_id_ctx = contextvars.ContextVar("session_id", default="default")
    _MEMORY_CLEAR_KEYWORDS = (
        "清除记忆",
        "清空记忆",
        "删除记忆",
        "清除偏好",
        "清空偏好",
        "删除偏好",
        "忘记之前",
        "忘掉之前",
        "清掉记忆",
    )

    def __init__(self):
        self._workflow_specs = list_workflow_specs()
        self._workflow_names = {spec.name for spec in self._workflow_specs}
        self._workflow_graphs: dict[str, object] = {}
        self.tools = build_agent_tools() + self._build_workflow_tools()
        self.agent = self._build_agent()
        self._pending_store = build_pending_followup_store()
        self._control_router = ControlIntentRouter()
        self._preference_store = build_preference_store()

    def handle(self, request: UserRequest) -> HandleResponse:
        session_id = request.session_id or "default"
        token = self._session_id_ctx.set(session_id)
        prompt = request.prompt
        try:
            if self._should_clear_memory(prompt):
                self._preference_store.delete(session_id)
                pending = self._pending_store.get(session_id)
                if pending and pending.get("mode") == "workflow":
                    pending["memory_decision"] = None
                    pending["memory_prompted"] = False
                    self._pending_store.set(session_id, pending)
                prompt = self._strip_memory_clear(prompt)
                if not prompt:
                    plan = WorkflowResponse(message="已清除记忆。")
                    return HandleResponse(mode="none", plan=plan)
            # If the session is mid follow-up, skip routing and resume directly.
            pending = self._pending_store.get(session_id)
            if pending:
                control = self._control_router.route(prompt, pending=pending)
                if control in {"cancel", "cancel_only"}:
                    self._pending_store.delete(session_id)
                    if control == "cancel_only":
                        plan = WorkflowResponse(
                            message="已取消追问，如需继续请描述新的问题。"
                        )
                        return HandleResponse(mode="none", plan=plan)
                    pending = None
                elif control == "new_question":
                    self._pending_store.delete(session_id)
                    pending = None
            if pending:
                if pending.get("mode") == "tool":
                    tool_payload = self._run_tool_followup(
                        prompt, pending, session_id
                    )
                    return HandleResponse(mode="tool", tool=tool_payload)
                workflow_name = pending.get("workflow_name")
                plan = self._run_named_workflow(prompt, workflow_name)
                return HandleResponse(mode="workflow", plan=plan)
            # No pending state: let the LLM agent decide tool vs workflow.
            log_event("llm_router_call", prompt=prompt)
            result = self.agent.invoke({"input": prompt})
        finally:
            self._session_id_ctx.reset(token)
        steps = result.get("intermediate_steps", [])
        log_event(
            "llm_router_response",
            output_summary=summarize_text(result.get("output", "")),
            steps_count=len(steps),
        )
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

    @classmethod
    def _should_clear_memory(cls, prompt: str) -> bool:
        text = (prompt or "").strip()
        if not text:
            return False
        return any(keyword in text for keyword in cls._MEMORY_CLEAR_KEYWORDS)

    @classmethod
    def _strip_memory_clear(cls, prompt: str) -> str:
        text = prompt or ""
        for keyword in cls._MEMORY_CLEAR_KEYWORDS:
            text = text.replace(keyword, "")
        return text.strip(" \t,.;:!?，。！？；：")

    def _build_agent(self) -> AgentExecutor:
        llm = get_chat_model()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "你是农事助手。根据用户意图决定是否调用工具或工作流："
                    "简单查询用工具；生育期预测或补充其所需种植信息时，"
                    f"必须调用 {GROWTH_WORKFLOW_NAME}；需要全流程/多环节方案或"
                    f"补充完整种植计划信息时，必须调用 {CROP_WORKFLOW_NAME}；"
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

    def _build_workflow_tools(self) -> List[object]:
        tools: List[object] = []

        for spec in self._workflow_specs:
            tools.append(self._build_workflow_tool(spec))
        return tools

    def _build_workflow_tool(self, spec) -> object:
        def _workflow(prompt: str) -> str:
            """
            输入：用户原始问题或补充描述；输出：WorkflowResponse 的 JSON 字符串。
            """
            try:
                plan = self._run_named_workflow(prompt, spec.name)
            except MissingPlantingInfoError as exc:
                plan = WorkflowResponse(message=str(exc))
            except Exception as exc:
                plan = WorkflowResponse(message=f"workflow 执行失败: {exc}")
            return json.dumps(
                plan.model_dump(mode="json"), ensure_ascii=True, default=str
            )

        _workflow.__doc__ = spec.description
        return lc_tool(spec.name)(_workflow)

    def _run_named_workflow(self, prompt: str, workflow_name: Optional[str]) -> WorkflowResponse:
        if not workflow_name:
            return WorkflowResponse(message="workflow_name 缺失，无法执行。")
        graph = self._workflow_graphs.get(workflow_name)
        if graph is None:
            spec = get_workflow_spec(workflow_name)
            if spec is None:
                return WorkflowResponse(message=f"workflow 未注册: {workflow_name}")
            graph = spec.builder()
            self._workflow_graphs[workflow_name] = graph
        return self._run_graph(prompt, graph, workflow_name)

    def _run_graph(self, prompt: str, graph, workflow_name: str) -> WorkflowResponse:
        session_id = self._session_id_ctx.get()
        initial_state = {"user_prompt": prompt, "trace": []}
        pending = self._pending_store.get(session_id)
        should_use_memory = False
        if pending and pending.get("workflow_name") == workflow_name:
            should_use_memory = True
        else:
            intent_mode, _ = classify_intent(prompt)
            should_use_memory = intent_mode != "tool"
        if should_use_memory:
            memory = self._preference_store.get(session_id)
            if memory:
                initial_state["memory_planting"] = memory.planting
        if pending and pending.get("workflow_name") == workflow_name:
            initial_state.update(
                {
                    "planting_draft": pending.get("planting_draft"),
                    "missing_fields": pending.get("missing_fields"),
                    "followup_count": pending.get("followup_count", 0),
                    "memory_decision": pending.get("memory_decision"),
                    "memory_prompted": pending.get("memory_prompted", False),
                }
            )
        state = graph.invoke(initial_state)
        planting = state.get("planting")
        if isinstance(planting, PlantingDetails):
            self._preference_store.set(session_id, planting)
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
            self._pending_store.delete(session_id)
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
            self._pending_store.delete(session_id)
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
        draft_payload = None
        if isinstance(draft, PlantingDetailsDraft):
            draft_payload = draft.model_dump(mode="json")
        elif isinstance(draft, dict):
            try:
                draft_payload = PlantingDetailsDraft.model_validate(draft).model_dump(
                    mode="json"
                )
            except Exception:
                draft_payload = draft
        if missing and isinstance(draft_payload, dict):
            payload = {
                "mode": "workflow",
                "workflow_name": workflow_name,
                "planting_draft": draft_payload,
                "missing_fields": missing,
                "followup_count": state.get("followup_count", 0),
                "memory_decision": state.get("memory_decision"),
                "memory_prompted": state.get("memory_prompted", False),
            }
            self._pending_store.set(session_id, payload)
        else:
            pending = self._pending_store.get(session_id)
            if (
                pending
                and pending.get("mode") == "workflow"
                and pending.get("workflow_name") == workflow_name
            ):
                self._pending_store.delete(session_id)

    def _update_tool_followup_state(
        self, session_id: str, tool_payload: ToolInvocation
    ) -> None:
        data = tool_payload.data or {}
        missing = data.get("missing_fields") or []
        draft = data.get("draft")
        if missing and isinstance(draft, dict):
            followup_count = data.get("followup_count", 0)
            self._pending_store.set(
                session_id,
                {
                    "mode": "tool",
                    "tool_name": tool_payload.name,
                    "draft": draft,
                    "missing_fields": missing,
                    "followup_count": followup_count,
                },
            )
        else:
            pending = self._pending_store.get(session_id)
            if pending and pending.get("mode") == "tool":
                self._pending_store.delete(session_id)

    def _format_trace(self, steps: List[object]) -> List[str]:
        trace: List[str] = []
        for action, observation in steps:
            trace.append(f"tool={action.tool} input={action.tool_input}")
            trace.append(f"observation={observation}")
        return trace

    def _extract_workflow_payload(self, steps: List[object]) -> Optional[dict]:
        for action, observation in reversed(steps):
            if action.tool not in self._workflow_names:
                continue
            payload = self._safe_json(observation)
            if isinstance(payload, dict):
                return payload
        return None

    def _extract_tool_payload(self, steps: List[object]) -> Optional[ToolInvocation]:
        for action, observation in reversed(steps):
            if action.tool in self._workflow_names:
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
