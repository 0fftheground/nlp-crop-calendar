from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ..infra.llm import get_chat_model
from ..observability.logging_utils import log_event, summarize_text
from .workflows.registry import WorkflowSpec


class ActionPlan(BaseModel):
    action: Literal["tool", "workflow", "none"]
    name: Optional[str] = None
    input: Optional[Any] = None
    response: Optional[str] = None
    reason: Optional[str] = None


class PlannerRunner:
    def __init__(
        self,
        tool_specs: Iterable[Dict[str, str]],
        workflow_specs: Iterable[WorkflowSpec],
    ) -> None:
        self._tool_specs = list(tool_specs)
        self._workflow_specs = list(workflow_specs)
        self._llm = get_chat_model()
        self._system_prompt = self._build_prompt()

    def plan(self, prompt: str, *, pending: Optional[dict] = None) -> Optional[ActionPlan]:
        payload = {
            "prompt": prompt,
            "pending": self._summarize_pending(pending),
        }
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(
                content=json.dumps(payload, ensure_ascii=True, default=str)
            ),
        ]
        log_event(
            "planner_call",
            prompt_summary=summarize_text(prompt),
            pending=payload["pending"],
        )
        try:
            planner = self._llm.with_structured_output(ActionPlan)
            result = planner.invoke(messages)
        except Exception as exc:
            log_event("planner_error", error=str(exc))
            return None
        try:
            plan = (
                result
                if isinstance(result, ActionPlan)
                else ActionPlan.model_validate(result)
            )
        except Exception as exc:
            log_event("planner_error", error=f"invalid_plan:{exc}")
            return None
        log_event(
            "planner_response",
            action=plan.action,
            name=plan.name,
            reason=summarize_text(plan.reason or ""),
            response_summary=summarize_text(plan.response or ""),
        )
        return plan

    def _build_prompt(self) -> str:
        tools_text = "\n".join(
            [f"- {tool['name']}: {tool['description']}" for tool in self._tool_specs]
        ) or "(none)"
        workflows_text = "\n".join(
            [f"- {spec.name}: {spec.description}" for spec in self._workflow_specs]
        ) or "(none)"
        return (
            "你是农事助手的 Planner，负责决定下一步动作。\n"
            "可选动作:\n"
            "- tool: 调用单个工具完成查询或简短建议。\n"
            "- workflow: 进入多步骤流程以补充种植信息并生成方案。\n"
            "- none: 与农事无关或可直接回答。\n\n"
            "判定规则（优先级从上到下，尽量命中工具/流程而不是 none）:\n"
            "- 天气/气象/预报/降雨/气温相关 -> tool: weather_lookup\n"
            "- 生育期预测/当前生育阶段判断（含播种信息） -> workflow: growth_stage_workflow\n"
            "- 仅查询品种信息/特性/抗性/生育期/熟期/审定信息 -> tool: variety_lookup\n"
            "- 需要完整种植计划/全流程/多环节方案 -> workflow: crop_calendar_workflow\n"
            "- 简短农事建议/注意事项（非完整方案） -> tool: farming_recommendation\n\n"
            "如果准备输出 action=none，必须先逐条核对以上规则，确认都不匹配才允许输出 none。\n"
            "输出 none 时必须给出 response，并在 reason 里简要说明不匹配的原因。\n\n"
            "若 pending 不为空，表示当前处于追问流程：\n"
            "- 若用户在补充 missing_fields 或表达不知道/不确定，应继续同一 tool/workflow。\n"
            "- 若 pending.memory_prompted=true 且用户回答沿用/不用，应继续同一 workflow。\n"
            "- 若用户提出与 pending 无关的新问题，应选择新的 tool/workflow 或 none。\n\n"
            "工具列表:\n"
            f"{tools_text}\n\n"
            "工作流列表:\n"
            f"{workflows_text}\n\n"
            "输出 ActionPlan JSON 字段:\n"
            "- action: tool|workflow|none\n"
            "- name: 当 action 为 tool/workflow 时，必须是列表中的名称\n"
            "- input: 可选，若需要结构化输入可给对象/字符串；否则留空\n"
            "- response: 当 action 为 none 时给出回答\n"
            "- reason: 可选简短理由\n"
        )

    @staticmethod
    def _summarize_pending(pending: Optional[dict]) -> Optional[dict]:
        if not isinstance(pending, dict):
            return None
        summary: Dict[str, Any] = {
            "mode": pending.get("mode"),
            "name": pending.get("tool_name") or pending.get("workflow_name"),
            "missing_fields": pending.get("missing_fields"),
            "followup_count": pending.get("followup_count"),
        }
        if "memory_prompted" in pending:
            summary["memory_prompted"] = pending.get("memory_prompted")
        return summary
