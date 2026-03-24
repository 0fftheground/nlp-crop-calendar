from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .crop_calendar_graph import build_crop_calendar_graph


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    description: str
    builder: Callable[[], object]


CROP_WORKFLOW_NAME = "crop_calendar_workflow"

_WORKFLOWS = (
    WorkflowSpec(
        name=CROP_WORKFLOW_NAME,
        description=(
            "完整种植计划工作流（抽取→追问→外部计算→推荐）。"
            "适用：用户要全流程/多环节方案，或在补充作物/品种/播种方式/播期等关键信息时。"
            "与种植无关不要调用"
        ),
        builder=build_crop_calendar_graph,
    ),
)
_WORKFLOW_INDEX: Dict[str, WorkflowSpec] = {spec.name: spec for spec in _WORKFLOWS}


def list_workflow_specs() -> List[WorkflowSpec]:
    """返回所有可路由工作流，供上层做选择或生成提示。"""
    return list(_WORKFLOWS)


def get_workflow_spec(name: str) -> Optional[WorkflowSpec]:
    """按名称取工作流定义，随后由 builder 构建实际 LangGraph。"""
    return _WORKFLOW_INDEX.get(name)
