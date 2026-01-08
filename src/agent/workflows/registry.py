from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .crop_graph import build_graph as build_crop_graph
from .crop_graph import build_growth_stage_graph


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    description: str
    builder: Callable[[], object]


CROP_WORKFLOW_NAME = "crop_calendar_workflow"
GROWTH_WORKFLOW_NAME = "growth_stage_workflow"

_WORKFLOWS = (
    WorkflowSpec(
        name=CROP_WORKFLOW_NAME,
        description=(
            "完整种植计划工作流（抽取→追问→并行工具→推荐）。"
            "适用：用户要全流程/多环节方案，或在补充作物/品种/播种方式/播期等关键信息时。"
            "与种植无关不要调用"
        ),
        builder=build_crop_graph,
    ),
    WorkflowSpec(
        name=GROWTH_WORKFLOW_NAME,
        description=(
            "生育期预测工作流（信息抽取-气象数据-生育期计算）。"
            "当问题涉及生育期预测或用户在补充相关种植要素时使用。"
        ),
        builder=build_growth_stage_graph,
    ),
)
_WORKFLOW_INDEX: Dict[str, WorkflowSpec] = {spec.name: spec for spec in _WORKFLOWS}


def list_workflow_specs() -> List[WorkflowSpec]:
    return list(_WORKFLOWS)


def get_workflow_spec(name: str) -> Optional[WorkflowSpec]:
    return _WORKFLOW_INDEX.get(name)
