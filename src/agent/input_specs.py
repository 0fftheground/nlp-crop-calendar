from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Type

from pydantic import BaseModel

from ..schemas import MemoryClearInput, PromptInput, QueryInput, WeatherQueryInput


@dataclass(frozen=True)
class InputSpec:
    model: Type[BaseModel]
    field_labels: Dict[str, str]
    to_prompt: Callable[[BaseModel], str]

    @property
    def required_fields(self) -> Sequence[str]:
        return [
            name
            for name, field in self.model.model_fields.items()
            if field.is_required()
        ]


def _to_json_payload(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), ensure_ascii=True, default=str)


WEATHER_FIELD_LABELS = {
    "region": "地区(省/市/区/县/站点)",
    "year": "年份(默认当前年)",
    "granularity": "粒度(hourly/daily)",
    "include_advice": "是否包含建议",
}
QUERY_FIELD_LABELS = {
    "query": "查询内容",
}
PROMPT_FIELD_LABELS = {
    "prompt": "用户问题",
}
MEMORY_CLEAR_FIELD_LABELS = {
    "reason": "清除原因(可选)",
}

TOOL_INPUT_SPECS: Dict[str, InputSpec] = {
    "weather_lookup": InputSpec(
        model=WeatherQueryInput,
        field_labels=WEATHER_FIELD_LABELS,
        to_prompt=_to_json_payload,
    ),
    "variety_lookup": InputSpec(
        model=QueryInput,
        field_labels=QUERY_FIELD_LABELS,
        to_prompt=_to_json_payload,
    ),
    "farming_recommendation": InputSpec(
        model=QueryInput,
        field_labels=QUERY_FIELD_LABELS,
        to_prompt=_to_json_payload,
    ),
    "memory_clear": InputSpec(
        model=MemoryClearInput,
        field_labels=MEMORY_CLEAR_FIELD_LABELS,
        to_prompt=_to_json_payload,
    ),
}

WORKFLOW_INPUT_SPECS: Dict[str, InputSpec] = {
    "crop_calendar_workflow": InputSpec(
        model=PromptInput,
        field_labels=PROMPT_FIELD_LABELS,
        to_prompt=lambda payload: str(payload.prompt),
    ),
    "growth_stage_workflow": InputSpec(
        model=PromptInput,
        field_labels=PROMPT_FIELD_LABELS,
        to_prompt=lambda payload: str(payload.prompt),
    ),
}


def get_input_spec(action: str, name: str) -> Optional[InputSpec]:
    if action == "tool":
        return TOOL_INPUT_SPECS.get(name)
    if action == "workflow":
        return WORKFLOW_INPUT_SPECS.get(name)
    return None


def format_input_schema(spec: InputSpec) -> str:
    parts = []
    for name, field in spec.model.model_fields.items():
        label = spec.field_labels.get(name, name)
        required = "必填" if field.is_required() else "可选"
        parts.append(f"{name}({label},{required})")
    joined = "；".join(parts)
    return f"{spec.model.__name__}{{{joined}}}"
