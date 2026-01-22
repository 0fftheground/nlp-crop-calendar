"""
LangGraph state scaffolding.

Define your own TypedDict (or Pydantic model) to describe the state shared between
workflow nodes. Remove the placeholders below once you finalize the fields you need.
"""

from typing import Dict, List, Optional, TypedDict

from ...schemas import (
    GrowthStageResult,
    PlantingDetails,
    PlantingDetailsDraft,
    Recommendation,
)


class GraphState(TypedDict, total=False):
    """State shared across crop calendar workflow nodes."""

    user_id: str
    user_prompt: str
    trace: List[str]
    planting_draft: PlantingDetailsDraft
    planting: PlantingDetails
    missing_fields: List[str]
    followup_count: int
    assumptions: List[str]
    cache_hit: bool
    weather_info: Dict[str, object]
    weather_series_ref: str
    growth_stage: GrowthStageResult
    variety_info: Dict[str, object]
    recommendation_info: Dict[str, object]
    recommendations: List[Recommendation]
    message: str
    data: Dict[str, object]
    experience_key: Optional[str]
    experience_applied: List[str]
    experience_skip_fields: List[str]
    experience_notice: Optional[str]
    pending_options: List[str]
    pending_message: Optional[str]
    variety_candidates: List[str]
    variety_record: Optional[Dict[str, object]]
    variety_tool_query: Optional[str]
    variety_tool_draft: Optional[Dict[str, object]]
    variety_tool_missing_fields: List[str]
    variety_tool_followup_count: int
    future_sowing_date_warning: bool
    halt: bool


def add_trace(state: GraphState, message: str) -> GraphState:
    """Append a message to the workflow trace."""
    trace = list(state.get("trace") or [])
    trace.append(message)
    return {**state, "trace": trace}
