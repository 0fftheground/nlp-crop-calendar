from typing import List, TypedDict

from ..models import FarmerQuery, Recommendation


class GraphState(TypedDict, total=False):
    user_prompt: str
    query: FarmerQuery
    recommendations: List[Recommendation]
    trace: List[str]


def add_trace(state: GraphState, message: str) -> GraphState:
    trace = list(state.get("trace", []))
    trace.append(message)
    state["trace"] = trace
    return state
