"""
LangGraph state scaffolding.

Define your own TypedDict (or Pydantic model) to describe the state shared between
workflow nodes. Remove the placeholders below once you finalize the fields you need.
"""

from typing import TypedDict


class GraphState(TypedDict, total=False):
    """TODO: define workflow state fields."""
    pass


def add_trace(state: GraphState, message: str) -> GraphState:
    """TODO: implement trace helper if needed."""
    raise NotImplementedError("请根据新的 GraphState 实现 add_trace。")
