"""
LangGraph workflow scaffolding.

The previous demo nodes (analysis/plan/finalize) have been intentionally removed so
that you can design the crop calendar workflow from scratch. Define your own state
transformations, register them on a `StateGraph`, and return the compiled graph from
`build_graph()`.
"""

from langgraph.graph import StateGraph

from .state import GraphState


def build_graph():
    """
    Construct and return your LangGraph workflow.

    Example usage:

        graph = StateGraph(GraphState)
        # graph.add_node("your_node", your_handler)
        # graph.set_entry_point("your_node")
        # graph.add_edge("your_node", END)
        return graph.compile()

    Replace this stub with your actual implementation.
    """
    raise NotImplementedError("请在 src/workflows/crop_graph.py 中实现自定义 LangGraph workflow。")
