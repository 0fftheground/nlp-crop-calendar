"""
Pydantic schema scaffolding.

The previous concrete models have been removed so that you can redefine the request /
response payloads from scratch. Replace the placeholder classes below with the fields
that match your new workflow and tool contract.
"""

from pydantic import BaseModel


class FarmerQuery(BaseModel):
    """TODO: define normalized farmer query schema."""
    pass


class Recommendation(BaseModel):
    """TODO: define recommendation payload schema."""
    pass


class PlanRequest(BaseModel):
    """TODO: define planner request schema."""
    pass


class PlanResponse(BaseModel):
    """TODO: define planner response schema."""
    pass


class ToolInvocation(BaseModel):
    """TODO: define tool invocation response schema."""
    pass


class HandleResponse(BaseModel):
    """TODO: define router response schema."""
    pass
