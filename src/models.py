from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class FarmerQuery(BaseModel):
    crop: str = Field(..., description="Normalized crop name")
    month: Optional[str] = Field(None, description="Calendar month keyword")
    region: Optional[str] = Field(None, description="Geographic region")
    growth_stage: Optional[str] = Field(None, description="Crop growth stage")
    issues: List[str] = Field(default_factory=list)
    requested_actions: List[str] = Field(default_factory=list)
    urgency: Optional[str] = Field(None, description="Urgency or priority")

    def summary(self) -> str:
        parts = [self.crop]
        if self.growth_stage:
            parts.append(self.growth_stage)
        if self.region:
            parts.append(self.region)
        if self.month:
            parts.append(self.month)
        return " / ".join(parts)


class Recommendation(BaseModel):
    title: str
    description: str
    reasoning: str
    resources: Optional[str] = None


class PlanRequest(BaseModel):
    prompt: str = Field(..., description="User natural language request")
    region: Optional[str] = None
    locale: Optional[str] = "zh"


class PlanResponse(BaseModel):
    query: FarmerQuery
    recommendations: List[Recommendation]
    message: str
    trace: List[str] = Field(default_factory=list)


class ToolInvocation(BaseModel):
    name: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class HandleResponse(BaseModel):
    mode: Literal["tool", "workflow"]
    plan: Optional[PlanResponse] = None
    tool: Optional[ToolInvocation] = None
