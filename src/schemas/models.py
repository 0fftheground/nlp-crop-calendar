from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional

from ..domain.enums import PlantingMethod
from ..domain.planting_models import PlantingDetails
from pydantic import BaseModel, Field, field_validator, model_validator


class UserRequest(BaseModel):
    """Incoming request payload from UI clients."""

    prompt: str
    region: Optional[str] = None
    user_id: Optional[str] = Field(
        default=None,
        description="稳定用户标识（如前端生成的 client_id），用于跨会话记忆。",
    )
    session_id: Optional[str] = Field(
        default=None, description="客户端会话标识，用于多用户状态隔离。"
    )


class Recommendation(BaseModel):
    """Single agronomy task recommendation rendered to the client."""

    crop: str
    stage: str
    title: str
    description: str
    reasoning: str
    months: List[str]
    regions: List[str]


class WorkflowResponse(BaseModel):
    """LangGraph workflow output."""

    recommendations: List[Recommendation] = Field(default_factory=list)
    growth_stage: Optional[GrowthStageResult] = None
    message: str = ""
    trace: List[str] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict)


class ToolInvocation(BaseModel):
    """Canonical tool execution payload shared with UI + router."""

    name: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


class HandleResponse(BaseModel):
    """Unified response for both tool and workflow execution paths."""

    mode: Literal["tool", "workflow", "none"]
    tool: Optional[ToolInvocation] = None
    plan: Optional[WorkflowResponse] = None


class PredictGrowthStageInput(BaseModel):
    """Inputs required for the growth stage result query service."""

    weatherSeries:WeatherSeries = Field(...)
    planting: PlantingDetails = Field(
        ...,
        description="标准化后的种植详情，可被不同工具共享。",
    )
    variety_record: Optional[Dict[str, object]] = Field(
        default=None,
        description="可选的数据库品种记录（含审定区域/稻作类型/对照品种等），用于固定匹配结果。",
    )

    @property
    def crop(self) -> str:
        return self.planting.crop

    @property
    def variety(self) -> Optional[str]:
        return self.planting.variety

    @property
    def planting_method(self) -> PlantingMethod:
        return self.planting.planting_method

    @property
    def sowing_date(self) -> date:
        return self.planting.sowing_date



class GrowthStageResult(BaseModel):
    """Result payload returned by the growth stage result query service."""
    stages: Dict[str, str] = Field(default_factory=dict)


def _strip_admin_prefix(value: str) -> str:
    for marker in ("特别行政区", "自治区", "省"):
        if marker in value:
            candidate = value.split(marker)[-1]
            if candidate:
                return candidate
    return value


class WeatherQueryInput(BaseModel):
    """Parameters for querying weather data."""

    region: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=64,
        description="区域名称；weather_lookup 会先按区域表匹配 region_id，再调用天气接口。",
    )
    start_date: date = Field(
        ..., description="查询起始日期（含），格式 YYYY-MM-DD。"
    )
    end_date: date = Field(
        ..., description="查询结束日期（含），格式 YYYY-MM-DD。"
    )
    year: int = Field(
        default_factory=lambda: date.today().year,
        ge=1900,
        le=2100,
        description="查询年份（按自然年返回气象序列）。未提供时使用当前年。",
    )
    granularity: Literal["hourly", "daily"] = "daily"
    include_advice: bool = False
    requested_operations: List[str] = Field(
        default_factory=list,
        description="需要展示的农事适宜度标签，如施肥、打药。",
    )

    @field_validator("region", mode="before")
    @classmethod
    def _normalize_region(cls, value: object) -> object:
        if value is None:
            return value
        text = str(value).strip()
        if not text:
            return text
        text = re.sub(r"\s+", "", text)
        match = re.search(r".+?市", text)
        if match:
            return _strip_admin_prefix(match.group(0))
        match = re.search(r".+?(州|盟|地区)", text)
        if match:
            return _strip_admin_prefix(match.group(0))
        return _strip_admin_prefix(text)

    @model_validator(mode="after")
    def _validate_date_range(self) -> "WeatherQueryInput":
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date")
        days = (self.end_date - self.start_date).days + 1
        if days > 30:
            raise ValueError("date range must be within 30 days")
        self.requested_operations = [
            str(item).strip() for item in self.requested_operations if str(item).strip()
        ]
        return self


class WeatherDataPoint(BaseModel):
    """Single weather observation or forecast data point."""

    timestamp: datetime
    temperature: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    humidity: Optional[float] = None
    precipitation: Optional[float] = None
    wind_speed: Optional[float] = None
    condition: Optional[str] = None
    sf_ws: Optional[float] = None
    sf_reason: Optional[str] = None
    lm_ws: Optional[float] = None
    lm_reason: Optional[str] = None
    yz_ws: Optional[float] = None
    yz_reason: Optional[str] = None
    fd_ws: Optional[float] = None
    fd_reason: Optional[str] = None
    dy_ws: Optional[float] = None
    dy_reason: Optional[str] = None
    sg_ws: Optional[float] = None
    sg_reason: Optional[str] = None
    zd_ws: Optional[float] = None
    zd_reason: Optional[str] = None


class WeatherSeries(BaseModel):
    """Reusable weather sequence with aligned metadata."""

    region: str = Field(..., description="气象序列所属区域或站点。")
    granularity: Literal["hourly", "daily"] = Field(
        default="daily", description="序列粒度：逐日/逐小时。"
    )
    start_date: Optional[date] = Field(
        default=None, description="序列覆盖的起始日期（含）。"
    )
    end_date: Optional[date] = Field(
        default=None, description="序列覆盖的结束日期（含）。"
    )
    points: List[WeatherDataPoint] = Field(default_factory=list)
    source: Optional[str] = Field(
        default=None, description="数据来源，例如自动站或模式。"
    )
    summary: Optional[str] = Field(
        default=None, description="气象摘要（可缓存）。"
    )
    export_file_id: Optional[str] = Field(
        default=None, description="导出的 CSV 文件标识。"
    )
    export_path: Optional[str] = Field(
        default=None, description="导出的 CSV 本地路径。"
    )


class QueryInput(BaseModel):
    """Generic query wrapper for tool invocation."""

    query: str = Field(
        ...,
        min_length=1,
        description="用户查询内容或原始问题。",
    )


class PlanTaskCreateInput(BaseModel):
    """Structured input for creating or recording tasks under an existing plan."""

    query: str = Field(
        ...,
        min_length=1,
        description="用户原始问题或当前追问回复。",
    )
    followup: Optional[Dict[str, Any]] = Field(
        default=None,
        description="可选的追问上下文，用于携带已有 draft、选项或计划 ID。",
    )


class SowingSuitabilityQueryInput(BaseModel):
    """Structured input for sowing suitability lookup."""

    query: str = Field(
        ...,
        min_length=1,
        description="用户原始问题，需包含品种、稻作类型、播种方式及区域或使用默认农场。",
    )
    variety: Optional[str] = None
    culti_type: Optional[str] = None
    planting_method: Optional[str] = None
    region_id: Optional[str] = None
    farm_id: Optional[str] = None
    crop: Optional[str] = None

    @field_validator("region_id", "farm_id", mode="before")
    @classmethod
    def _normalize_sowing_lookup_ids(cls, value: object) -> object:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class PromptInput(BaseModel):
    """Workflow prompt wrapper."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="用户原始问题或补充信息。",
    )


class MemoryClearInput(BaseModel):
    """Input payload for clearing stored memory."""

    reason: Optional[str] = Field(
        default=None,
        description="清除记忆的原因或备注。",
    )


class OperationItem(BaseModel):
    """Single recommended operation in the farm work plan."""

    stage: str
    title: str
    description: str
    dates: List[str] = Field(
        default_factory=list, description="Recommended execution dates from external service."
    )
    reasoning: Optional[str] = None
    window: Optional[str] = Field(
        default=None, description="Suggested execution window or timeframe."
    )
    priority: Literal["low", "medium", "high"] = "medium"


class OperationPlanResult(BaseModel):
    """Result payload returned by the recommendation service."""

    crop: str
    summary: str = ""
    operations: List[OperationItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
