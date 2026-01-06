from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Dict, List, Literal, Optional
from ..domain.enums import PlantingMethod
from ..domain.normalizers import EnumNormalizer
from pydantic import BaseModel, Field, field_validator


class FarmerQuery(BaseModel):
    """Normalized attributes extracted from the farmer's free-form query."""

    crop: Optional[str] = None
    variety: Optional[str] = None
    region: Optional[str] = None
    growth_stage: Optional[str] = Field(
        default=None, description="Canonical stage id (e.g., tillering, fruiting)"
    )
    sowing_date: Optional[date] = None
    question: Optional[str] = None


class UserRequest(BaseModel):
    """Incoming request payload from UI clients."""

    prompt: str
    region: Optional[str] = None
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

    query: Optional[FarmerQuery] = None
    recommendations: List[Recommendation] = Field(default_factory=list)
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


class PlantingDetailsDraft(BaseModel):
    """Raw planting info parsed from free-form user queries before validation."""

    source_text: Optional[str] = Field(
        default=None, description="原始语句或片段，便于 prompt 追溯。"
    )
    crop: Optional[str] = None
    variety: Optional[str] = None
    planting_method: Optional[str] = Field(
        default=None,
        description="自然语言中的种植方式，可包含别名（如直播/插秧）。",
    )
    sowing_date: Optional[date] = None
    transplant_date: Optional[date] = None
    region: Optional[str] = Field(
        default=None, description="行政区域，如省/市/县。"
    )
    planting_location: Optional[str] = None
    notes: Optional[str] = None
    assumptions: List[str] = Field(
        default_factory=list,
        description="若使用默认值或推断补齐，在此记录说明。",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="抽取置信度，方便策略层做兜底。",
    )

    def to_canonical(self) -> "PlantingDetails":
        """
        Convert draft data into the canonical PlantingDetails object.

        Raises:
            ValueError: if required fields remain missing.
        """
        payload = self.model_dump(exclude_none=True)
        required = ["crop", "planting_method", "sowing_date"]
        missing = [field for field in required if payload.get(field) is None]
        if missing:
            raise ValueError(f"Missing required fields for PlantingDetails: {missing}")
        return PlantingDetails(**payload)


class PlantingDetails(BaseModel):
    """Canonical planting context shared by downstream tools."""

    crop: str = Field(
        ...,
        description="作物名称，如水稻",
        examples=["水稻", "小麦"],
    )
    variety: Optional[str] = Field(
        default=None,
        description="品种名称，如美香占 2 号。",
        examples=["美香占2号"],
    )
    planting_method: PlantingMethod = Field(
        ...,
        description="种植方式：direct_seeding=直播，transplanting=移栽。",
        examples=["transplanting"],
    )
    sowing_date: date = Field(
        ...,
        description="播种日期，格式 YYYY-MM-DD。",
        examples=["2025-04-01"],
    )
    transplant_date: Optional[date] = Field(
        default=None,
        description="移栽/插秧日期；直播可留空。",
    )
    region: Optional[str] = Field(
        default=None,
        description="行政区域，如省/市/县。",
    )
    planting_location: Optional[str] = Field(
        default=None,
        description="更精细的地址或地块编号。",
    )

    @field_validator("planting_method", mode="before")
    @classmethod
    def _norm_planting_method(cls, v):
        return EnumNormalizer.normalize(PlantingMethod, v)



class PredictGrowthStageInput(BaseModel):
    """Inputs required for the growth stage prediction service."""

    weatherSeries:WeatherSeries = Field(...)
    planting: PlantingDetails = Field(
        ...,
        description="标准化后的种植详情，可被不同工具共享。",
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

    @property
    def region(self) -> Optional[str]:
        return self.planting.region



class GrowthStageResult(BaseModel):
    """Result payload returned by the growth stage prediction service."""
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

    region: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="市级行政区或站点名称。",
    )
    year: int = Field(
        ...,
        ge=1900,
        le=2100,
        description="查询年份（按自然年返回气象序列）。",
    )
    granularity: Literal["hourly", "daily"] = "daily"
    include_advice: bool = False

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


class WeatherSeriesDraft(BaseModel):
    """Weather intent extracted from user language before hitting data services."""

    source_text: Optional[str] = Field(
        default=None, description="原始语句，用于追踪或重写提示。"
    )
    region: Optional[str] = Field(
        default=None, description="自然语言·地区，如“武汉”或“松滋市”。"
    )
    year: Optional[int] = Field(
        default=None, description="自然语言中的年份。"
    )
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    granularity: Optional[Literal["hourly", "daily"]] = None
    variables: List[str] = Field(
        default_factory=list,
        description="用户提及的要素，如温度/降水/湿度。",
    )
    include_advice: Optional[bool] = None
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="抽取置信度。"
    )

    def to_query(self, *, defaults: Optional["WeatherQueryInput"] = None) -> "WeatherQueryInput":
        """
        Transform the draft intent into a concrete WeatherQueryInput.

        Args:
            defaults: Optional defaults, e.g., from user profile or workflow state.
        Raises:
            ValueError: if region or year is still missing.
        """
        base = defaults.model_dump() if defaults else {}
        merged: Dict[str, Any] = {**base, **self.model_dump(exclude_none=True)}
        year = merged.get("year")
        start_date = merged.get("start_date")
        end_date = merged.get("end_date")
        if year is None:
            if start_date and end_date and start_date.year != end_date.year:
                raise ValueError("start_date and end_date must be in the same year")
            if start_date:
                year = start_date.year
            elif end_date:
                year = end_date.year
            else:
                year = date.today().year
        merged["year"] = year
        required = ["region", "year"]
        missing = [f for f in required if merged.get(f) is None]
        if missing:
            raise ValueError(f"Missing required fields for WeatherQueryInput: {missing}")
        merged.setdefault("granularity", "daily")
        merged.setdefault("include_advice", False)
        allowed = {"region", "year", "granularity", "include_advice"}
        filtered = {k: v for k, v in merged.items() if k in allowed}
        return WeatherQueryInput(**filtered)


class FarmWorkRecommendInput(BaseModel):
    """Input payload for requesting farm operation recommendations."""

    weatherSeries: WeatherSeries
    planting: PlantingDetails = Field(
        default=None,
        description="种植详情，便于推荐引擎共享上下文。",
    )

    @property
    def crop(self) -> str:
        return self.planting.crop


class OperationItem(BaseModel):
    """Single recommended operation in the farm work plan."""

    stage: str
    title: str
    description: str
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
