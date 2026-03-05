from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from .enums import PlantingMethod
from .normalizers import EnumNormalizer


class PlantingDetailsDraft(BaseModel):
    """Raw planting info parsed from free-form user queries before validation."""

    farm_id: Optional[str] = Field(
        default=None,
        description="农场 ID（可选），用于绑定具体农场上下文。",
        examples=["1", "10001"],
    )
    region_id: Optional[str] = Field(
        default=None,
        description="区域 ID（可选）；也可暂存用户提供的区域名称，后续再解析成 ID。",
        examples=["320100", "4301"],
    )
    source_text: Optional[str] = Field(
        default=None, description="原始语句或片段，便于 prompt 追溯。"
    )
    crop: Optional[str] = None
    variety: Optional[str] = None
    culti_type: Optional[str] = Field(
        default=None,
        description="稻作类型/熟制，如早稻、双季晚稻等。",
    )
    planting_method: Optional[str] = Field(
        default=None,
        description="自然语言中的种植方式，可包含别名（如直播/插秧）。",
    )
    sowing_date: Optional[date] = None
    transplant_date: Optional[date] = None
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

    farm_id: Optional[str] = Field(
        default=None,
        description="农场 ID（可选），用于绑定具体农场上下文。",
        examples=["1", "10001"],
    )
    region_id: Optional[str] = Field(
        default=None,
        description="区域 ID（可选）；也可暂存用户提供的区域名称，后续再解析成 ID。",
        examples=["320100", "4301"],
    )
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
    culti_type: Optional[str] = Field(
        default=None,
        description="稻作类型/熟制，如早稻、双季晚稻等。",
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

    @field_validator("planting_method", mode="before")
    @classmethod
    def _norm_planting_method(cls, value):
        return EnumNormalizer.normalize(PlantingMethod, value)


__all__ = ["PlantingDetailsDraft", "PlantingDetails"]
