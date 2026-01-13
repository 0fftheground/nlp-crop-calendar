from __future__ import annotations

from ...domain.planting import DEFAULT_CROP
from ...infra.config import get_config
from ...infra.tool_provider import maybe_intranet_tool, normalize_provider
from ...schemas.models import OperationItem, OperationPlanResult, ToolInvocation


def _infer_crop(prompt: str) -> str:
    if not prompt:
        return DEFAULT_CROP
    crop_keywords = ["水稻", "小麦", "玉米", "大豆", "油菜", "棉花", "花生"]
    return next((item for item in crop_keywords if item in prompt), DEFAULT_CROP)


def recommend_farming(prompt: str) -> ToolInvocation:
    cfg = get_config()
    provider = normalize_provider(cfg.recommendation_provider)
    intranet = maybe_intranet_tool(
        "farming_recommendation",
        prompt,
        provider,
        cfg.recommendation_api_url,
        cfg.recommendation_api_key,
    )
    if intranet:
        return intranet
    crop = _infer_crop(prompt)
    ops = [
        OperationItem(
            stage="field_preparation",
            title="清沟排水",
            description="播种前疏通田间沟系，避免积水。",
            window="播种前 7 天",
            priority="medium",
        ),
        OperationItem(
            stage="seedling",
            title="查苗补苗",
            description="出苗后 10-15 天查苗，缺株处补播。",
            window="出苗后 10 天",
            priority="high",
        ),
        OperationItem(
            stage="fertilization",
            title="分蘖肥",
            description="分蘖期追施氮肥 5-8 公斤/亩。",
            window="出苗后 20-30 天",
            priority="medium",
        ),
    ]
    plan = OperationPlanResult(
        crop=crop,
        summary=f"{crop} 农事推荐（mock 数据）。",
        operations=ops,
        metadata={"source": "mock"},
    )
    return ToolInvocation(
        name="farming_recommendation",
        message="已返回模拟农事推荐。",
        data=plan.model_dump(mode="json"),
    )
