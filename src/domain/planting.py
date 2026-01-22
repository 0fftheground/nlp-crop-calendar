from __future__ import annotations

import re
from datetime import date
from typing import Callable, Dict, List, Optional

from ..observability.logging_utils import log_event
from ..schemas import PlantingDetails, PlantingDetailsDraft


DEFAULT_CROP = "水稻"

CROP_REQUIRED_FIELDS = ["crop", "variety", "planting_method", "sowing_date"]

CROP_KEYWORDS = ["水稻", "小麦", "玉米", "大豆", "油菜", "棉花", "花生"]
METHOD_KEYWORDS = {
    "直播": "direct_seeding",
    "撒播": "direct_seeding",
    "条播": "direct_seeding",
    "移栽": "transplanting",
    "插秧": "transplanting",
    "机插": "transplanting",
}
DATE_PATTERN = re.compile(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})")
_REGION_SUFFIX_RE = re.compile(
    r"[\u4e00-\u9fff]{2,8}(?:省|市|州|盟|地区|区|县)"
)
_REGION_INFIX_RE = re.compile(
    r"(?:在|于|位于)([\u4e00-\u9fff]{2,8})(?:种植|种|播种|直播|栽培)"
)

VarietyResolver = Callable[[str], List[str]]


class MissingPlantingInfoError(ValueError):
    """Raised when required planting fields are missing."""

    def __init__(self, missing_fields: List[str]):
        self.missing_fields = missing_fields
        message = (
            "缺少关键种植信息，请向用户追问："
            + "、".join(missing_fields)
            + "。若用户明确表示不知道，可允许使用默认值。"
        )
        super().__init__(message)


def list_missing_required_fields(draft: PlantingDetailsDraft) -> List[str]:
    payload = draft.model_dump(exclude_none=True)
    return [field for field in CROP_REQUIRED_FIELDS if payload.get(field) is None]


def merge_planting_answers(
    draft: PlantingDetailsDraft,
    *,
    answers: Optional[Dict[str, object]] = None,
    unknown_fields: Optional[List[str]] = None,
    fallback: Optional[PlantingDetails] = None,
) -> PlantingDetailsDraft:
    """
    Merge structured user answers back into the draft and apply defaults when necessary.
    """
    data = draft.model_dump()
    assumptions = list(data.get("assumptions") or [])

    if answers:
        for key, value in answers.items():
            if value is None:
                continue
            if key in PlantingDetailsDraft.model_fields:
                data[key] = value

    if unknown_fields:
        if fallback is None:
            raise MissingPlantingInfoError(unknown_fields)
        for field in unknown_fields:
            default_value = getattr(fallback, field, None)
            if default_value is None:
                raise MissingPlantingInfoError([field])
            data[field] = default_value
            assumptions.append(f"{field}: 用户不知道，使用默认值 {default_value}")

    data["assumptions"] = assumptions
    return PlantingDetailsDraft(**data)


def _infer_variety_from_prompt(
    prompt: str, variety_resolver: Optional[VarietyResolver]
) -> Optional[str]:
    if not prompt or variety_resolver is None:
        return None
    candidates = variety_resolver(prompt)
    return candidates[0] if candidates else None


def _infer_region_from_prompt(prompt: str) -> Optional[str]:
    if not prompt:
        return None
    match = _REGION_SUFFIX_RE.search(prompt)
    if match:
        return match.group(0)
    match = _REGION_INFIX_RE.search(prompt)
    if match:
        candidate = match.group(1)
        if candidate and candidate not in CROP_KEYWORDS:
            return candidate
    return None


def _is_known_variety(
    candidate: str, variety_resolver: Optional[VarietyResolver]
) -> bool:
    if not candidate or variety_resolver is None:
        return False
    candidates = variety_resolver(candidate)
    return bool(candidates) and candidates[0] == candidate


def _apply_heuristics(
    data: Dict[str, object],
    prompt: str,
    variety_resolver: Optional[VarietyResolver],
) -> None:
    if "crop" not in data:
        for crop in CROP_KEYWORDS:
            if crop in prompt:
                data["crop"] = crop
                break

    if "planting_method" not in data:
        for alias, canonical in METHOD_KEYWORDS.items():
            if alias in prompt:
                data["planting_method"] = canonical
                break

    if "sowing_date" not in data:
        date_match = DATE_PATTERN.search(prompt)
        if date_match:
            year, month, day = date_match.groups()
            month = month.zfill(2)
            day = day.zfill(2)
            iso_string = f"{year}-{month}-{day}"
            data["sowing_date"] = date.fromisoformat(iso_string)

    if "region" not in data:
        region = _infer_region_from_prompt(prompt)
        if region:
            data["region"] = region

    if "variety" not in data:
        variety = _infer_variety_from_prompt(prompt, variety_resolver)
        if variety:
            data["variety"] = variety


def _normalize_variety_field(
    data: Dict[str, object],
    prompt: str,
    variety_resolver: Optional[VarietyResolver],
) -> None:
    if not variety_resolver or not prompt:
        return
    candidate = _infer_variety_from_prompt(prompt, variety_resolver)
    if not candidate:
        return
    current = data.get("variety")
    if not isinstance(current, str) or not current.strip():
        return
    current = current.strip()
    if candidate != current and candidate in prompt and current not in prompt:
        data["variety"] = candidate


def _sanitize_crop_field(
    data: Dict[str, object],
    prompt: str,
    variety_resolver: Optional[VarietyResolver],
) -> None:
    crop = data.get("crop")
    if not isinstance(crop, str):
        return
    candidate = crop.strip()
    if not candidate:
        data.pop("crop", None)
        return
    if candidate in CROP_KEYWORDS:
        return
    variety = data.get("variety")
    if isinstance(variety, str) and variety.strip() == candidate:
        data.pop("crop", None)
        return
    if _is_known_variety(candidate, variety_resolver):
        if not data.get("variety"):
            data["variety"] = candidate
        data.pop("crop", None)
        return
    if candidate not in prompt:
        data.pop("crop", None)


def _normalize_region_field(data: Dict[str, object]) -> None:
    region = data.get("region")
    if not isinstance(region, str):
        return
    text = region.strip()
    if not text:
        data.pop("region", None)
        return
    text = re.sub(r"^(?:我要在|我在|在|于|位于)", "", text)
    match = _REGION_SUFFIX_RE.search(text)
    if match:
        data["region"] = match.group(0)
        return
    match = _REGION_INFIX_RE.search(text)
    if match:
        candidate = match.group(1)
        if candidate and candidate not in CROP_KEYWORDS:
            data["region"] = candidate


def _apply_rice_default(data: Dict[str, object]) -> None:
    if data.get("crop"):
        return
    data["crop"] = DEFAULT_CROP


def extract_planting_details(
    prompt: str,
    *,
    llm_extract: Optional[Callable[[str], Dict[str, object]]] = None,
    variety_resolver: Optional[VarietyResolver] = None,
) -> PlantingDetailsDraft:
    """
    调用真实 LLM 抽取种植信息；若没有提供 llm_extract，则退回到启发式解析。
    """
    data: Dict[str, object] = {"source_text": prompt}

    if llm_extract is not None:
        raw_payload = llm_extract(prompt)
        if not isinstance(raw_payload, dict):
            raise TypeError("llm_extract 必须返回包含字段的 dict。")
        data.update(raw_payload)
        _apply_heuristics(data, prompt, variety_resolver)
        _normalize_variety_field(data, prompt, variety_resolver)
        _sanitize_crop_field(data, prompt, variety_resolver)
        _normalize_region_field(data)
        _apply_rice_default(data)
        data.pop("variety", None)
        data.setdefault("confidence", 0.9)
        return PlantingDetailsDraft(**data)

    _apply_heuristics(data, prompt, variety_resolver)
    _sanitize_crop_field(data, prompt, variety_resolver)
    _normalize_region_field(data)
    _apply_rice_default(data)
    data.pop("variety", None)
    data["confidence"] = 0.4 if len(data) == 1 else 0.75
    return PlantingDetailsDraft(**data)


def normalize_and_validate_planting(draft: PlantingDetailsDraft) -> PlantingDetails:
    """
    将完成的 Draft 转换为严格的 PlantingDetails。
    """
    missing = list_missing_required_fields(draft)
    if missing:
        raise MissingPlantingInfoError(missing)

    try:
        planting = draft.to_canonical()
        log_event(
            "normalized_planting",
            planting=planting.model_dump(mode="json"),
        )
        return planting
    except ValueError as exc:
        raise ValueError(f"种植信息校验失败: {exc}") from exc
