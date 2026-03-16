from __future__ import annotations

import re
from typing import List

_REGION_SUFFIX_RE = re.compile(r"(特别行政区|自治区|自治州|省|市|州|盟|地区|区|县)$")
_REGION_BOUNDARY_RE = re.compile(r"(特别行政区|自治区|自治州|地区|省|市|州|盟)")
_COUNTRY_PREFIXES = (
    "中华人民共和国",
    "中国",
)
_PROVINCE_PREFIXES = (
    "内蒙古自治区",
    "广西壮族自治区",
    "西藏自治区",
    "宁夏回族自治区",
    "新疆维吾尔自治区",
    "香港特别行政区",
    "澳门特别行政区",
    "内蒙古",
    "广西",
    "西藏",
    "宁夏",
    "新疆",
    "香港",
    "澳门",
    "北京市",
    "天津市",
    "上海市",
    "重庆市",
    "北京",
    "天津",
    "上海",
    "重庆",
    "河北省",
    "山西省",
    "辽宁省",
    "吉林省",
    "黑龙江省",
    "江苏省",
    "浙江省",
    "安徽省",
    "福建省",
    "江西省",
    "山东省",
    "河南省",
    "湖北省",
    "湖南省",
    "广东省",
    "海南省",
    "四川省",
    "贵州省",
    "云南省",
    "陕西省",
    "甘肃省",
    "青海省",
    "台湾省",
    "河北",
    "山西",
    "辽宁",
    "吉林",
    "黑龙江",
    "江苏",
    "浙江",
    "安徽",
    "福建",
    "江西",
    "山东",
    "河南",
    "湖北",
    "湖南",
    "广东",
    "海南",
    "四川",
    "贵州",
    "云南",
    "陕西",
    "甘肃",
    "青海",
    "台湾",
)


def normalize_region_token(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[，。；、,.!！?？\s]+", "", text)


def build_region_text_variants(value: object) -> List[str]:
    normalized = normalize_region_token(value)
    if not normalized:
        return []
    variants: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        text = normalize_region_token(candidate)
        if not text or text in seen:
            return
        seen.add(text)
        variants.append(text)
        trimmed = _REGION_SUFFIX_RE.sub("", text)
        if trimmed and trimmed not in seen:
            seen.add(trimmed)
            variants.append(trimmed)

    def walk(candidate: str) -> None:
        text = normalize_region_token(candidate)
        if not text:
            return
        add(text)
        for match in _REGION_BOUNDARY_RE.finditer(text):
            prefix = text[: match.end()]
            if len(prefix) >= 2:
                add(prefix)
            if match.end() < len(text):
                remainder = text[match.end() :]
                if len(remainder) >= 2:
                    walk(remainder)
        for prefix in _COUNTRY_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                remainder = text[len(prefix) :]
                if len(remainder) >= 2:
                    walk(remainder)
        for prefix in _PROVINCE_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                remainder = text[len(prefix) :]
                if len(remainder) >= 2:
                    walk(remainder)

    walk(normalized)
    return variants
