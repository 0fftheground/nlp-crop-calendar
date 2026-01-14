from __future__ import annotations


BASE_PLANTING_EXTRACT_PROMPT = (
    "你是农事助手，负责从用户描述中抽取种植信息。"
    "只输出可确定的信息；不确定或未提及时保持为空。"
    "种植方式使用 direct_seeding 或 transplanting。"
    "日期格式为 YYYY-MM-DD。"
)


def build_planting_extract_prompt(hint: str = "") -> str:
    if hint:
        return f"{BASE_PLANTING_EXTRACT_PROMPT}{hint}"
    return BASE_PLANTING_EXTRACT_PROMPT
