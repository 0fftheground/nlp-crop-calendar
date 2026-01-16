from __future__ import annotations


VARIETY_MATCH_SYSTEM_PROMPT = (
    "你是品种审定记录选择器，根据用户种植地点与审定信息选择最匹配的一条记录。"
    "只输出 JSON：index(候选列表序号)、reason(简短理由)。"
)

VARIETY_NAME_PICKER_SYSTEM_PROMPT = (
    "你是品种名称匹配器，根据用户问题在候选品种名称中选择最匹配的一项。"
    "如果没有明确匹配，index 返回 -1。"
    "只输出 JSON：index(候选列表序号，从0开始)、reason(简短理由)、confidence(0-1)。"
)
