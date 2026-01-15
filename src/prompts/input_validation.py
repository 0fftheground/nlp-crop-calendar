from __future__ import annotations

from typing import Dict, Iterable


INPUT_SCHEMA_FALLBACK_MESSAGE = "仍缺少执行所需信息，请重新描述你的问题。"


def format_input_validation_message(
    action_name: str, missing_fields: Iterable[str], field_labels: Dict[str, str]
) -> str:
    labels = [field_labels.get(field, field) for field in missing_fields]
    joined = "、".join(labels)
    return f"为了执行 {action_name}，还需要补充：{joined}。"
