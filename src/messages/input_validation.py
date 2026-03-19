from __future__ import annotations

from typing import Dict, Iterable


INPUT_SCHEMA_FALLBACK_MESSAGE = "仍缺少执行所需信息，请重新描述你的问题。"


def format_input_validation_message(
    action_name: str,
    missing_fields: Iterable[str],
    field_labels: Dict[str, str],
    *,
    invalid_fields: Iterable[str] = (),
) -> str:
    missing_labels = [field_labels.get(field, field) for field in missing_fields]
    invalid_labels = [field_labels.get(field, field) for field in invalid_fields]
    if missing_labels and invalid_labels:
        return (
            f"为了执行 {action_name}，还需要补充：{'、'.join(missing_labels)}；"
            f"并请检查这些字段的格式：{'、'.join(invalid_labels)}。"
        )
    if missing_labels:
        return f"为了执行 {action_name}，还需要补充：{'、'.join(missing_labels)}。"
    if invalid_labels:
        return f"为了执行 {action_name}，请检查这些字段的格式：{'、'.join(invalid_labels)}。"
    return INPUT_SCHEMA_FALLBACK_MESSAGE
