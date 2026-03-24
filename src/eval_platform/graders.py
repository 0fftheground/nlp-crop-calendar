from __future__ import annotations

from typing import Any, Dict, List


def _compare_subset(expected: Any, actual: Any, path: str, mismatches: List[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            mismatches.append(f"{path or '<root>'}: expected object, got {type(actual).__name__}")
            return
        for key, value in expected.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in actual:
                mismatches.append(f"{child_path}: missing")
                continue
            _compare_subset(value, actual[key], child_path, mismatches)
        return
    if isinstance(expected, list):
        if actual != expected:
            mismatches.append(f"{path or '<root>'}: expected {expected!r}, got {actual!r}")
        return
    if actual != expected:
        mismatches.append(f"{path or '<root>'}: expected {expected!r}, got {actual!r}")


def grade_case(expected: Dict[str, Any], actual: Dict[str, Any]) -> Dict[str, Any]:
    mismatches: List[str] = []
    _compare_subset(expected, actual, "", mismatches)
    checked = max(1, _count_leaf_fields(expected))
    passed = not mismatches
    matched = checked if passed else max(0, checked - len(mismatches))
    return {
        "passed": passed,
        "checked_fields": checked,
        "matched_fields": matched,
        "mismatches": mismatches,
        "score": round(matched / checked, 4),
    }


def _count_leaf_fields(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_count_leaf_fields(item) for item in value.values())
    return 1
