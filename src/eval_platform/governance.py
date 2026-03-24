from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .common import ensure_dataset_path, load_yaml_file


def load_governance_file(path: Path) -> Dict[str, Any]:
    payload = load_yaml_file(path)
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Governance file has no profiles: {path}")
    return payload


def resolve_profile(
    governance_path: Path, profile_name: str
) -> Dict[str, Any]:
    payload = load_governance_file(governance_path)
    profiles = payload["profiles"]
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise ValueError(f"Unknown eval profile '{profile_name}' in {governance_path}")
    return profile


def resolve_dataset_specs(
    *,
    dataset_args: Optional[Iterable[str]] = None,
    governance_path: Optional[Path] = None,
    profile_name: Optional[str] = None,
) -> Dict[str, Any]:
    dataset_specs: List[Dict[str, Any]] = []
    profile: Optional[Dict[str, Any]] = None
    if profile_name:
        if governance_path is None:
            raise ValueError("governance_path is required when profile_name is set")
        profile = resolve_profile(governance_path, profile_name)
        for item in profile.get("datasets") or []:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not path:
                continue
            dataset_specs.append(
                {
                    "path": ensure_dataset_path(str(path)),
                    "include_gates": list(item.get("include_gates") or []),
                    "line": item.get("line"),
                }
            )
    for dataset_arg in dataset_args or []:
        dataset_specs.append(
            {"path": ensure_dataset_path(str(dataset_arg)), "include_gates": [], "line": None}
        )
    if not dataset_specs:
        raise ValueError("No datasets resolved. Provide --dataset or --profile.")
    return {"profile": profile, "dataset_specs": dataset_specs}


def case_gate(case: Dict[str, Any]) -> str:
    return str(case.get("gate") or "regression").strip().lower()


def filter_cases_by_gate(
    cases: List[Dict[str, Any]], include_gates: Iterable[str]
) -> List[Dict[str, Any]]:
    allowed = {str(item).strip().lower() for item in include_gates if str(item).strip()}
    if not allowed:
        return list(cases)
    return [case for case in cases if case_gate(case) in allowed]


def evaluate_profile_result(
    reports: List[Dict[str, Any]], profile: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    if not profile:
        total_failed = sum(item["summary"]["failed"] for item in reports)
        total_cases = sum(item["summary"]["total"] for item in reports)
        total_latency_ms = round(
            sum(float(item["summary"].get("total_latency_ms", 0.0)) for item in reports),
            2,
        )
        estimated_total_tokens = sum(
            int(item["summary"].get("estimated_total_tokens", 0)) for item in reports
        )
        return {
            "passed": total_failed == 0,
            "exit_code": 0 if total_failed == 0 else 1,
            "policy": {"mode": "all_datasets_must_pass"},
            "blocking_failed": sum(
                item["summary"].get("blocking_failed", 0) for item in reports
            ),
            "pass_rate": round(
                ((total_cases - total_failed) / total_cases), 4
            )
            if total_cases
            else 0.0,
            "avg_latency_ms": round(total_latency_ms / total_cases, 2)
            if total_cases
            else 0.0,
            "estimated_total_tokens": estimated_total_tokens,
        }

    policy = dict(profile.get("policy") or {})
    max_blocking_failures = int(policy.get("max_blocking_failures", 0))
    min_pass_rate = float(policy.get("min_pass_rate", 0.0))
    enforce_exit_code = bool(policy.get("enforce_exit_code", True))
    max_avg_latency_ms = policy.get("max_avg_latency_ms")
    max_estimated_total_tokens = policy.get("max_estimated_total_tokens")

    blocking_failed = sum(item["summary"].get("blocking_failed", 0) for item in reports)
    total = sum(item["summary"]["total"] for item in reports)
    passed = sum(item["summary"]["passed"] for item in reports)
    pass_rate = round((passed / total), 4) if total else 0.0
    total_latency_ms = round(
        sum(float(item["summary"].get("total_latency_ms", 0.0)) for item in reports), 2
    )
    avg_latency_ms = round(total_latency_ms / total, 2) if total else 0.0
    estimated_total_tokens = sum(
        int(item["summary"].get("estimated_total_tokens", 0)) for item in reports
    )
    latency_ok = (
        True
        if max_avg_latency_ms is None
        else avg_latency_ms <= float(max_avg_latency_ms)
    )
    tokens_ok = (
        True
        if max_estimated_total_tokens is None
        else estimated_total_tokens <= int(max_estimated_total_tokens)
    )
    ok = (
        blocking_failed <= max_blocking_failures
        and pass_rate >= min_pass_rate
        and latency_ok
        and tokens_ok
    )
    exit_code = 0 if (ok or not enforce_exit_code) else 1
    return {
        "passed": ok,
        "exit_code": exit_code,
        "policy": policy,
        "blocking_failed": blocking_failed,
        "pass_rate": pass_rate,
        "avg_latency_ms": avg_latency_ms,
        "estimated_total_tokens": estimated_total_tokens,
        "latency_ok": latency_ok,
        "tokens_ok": tokens_ok,
    }
