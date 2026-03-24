from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .assets import DEFAULT_GOVERNANCE_FILE
from .cli import run_dataset_file
from .common import ensure_dataset_path, load_yaml_file, temporary_model_overrides
from .common import validate_openai_model_available
from .governance import evaluate_profile_result, resolve_dataset_specs
from ..infra.config import get_config


_LLM_TASKS = {"planner", "variety_match"}
_EXTRACTOR_TASKS = {"extractor", "workflow_extract"}


def _resolve_effective_models(
    *,
    llm_model: Optional[str],
    extractor_model: Optional[str],
) -> Dict[str, Optional[str]]:
    cfg = get_config()
    return {
        "llm_model": llm_model or cfg.llm_model,
        "extractor_model": extractor_model or cfg.extractor_model,
    }


def _validate_compare_models(
    *,
    baseline_llm_model: Optional[str],
    baseline_extractor_model: Optional[str],
    candidate_llm_model: Optional[str],
    candidate_extractor_model: Optional[str],
) -> None:
    cfg = get_config()
    effective_baseline = _resolve_effective_models(
        llm_model=baseline_llm_model,
        extractor_model=baseline_extractor_model,
    )
    effective_candidate = _resolve_effective_models(
        llm_model=candidate_llm_model,
        extractor_model=candidate_extractor_model,
    )
    llm_models = {
        value
        for value in [effective_baseline["llm_model"], effective_candidate["llm_model"]]
        if value
    }
    extractor_models = {
        value
        for value in [
            effective_baseline["extractor_model"],
            effective_candidate["extractor_model"],
        ]
        if value
    }
    for model_name in sorted(llm_models):
        validate_openai_model_available(
            model_name=model_name,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_api_base,
            timeout_seconds=cfg.backend_timeout_seconds,
            label="llm",
        )
    for model_name in sorted(extractor_models):
        validate_openai_model_available(
            model_name=model_name,
            api_key=cfg.extractor_api_key or cfg.openai_api_key,
            base_url=cfg.extractor_api_base or cfg.openai_api_base,
            timeout_seconds=cfg.backend_timeout_seconds,
            label="extractor",
        )


def _select_impacted_tasks(
    *,
    baseline_llm_model: Optional[str],
    baseline_extractor_model: Optional[str],
    candidate_llm_model: Optional[str],
    candidate_extractor_model: Optional[str],
) -> Set[str]:
    baseline = _resolve_effective_models(
        llm_model=baseline_llm_model,
        extractor_model=baseline_extractor_model,
    )
    candidate = _resolve_effective_models(
        llm_model=candidate_llm_model,
        extractor_model=candidate_extractor_model,
    )
    tasks: Set[str] = set()
    if baseline["llm_model"] != candidate["llm_model"]:
        tasks.update(_LLM_TASKS)
    if baseline["extractor_model"] != candidate["extractor_model"]:
        tasks.update(_EXTRACTOR_TASKS)
    return tasks


def _dataset_task(dataset_path: Path) -> str:
    payload = load_yaml_file(dataset_path)
    return str(payload.get("task") or "").strip()


def _filter_dataset_specs_by_task(
    dataset_specs: Iterable[Dict[str, Any]],
    *,
    allowed_tasks: Set[str],
) -> List[Dict[str, Any]]:
    if not allowed_tasks:
        return []
    filtered: List[Dict[str, Any]] = []
    for item in dataset_specs:
        task = _dataset_task(item["path"])
        if task in allowed_tasks:
            filtered.append(item)
    return filtered


def _run_profile(
    *,
    profile_name: str,
    governance_file: str,
    llm_model: Optional[str],
    extractor_model: Optional[str],
    allowed_tasks: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    resolved = resolve_dataset_specs(
        governance_path=ensure_dataset_path(governance_file),
        profile_name=profile_name,
    )
    dataset_specs = list(resolved["dataset_specs"])
    if allowed_tasks is not None:
        dataset_specs = _filter_dataset_specs_by_task(
            dataset_specs,
            allowed_tasks=allowed_tasks,
        )
    reports: List[Dict[str, Any]] = []
    with temporary_model_overrides(
        llm_model=llm_model,
        extractor_model=extractor_model,
    ):
        for dataset_spec in dataset_specs:
            report = run_dataset_file(
                dataset_spec["path"],
                include_gates=list(dataset_spec.get("include_gates") or []),
                dataset_line=dataset_spec.get("line"),
            )
            reports.append(report)
    if reports:
        profile_result = evaluate_profile_result(reports, resolved.get("profile"))
    else:
        profile_result = {
            "passed": True,
            "exit_code": 0,
            "policy": dict((resolved.get("profile") or {}).get("policy") or {}),
            "blocking_failed": 0,
            "pass_rate": 1.0,
            "avg_latency_ms": 0.0,
            "estimated_total_tokens": 0,
            "skipped": True,
            "skip_reason": "no_impacted_tasks",
        }
    return {
        "profile": profile_name,
        "reports": reports,
        "profile_result": profile_result,
        "models": {
            "llm_model": llm_model,
            "extractor_model": extractor_model,
        },
        "allowed_tasks": sorted(allowed_tasks or []),
    }


def _index_reports(reports: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    indexed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for report in reports:
        indexed[(str(report.get("task") or ""), str(report.get("dataset") or ""))] = report
    return indexed


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compare_release_candidates(
    *,
    governance_file: str,
    blocking_profile: str,
    regression_profile: str,
    baseline_llm_model: Optional[str],
    baseline_extractor_model: Optional[str],
    candidate_llm_model: Optional[str],
    candidate_extractor_model: Optional[str],
) -> Dict[str, Any]:
    impacted_tasks = _select_impacted_tasks(
        baseline_llm_model=baseline_llm_model,
        baseline_extractor_model=baseline_extractor_model,
        candidate_llm_model=candidate_llm_model,
        candidate_extractor_model=candidate_extractor_model,
    )
    candidate_blocking = _run_profile(
        profile_name=blocking_profile,
        governance_file=governance_file,
        llm_model=candidate_llm_model,
        extractor_model=candidate_extractor_model,
        allowed_tasks=impacted_tasks,
    )
    baseline_regression = _run_profile(
        profile_name=regression_profile,
        governance_file=governance_file,
        llm_model=baseline_llm_model,
        extractor_model=baseline_extractor_model,
        allowed_tasks=impacted_tasks,
    )
    candidate_regression = _run_profile(
        profile_name=regression_profile,
        governance_file=governance_file,
        llm_model=candidate_llm_model,
        extractor_model=candidate_extractor_model,
        allowed_tasks=impacted_tasks,
    )

    baseline_index = _index_reports(baseline_regression["reports"])
    candidate_index = _index_reports(candidate_regression["reports"])
    regression_policy = dict(candidate_regression["profile_result"].get("policy") or {})
    max_latency_regression_ratio = _safe_float(
        regression_policy.get("max_latency_regression_ratio")
    )
    max_total_tokens_regression_ratio = _safe_float(
        regression_policy.get("max_total_tokens_regression_ratio")
    )
    regressions: List[Dict[str, Any]] = []
    for key, baseline_report in baseline_index.items():
        candidate_report = candidate_index.get(key)
        if not candidate_report:
            regressions.append(
                {
                    "task": key[0],
                    "dataset": key[1],
                    "reason": "candidate_missing_dataset",
                }
            )
            continue
        base_summary = baseline_report["summary"]
        cand_summary = candidate_report["summary"]
        reasons: List[str] = []
        if float(cand_summary["pass_rate"]) < float(base_summary["pass_rate"]):
            reasons.append(
                f"pass_rate dropped: baseline={base_summary['pass_rate']:.4f}, candidate={cand_summary['pass_rate']:.4f}"
            )
        if float(cand_summary["avg_score"]) < float(base_summary["avg_score"]):
            reasons.append(
                f"avg_score dropped: baseline={base_summary['avg_score']:.4f}, candidate={cand_summary['avg_score']:.4f}"
            )
        if int(cand_summary.get("blocking_failed", 0)) > int(
            base_summary.get("blocking_failed", 0)
        ):
            reasons.append(
                "blocking_failed increased: "
                f"baseline={base_summary.get('blocking_failed', 0)}, "
                f"candidate={cand_summary.get('blocking_failed', 0)}"
            )
        base_latency = _safe_float(base_summary.get("avg_latency_ms"))
        cand_latency = _safe_float(cand_summary.get("avg_latency_ms"))
        if (
            max_latency_regression_ratio is not None
            and base_latency is not None
            and cand_latency is not None
            and base_latency > 0
            and cand_latency > (base_latency * max_latency_regression_ratio)
        ):
            reasons.append(
                "avg_latency_ms exceeded regression ratio: "
                f"baseline={base_latency:.2f}, candidate={cand_latency:.2f}, "
                f"limit={base_latency * max_latency_regression_ratio:.2f}"
            )
        base_tokens = _safe_int(base_summary.get("estimated_total_tokens"))
        cand_tokens = _safe_int(cand_summary.get("estimated_total_tokens"))
        if (
            max_total_tokens_regression_ratio is not None
            and base_tokens is not None
            and cand_tokens is not None
            and base_tokens > 0
            and cand_tokens > int(base_tokens * max_total_tokens_regression_ratio)
        ):
            reasons.append(
                "estimated_total_tokens exceeded regression ratio: "
                f"baseline={base_tokens}, candidate={cand_tokens}, "
                f"limit={int(base_tokens * max_total_tokens_regression_ratio)}"
            )
        if reasons:
            regressions.append(
                {
                    "task": key[0],
                    "dataset": key[1],
                    "reasons": reasons,
                    "baseline_summary": base_summary,
                    "candidate_summary": cand_summary,
                }
            )

    candidate_blocking_passed = bool(candidate_blocking["profile_result"]["passed"])
    candidate_regression_passed = bool(candidate_regression["profile_result"]["passed"])
    no_regressions = not regressions
    passed = candidate_blocking_passed and candidate_regression_passed and no_regressions
    return {
        "passed": passed,
        "impacted_tasks": sorted(impacted_tasks),
        "candidate_blocking": candidate_blocking,
        "baseline_regression": baseline_regression,
        "candidate_regression": candidate_regression,
        "comparison": {
            "no_regressions": no_regressions,
            "regressions": regressions,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate models on release eval profiles."
    )
    parser.add_argument(
        "--governance-file",
        default=DEFAULT_GOVERNANCE_FILE,
        help="Governance file used to resolve profiles.",
    )
    parser.add_argument(
        "--blocking-profile",
        default="expert_blocking_gate",
        help="Blocking profile run only on the candidate model.",
    )
    parser.add_argument(
        "--regression-profile",
        default="expert_regression_gate",
        help="Regression profile run on both baseline and candidate models.",
    )
    parser.add_argument("--baseline-llm-model", default=None)
    parser.add_argument("--baseline-extractor-model", default=None)
    parser.add_argument("--candidate-llm-model", default=None)
    parser.add_argument("--candidate-extractor-model", default=None)
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional override for the compare JSON output path. Defaults to .cache/eval/release_compare/latest.json.",
    )
    return parser


def _default_json_out_path() -> Path:
    return Path(".cache") / "eval" / "release_compare" / "latest.json"


def _write_json_payload(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched["generated_at"] = datetime.now().isoformat()
    path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _print_summary(*, payload: Dict[str, Any], out_path: Path) -> None:
    impacted_tasks = payload.get("impacted_tasks") or []
    comparison = payload.get("comparison") or {}
    regressions = list(comparison.get("regressions") or [])
    print(f"compare passed={payload['passed']}")
    print(f"impacted_tasks={', '.join(impacted_tasks) if impacted_tasks else 'none'}")
    print(f"regressions={len(regressions)}")
    if regressions:
        preview = []
        for item in regressions[:3]:
            preview.append(f"{item.get('task')}:{Path(str(item.get('dataset') or '')).name}")
        print(f"regression_preview={', '.join(preview)}")
    print(f"json_out={out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_compare_models(
            baseline_llm_model=args.baseline_llm_model,
            baseline_extractor_model=args.baseline_extractor_model,
            candidate_llm_model=args.candidate_llm_model,
            candidate_extractor_model=args.candidate_extractor_model,
        )
    except Exception as exc:
        print(f"model_validation_error={exc}")
        return 2
    payload = compare_release_candidates(
        governance_file=args.governance_file,
        blocking_profile=args.blocking_profile,
        regression_profile=args.regression_profile,
        baseline_llm_model=args.baseline_llm_model,
        baseline_extractor_model=args.baseline_extractor_model,
        candidate_llm_model=args.candidate_llm_model,
        candidate_extractor_model=args.candidate_extractor_model,
    )
    out_path = ensure_dataset_path(args.json_out) if args.json_out else ensure_dataset_path(
        str(_default_json_out_path())
    )
    _write_json_payload(out_path, payload)
    if not payload["impacted_tasks"]:
        print("compare passed=True")
        print("impacted_tasks=none")
        print("regressions=0")
        print("status=skipped_no_model_difference")
        print(f"json_out={out_path}")
        return 0
    _print_summary(payload=payload, out_path=out_path)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
