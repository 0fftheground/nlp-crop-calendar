from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .assets import DEFAULT_GOVERNANCE_FILE
from .common import (
    ensure_dataset_path,
    load_yaml_file,
    percentile,
    temporary_model_overrides,
    to_jsonable,
    validate_openai_model_available,
)
from .graders import grade_case
from .governance import (
    case_gate,
    evaluate_profile_result,
    filter_cases_by_gate,
    resolve_dataset_specs,
)
from ..infra.config import get_config


def _get_task_runners():
    from .runners import TASK_RUNNERS

    return TASK_RUNNERS


def _split_runner_output(result: Any) -> tuple[Any, Dict[str, Any]]:
    if isinstance(result, dict) and "__eval_actual__" in result:
        actual = to_jsonable(result.get("__eval_actual__"))
        metrics = dict(result.get("__eval_metrics__") or {})
        return actual, to_jsonable(metrics)
    return to_jsonable(result), {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval datasets for planner, extractor, variety match, "
            "workflow extract, session context, and follow-up resume."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Path to a YAML dataset file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Named eval governance profile to run.",
    )
    parser.add_argument(
        "--governance-file",
        default=DEFAULT_GOVERNANCE_FILE,
        help="Governance file used by --profile.",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Override LLM_MODEL for planner/variety-match evals.",
    )
    parser.add_argument(
        "--extractor-model",
        default=None,
        help="Override EXTRACTOR_MODEL for extractor evals.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to save the full eval result as JSON.",
    )
    return parser


def run_dataset_file(
    path: Path, *, include_gates: List[str] | None = None, dataset_line: str | None = None
) -> Dict[str, Any]:
    payload = load_yaml_file(path)
    task = str(payload.get("task") or "").strip()
    task_runners = _get_task_runners()
    if task not in task_runners:
        raise ValueError(f"Unsupported eval task '{task}' in {path}")
    runner = task_runners[task]
    cases = payload.get("cases") or []
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Dataset has no cases: {path}")
    cases = filter_cases_by_gate(cases, include_gates or [])
    if not cases:
        raise ValueError(f"Dataset has no cases after gate filtering: {path}")

    results: List[Dict[str, Any]] = []
    passed = 0
    total_score = 0.0
    blocking_failed = 0
    latency_values: List[float] = []
    estimated_input_tokens = 0
    estimated_output_tokens = 0
    estimated_total_tokens = 0
    model_names: set[str] = set()
    for case in cases:
        case_id = case.get("id") or "unknown"
        raw_result = runner(case)
        actual, metrics = _split_runner_output(raw_result)
        expected = dict(case.get("expected") or {})
        grade = grade_case(expected, actual)
        total_score += float(grade["score"])
        if grade["passed"]:
            passed += 1
        gate = case_gate(case)
        if gate == "blocking" and not grade["passed"]:
            blocking_failed += 1
        latency_ms = metrics.get("latency_ms")
        if isinstance(latency_ms, (int, float)):
            latency_values.append(float(latency_ms))
        if isinstance(metrics.get("estimated_input_tokens"), int):
            estimated_input_tokens += int(metrics["estimated_input_tokens"])
        if isinstance(metrics.get("estimated_output_tokens"), int):
            estimated_output_tokens += int(metrics["estimated_output_tokens"])
        if isinstance(metrics.get("estimated_total_tokens"), int):
            estimated_total_tokens += int(metrics["estimated_total_tokens"])
        model_name = metrics.get("model")
        if isinstance(model_name, str) and model_name.strip():
            model_names.add(model_name.strip())
        results.append(
            {
                "id": case_id,
                "gate": gate,
                "source": dict(case.get("source") or {}),
                "actual": actual,
                "metrics": metrics,
                "expected": expected,
                "grade": grade,
            }
        )

    total = len(results)
    total_latency_ms = round(sum(latency_values), 2)
    return {
        "dataset": str(path),
        "task": task,
        "line": dataset_line or payload.get("line"),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "blocking_failed": blocking_failed,
            "pass_rate": round(passed / total, 4),
            "avg_score": round(total_score / total, 4),
            "total_latency_ms": total_latency_ms,
            "avg_latency_ms": round(total_latency_ms / total, 2),
            "p95_latency_ms": percentile(latency_values, 95),
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_total_tokens": estimated_total_tokens,
            "avg_estimated_total_tokens": round(estimated_total_tokens / total, 2),
            "model_names": sorted(model_names),
        },
        "cases": results,
    }


def print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    line = report.get("line") or "default"
    print(f"[{report['task']}] {report['dataset']} line={line}")
    print(
        f"pass {summary['passed']}/{summary['total']} "
        f"rate={summary['pass_rate']:.2%} avg_score={summary['avg_score']:.2%} "
        f"blocking_failed={summary.get('blocking_failed', 0)}"
    )
    print(
        "latency avg={avg_latency_ms:.2f}ms p95={p95_latency_ms:.2f}ms "
        "tokens total={estimated_total_tokens} avg={avg_estimated_total_tokens:.2f}".format(
            avg_latency_ms=float(summary.get("avg_latency_ms", 0.0)),
            p95_latency_ms=float(summary.get("p95_latency_ms", 0.0)),
            estimated_total_tokens=int(summary.get("estimated_total_tokens", 0)),
            avg_estimated_total_tokens=float(
                summary.get("avg_estimated_total_tokens", 0.0)
            ),
        )
    )
    if summary.get("model_names"):
        print(f"models: {', '.join(summary['model_names'])}")
    for item in report["cases"]:
        grade = item["grade"]
        status = "PASS" if grade["passed"] else "FAIL"
        print(f"  - {status} {item['id']} gate={item.get('gate')}")
        for mismatch in grade["mismatches"]:
            print(f"    {mismatch}")


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dataset and not args.profile:
        parser.error("one of --dataset or --profile is required")
    cfg = get_config()
    try:
        if args.llm_model:
            validate_openai_model_available(
                model_name=args.llm_model,
                api_key=cfg.openai_api_key,
                base_url=cfg.openai_api_base,
                timeout_seconds=cfg.backend_timeout_seconds,
                label="llm",
            )
        if args.extractor_model:
            validate_openai_model_available(
                model_name=args.extractor_model,
                api_key=cfg.extractor_api_key or cfg.openai_api_key,
                base_url=cfg.extractor_api_base or cfg.openai_api_base,
                timeout_seconds=cfg.backend_timeout_seconds,
                label="extractor",
            )
    except Exception as exc:
        print(f"model_validation_error={exc}")
        return 2
    reports: List[Dict[str, Any]] = []
    resolved = resolve_dataset_specs(
        dataset_args=args.dataset,
        governance_path=ensure_dataset_path(args.governance_file),
        profile_name=args.profile,
    )
    with temporary_model_overrides(
        llm_model=args.llm_model,
        extractor_model=args.extractor_model,
    ):
        for dataset_spec in resolved["dataset_specs"]:
            dataset_path = dataset_spec["path"]
            report = run_dataset_file(
                dataset_path,
                include_gates=list(dataset_spec.get("include_gates") or []),
                dataset_line=dataset_spec.get("line"),
            )
            reports.append(report)
            print_report(report)

    profile_result = evaluate_profile_result(reports, resolved.get("profile"))
    payload = {"reports": reports, "profile_result": profile_result}
    if args.json_out:
        out_path = ensure_dataset_path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return int(profile_result["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
