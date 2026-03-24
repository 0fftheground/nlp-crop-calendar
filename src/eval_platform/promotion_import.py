from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .assets import (
    DEFAULT_EXPERT_ROOT,
    DEFAULT_GOVERNANCE_FILE,
    DEFAULT_PRODUCTION_AUDIT_ROOT,
)
from .cli import print_report, run_dataset_file
from .common import ensure_dataset_path, load_yaml_file, temporary_model_overrides
from .governance import evaluate_profile_result, resolve_dataset_specs
from .audit_pipeline import yaml_dump


def _default_expert_dataset_path(task: str, expert_root: Path) -> Path:
    return expert_root / f"{task}.yaml"


def _default_production_audit_dataset_path(task: str, production_audit_root: Path) -> Path:
    return production_audit_root / f"{task}.yaml"


def _normalize_match_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_match_value(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list):
        return [_normalize_match_value(item) for item in value]
    return value


def _case_signature(case: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input": _normalize_match_value(case.get("input") or {}),
        "expected": _normalize_match_value(case.get("expected") or {}),
    }


def _case_matches_promoted_case(
    production_case: Dict[str, Any],
    promoted_case: Dict[str, Any],
) -> bool:
    production_source = production_case.get("source") or {}
    promoted_source = promoted_case.get("source") or {}
    production_interaction_id = production_source.get("interaction_id")
    promoted_interaction_id = promoted_source.get("interaction_id")
    if (
        production_interaction_id is not None
        and promoted_interaction_id is not None
        and str(production_interaction_id) == str(promoted_interaction_id)
    ):
        return True
    production_id = str(production_case.get("id") or "").strip()
    promoted_id = str(promoted_case.get("id") or "").strip()
    if production_id and promoted_id and production_id == promoted_id:
        return True
    return _case_signature(production_case) == _case_signature(promoted_case)


def _prune_production_audit_cases(
    *,
    promoted_cases_by_task: Dict[str, List[Dict[str, Any]]],
    production_audit_root: Path,
) -> List[Dict[str, Any]]:
    pruned: List[Dict[str, Any]] = []
    for task, promoted_cases in promoted_cases_by_task.items():
        target_path = _default_production_audit_dataset_path(task, production_audit_root)
        if not target_path.exists():
            continue
        payload = load_yaml_file(target_path) or {}
        existing_cases = list(payload.get("cases") or [])
        remaining_cases: List[Dict[str, Any]] = []
        removed_cases: List[Dict[str, Any]] = []
        for case in existing_cases:
            if any(
                _case_matches_promoted_case(case, promoted_case)
                for promoted_case in promoted_cases
            ):
                removed_cases.append(case)
            else:
                remaining_cases.append(case)
        if len(remaining_cases) == len(existing_cases):
            continue
        payload["cases"] = remaining_cases
        yaml_dump(payload, target_path)
        pruned.append(
            {
                "task": task,
                "target_dataset": str(target_path),
                "removed_cases": len(removed_cases),
                "removed_case_ids": [
                    str(item.get("id") or "")
                    for item in removed_cases
                    if str(item.get("id") or "")
                ],
            }
        )
    return pruned


def import_promotion_payloads(
    *,
    promotion_files: Iterable[str],
    expert_root: str = DEFAULT_EXPERT_ROOT,
    production_audit_root: str = DEFAULT_PRODUCTION_AUDIT_ROOT,
    prune_production_audit: bool = True,
) -> Dict[str, Any]:
    root = ensure_dataset_path(expert_root)
    production_root = ensure_dataset_path(production_audit_root)
    imported: List[Dict[str, Any]] = []
    promoted_cases_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for promotion_file in promotion_files:
        promotion_path = ensure_dataset_path(promotion_file)
        payload = load_yaml_file(promotion_path)
        task = str(payload.get("task") or "").strip()
        if not task:
            raise ValueError(f"Promotion payload missing task: {promotion_path}")
        target_path = _default_expert_dataset_path(task, root)
        target = load_yaml_file(target_path) if target_path.exists() else {}
        if not target:
            target = {
                "task": task,
                "line": "expert",
                "owner": "production_audit_promotion",
                "cases": [],
            }
        existing_cases = list(target.get("cases") or [])
        existing_index = {
            str(case.get("id") or ""): idx
            for idx, case in enumerate(existing_cases)
            if str(case.get("id") or "")
        }
        upserted = 0
        for case in payload.get("cases") or []:
            case_id = str(case.get("id") or "").strip()
            if not case_id:
                continue
            promoted_cases_by_task.setdefault(task, []).append(case)
            if case_id in existing_index:
                existing_cases[existing_index[case_id]] = case
            else:
                existing_cases.append(case)
            upserted += 1
        target["cases"] = existing_cases
        yaml_dump(target, target_path)
        imported.append(
            {
                "promotion_file": str(promotion_path),
                "target_dataset": str(target_path),
                "task": task,
                "upserted_cases": upserted,
            }
        )
    pruned = (
        _prune_production_audit_cases(
            promoted_cases_by_task=promoted_cases_by_task,
            production_audit_root=production_root,
        )
        if prune_production_audit
        else []
    )
    return {"imported": imported, "pruned_production_audit": pruned}


def rerun_profiles(
    *,
    governance_file: str,
    profiles: Iterable[str],
    llm_model: Optional[str] = None,
    extractor_model: Optional[str] = None,
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    with temporary_model_overrides(
        llm_model=llm_model,
        extractor_model=extractor_model,
    ):
        for profile_name in profiles:
            resolved = resolve_dataset_specs(
                governance_path=ensure_dataset_path(governance_file),
                profile_name=profile_name,
            )
            reports: List[Dict[str, Any]] = []
            for dataset_spec in resolved["dataset_specs"]:
                reports.append(
                    run_dataset_file(
                        dataset_spec["path"],
                        include_gates=list(dataset_spec.get("include_gates") or []),
                        dataset_line=dataset_spec.get("line"),
                    )
                )
            profile_result = evaluate_profile_result(reports, resolved.get("profile"))
            runs.append(
                {
                    "profile": profile_name,
                    "reports": reports,
                    "profile_result": profile_result,
                }
            )
    return {"runs": runs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import production-audit promotion candidates into expert datasets."
    )
    parser.add_argument("--promotion", action="append", required=True)
    parser.add_argument("--expert-root", default=DEFAULT_EXPERT_ROOT)
    parser.add_argument(
        "--production-audit-root",
        default=DEFAULT_PRODUCTION_AUDIT_ROOT,
    )
    parser.add_argument("--governance-file", default=DEFAULT_GOVERNANCE_FILE)
    parser.add_argument(
        "--rerun-profile",
        action="append",
        default=[],
        help="Optional profiles to rerun after import.",
    )
    parser.add_argument(
        "--keep-production-audit",
        action="store_true",
        help="Do not remove matching cases from production_audit after promotion import.",
    )
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--extractor-model", default=None)
    parser.add_argument("--json-out", default=None)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    imported = import_promotion_payloads(
        promotion_files=args.promotion,
        expert_root=args.expert_root,
        production_audit_root=args.production_audit_root,
        prune_production_audit=not args.keep_production_audit,
    )
    for item in imported["imported"]:
        print(
            f"imported task={item['task']} upserted={item['upserted_cases']} -> {item['target_dataset']}"
        )
    for item in imported.get("pruned_production_audit") or []:
        print(
            f"pruned task={item['task']} removed={item['removed_cases']} from {item['target_dataset']}"
        )
    rerun = None
    if args.rerun_profile:
        rerun = rerun_profiles(
            governance_file=args.governance_file,
            profiles=args.rerun_profile,
            llm_model=args.llm_model,
            extractor_model=args.extractor_model,
        )
        for run in rerun["runs"]:
            print(f"== rerun {run['profile']} ==")
            for report in run["reports"]:
                print_report(report)
            print(f"passed={run['profile_result']['passed']}")
    payload = {"imported": imported, "rerun": rerun}
    if args.json_out:
        out_path = ensure_dataset_path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if rerun:
        return 0 if all(run["profile_result"]["passed"] for run in rerun["runs"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
