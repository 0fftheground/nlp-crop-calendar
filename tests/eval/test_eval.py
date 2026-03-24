import importlib.util
import csv
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from uuid import uuid4
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TMP_ROOT = ROOT / ".cache" / "test_tmp"

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class EvalHelpersTests(unittest.TestCase):
    def _make_tmpdir(self):
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        path = TMP_ROOT / f"case-{uuid4().hex}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_grade_case_reports_nested_mismatch(self) -> None:
        from src.eval_platform.graders import grade_case

        result = grade_case(
            {"action": "tool", "input": {"region": "长沙"}},
            {"action": "tool", "input": {"region": "株洲"}},
        )

        self.assertFalse(result["passed"])
        self.assertIn("input.region", result["mismatches"][0])

    def test_temporary_model_overrides_updates_cached_config(self) -> None:
        from src.eval_platform.common import temporary_model_overrides
        from src.infra.config import get_config

        backup = {
            "LLM_MODEL": os.environ.get("LLM_MODEL"),
            "EXTRACTOR_MODEL": os.environ.get("EXTRACTOR_MODEL"),
        }
        try:
            os.environ["LLM_MODEL"] = "baseline-chat"
            os.environ["EXTRACTOR_MODEL"] = "baseline-extractor"
            get_config.cache_clear()
            self.assertEqual(get_config().llm_model, "baseline-chat")
            self.assertEqual(get_config().extractor_model, "baseline-extractor")

            with temporary_model_overrides(
                llm_model="candidate-chat",
                extractor_model="candidate-extractor",
            ):
                self.assertEqual(get_config().llm_model, "candidate-chat")
                self.assertEqual(get_config().extractor_model, "candidate-extractor")

            self.assertEqual(get_config().llm_model, "baseline-chat")
            self.assertEqual(get_config().extractor_model, "baseline-extractor")
        finally:
            for key, value in backup.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            get_config.cache_clear()

    def test_validate_openai_model_available_raises_on_404(self) -> None:
        from src.eval_platform.common import validate_openai_model_available

        class _Resp:
            status_code = 404
            text = "not found"

        with patch("src.eval_platform.common.httpx.get", return_value=_Resp()):
            with self.assertRaisesRegex(ValueError, "does not exist"):
                validate_openai_model_available(
                    model_name="missing-model",
                    api_key="test-key",
                    base_url="https://api.openai.com/v1",
                    timeout_seconds=30,
                    label="llm",
                )

    def test_eval_platform_asset_defaults_point_to_decoupled_paths(self) -> None:
        from src.eval_platform.assets import (
            DEFAULT_EXPERT_ROOT,
            DEFAULT_GOVERNANCE_FILE,
            DEFAULT_PRODUCTION_AUDIT_ROOT,
        )

        self.assertEqual(DEFAULT_GOVERNANCE_FILE, "src/eval_assets/governance.yaml")
        self.assertEqual(DEFAULT_EXPERT_ROOT, "src/eval_assets/expert")
        self.assertEqual(
            DEFAULT_PRODUCTION_AUDIT_ROOT,
            "src/eval_assets/production_audit",
        )

    def test_eval_platform_entrypoint_dispatches_compare(self) -> None:
        from src.eval_platform.entrypoint import main

        with patch("src.eval_platform.entrypoint.compare_module.main", return_value=0) as mocked:
            exit_code = main(["compare", "--json-out", "x.json"])

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(["--json-out", "x.json"])

    def test_eval_platform_entrypoint_keeps_backward_compatible_run_mode(self) -> None:
        from src.eval_platform.entrypoint import main

        with patch("src.eval_platform.entrypoint.run_module.main", return_value=0) as mocked:
            exit_code = main(["--profile", "expert_blocking_gate"])

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(["--profile", "expert_blocking_gate"])

    def test_eval_platform_entrypoint_reads_sys_argv_when_argv_is_none(self) -> None:
        from src.eval_platform.entrypoint import main

        with patch.object(sys, "argv", ["python", "compare", "--json-out", "x.json"]):
            with patch("src.eval_platform.entrypoint.compare_module.main", return_value=0) as mocked:
                exit_code = main()

        self.assertEqual(exit_code, 0)
        mocked.assert_called_once_with(["--json-out", "x.json"])

    def test_run_dataset_file_uses_registered_runner(self) -> None:
        from src.eval_platform.cli import run_dataset_file

        tmp = self._make_tmpdir()
        try:
            dataset_path = tmp / "dataset.yaml"
            dataset_path.write_text(
                "task: planner\nline: expert\ncases:\n"
                "  - id: one\n"
                "    gate: blocking\n"
                "    input:\n"
                "      prompt: hi\n"
                "    expected:\n"
                "      action: tool\n"
                "      name: weather_lookup\n",
                encoding="utf-8",
            )
            with patch(
                "src.eval_platform.cli._get_task_runners",
                return_value={
                    "planner": lambda case: {
                        "__eval_actual__": {
                            "action": "tool",
                            "name": "weather_lookup",
                        },
                        "__eval_metrics__": {
                            "latency_ms": 123.4,
                            "estimated_input_tokens": 11,
                            "estimated_output_tokens": 7,
                            "estimated_total_tokens": 18,
                            "model": "fake-model",
                        },
                    }
                },
            ):
                report = run_dataset_file(dataset_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(report["summary"]["passed"], 1)
        self.assertEqual(report["summary"]["blocking_failed"], 0)
        self.assertEqual(report["summary"]["avg_latency_ms"], 123.4)
        self.assertEqual(report["summary"]["estimated_total_tokens"], 18)
        self.assertEqual(report["summary"]["model_names"], ["fake-model"])
        self.assertEqual(report["cases"][0]["actual"]["name"], "weather_lookup")
        self.assertEqual(report["cases"][0]["metrics"]["latency_ms"], 123.4)

    def test_cli_writes_json_report(self) -> None:
        from src.eval_platform.cli import main

        tmp = self._make_tmpdir()
        try:
            dataset_path = tmp / "dataset.yaml"
            report_path = tmp / "report.json"
            dataset_path.write_text(
                "task: extractor\ncases:\n"
                "  - id: one\n"
                "    gate: blocking\n"
                "    input:\n"
                "      prompt: hi\n"
                "    expected:\n"
                "      crop: 水稻\n",
                encoding="utf-8",
            )
            with patch(
                "src.eval_platform.cli._get_task_runners",
                return_value={"extractor": lambda case: {"crop": "水稻"}},
            ):
                exit_code = main(
                    [
                        "--dataset",
                        str(dataset_path),
                        "--json-out",
                        str(report_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["reports"][0]["summary"]["passed"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_run_dataset_file_filters_by_gate(self) -> None:
        from src.eval_platform.cli import run_dataset_file

        tmp = self._make_tmpdir()
        try:
            dataset_path = tmp / "dataset.yaml"
            dataset_path.write_text(
                "task: planner\ncases:\n"
                "  - id: blocking_case\n"
                "    gate: blocking\n"
                "    input:\n"
                "      prompt: a\n"
                "    expected:\n"
                "      action: tool\n"
                "  - id: regression_case\n"
                "    gate: regression\n"
                "    input:\n"
                "      prompt: b\n"
                "    expected:\n"
                "      action: tool\n",
                encoding="utf-8",
            )
            with patch(
                "src.eval_platform.cli._get_task_runners",
                return_value={"planner": lambda case: {"action": "tool"}},
            ):
                report = run_dataset_file(dataset_path, include_gates=["blocking"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["cases"][0]["id"], "blocking_case")

    def test_cli_profile_uses_governance_and_non_gating_audit_exits_zero(self) -> None:
        from src.eval_platform.cli import main

        tmp = self._make_tmpdir()
        try:
            dataset_path = tmp / "audit.yaml"
            governance_path = tmp / "governance.yaml"
            dataset_path.write_text(
                "task: planner\nline: production_audit\ncases:\n"
                "  - id: audit_case\n"
                "    gate: audit\n"
                "    input:\n"
                "      prompt: hi\n"
                "    expected:\n"
                "      action: tool\n",
                encoding="utf-8",
            )
            governance_path.write_text(
                "profiles:\n"
                "  production_audit_review:\n"
                "    datasets:\n"
                f"      - path: {dataset_path.as_posix()}\n"
                "        include_gates: [audit]\n"
                "        line: production_audit\n"
                "    policy:\n"
                "      max_blocking_failures: 999\n"
                "      min_pass_rate: 0.0\n"
                "      enforce_exit_code: false\n",
                encoding="utf-8",
            )
            with patch(
                "src.eval_platform.cli._get_task_runners",
                return_value={"planner": lambda case: {"action": "none"}},
            ):
                exit_code = main(
                    [
                        "--profile",
                        "production_audit_review",
                        "--governance-file",
                        str(governance_path),
                    ]
                )
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        self.assertEqual(exit_code, 0)

    def test_evaluate_profile_result_supports_latency_and_token_limits(self) -> None:
        from src.eval_platform.governance import evaluate_profile_result

        result = evaluate_profile_result(
            [
                {
                    "summary": {
                        "total": 2,
                        "passed": 2,
                        "failed": 0,
                        "blocking_failed": 0,
                        "pass_rate": 1.0,
                        "avg_score": 1.0,
                        "total_latency_ms": 500.0,
                        "estimated_total_tokens": 120,
                    }
                }
            ],
            {
                "policy": {
                    "max_blocking_failures": 0,
                    "min_pass_rate": 1.0,
                    "max_avg_latency_ms": 200.0,
                    "max_estimated_total_tokens": 100,
                    "enforce_exit_code": True,
                }
            },
        )

        self.assertFalse(result["passed"])
        self.assertFalse(result["latency_ok"])
        self.assertFalse(result["tokens_ok"])
        self.assertEqual(result["avg_latency_ms"], 250.0)
        self.assertEqual(result["estimated_total_tokens"], 120)

    def test_session_context_runner_reuses_cached_context(self) -> None:
        from src.eval_platform.runners import run_session_context_case

        result = run_session_context_case(
            {
                "input": {
                    "prompt": "芜湖呢",
                    "session_context": {
                        "last_context": {
                            "kind": "tool",
                            "name": "sowing_suitability_lookup",
                        },
                        "tool_contexts": {
                            "sowing_suitability_lookup": {
                                "crop": "水稻",
                                "variety": "湘早籼24号",
                                "region_id": "常德",
                                "culti_type": "早稻",
                                "planting_method": "transplanting",
                            }
                        },
                    },
                }
            }
        )
        actual = result["__eval_actual__"]

        self.assertTrue(actual["matched"])
        self.assertEqual(actual["name"], "sowing_suitability_lookup")
        self.assertEqual(actual["input"]["region_id"], "芜湖")

    def test_followup_resume_runner_builds_tool_followup_payload(self) -> None:
        from src.eval_platform.runners import run_followup_resume_case

        result = run_followup_resume_case(
            {
                "input": {
                    "prompt": "芜湖",
                    "memory_id": "eval-user",
                    "pending": {
                        "mode": "tool",
                        "tool_name": "weather_lookup",
                        "query": "今天适合施肥吗",
                        "draft": {
                            "start_date": "2026-03-17",
                            "end_date": "2026-03-17",
                        },
                        "missing_fields": ["region"],
                        "followup_count": 0,
                    },
                }
            }
        )
        actual = result["__eval_actual__"]

        self.assertTrue(actual["should_resume"])
        self.assertEqual(actual["tool_name"], "weather_lookup")
        self.assertEqual(actual["followup_payload"]["user_id"], "eval-user")
        self.assertEqual(actual["followup_payload"]["followup"]["prompt"], "芜湖")

    def test_workflow_extract_runner_returns_draft_and_missing_fields(self) -> None:
        from src.eval_platform.runners import run_workflow_extract_case

        with patch(
            "src.eval_platform.runners.crop_calendar_extract_node",
            return_value={
                "draft": {
                    "crop": "水稻",
                    "variety": "美香占2号",
                    "planting_method": "direct_seeding",
                    "sowing_date": "2026-03-20",
                },
                "missing_fields": ["culti_type"],
                "pending_message": "请补充稻作类型。",
                "options": [],
            },
        ):
            result = run_workflow_extract_case(
                {
                    "input": {
                        "prompt": "我在长沙种美香占2号，2026-03-20直播，帮我生成种植计划",
                    }
                }
            )

        actual = result["__eval_actual__"]
        self.assertEqual(actual["draft"]["crop"], "水稻")
        self.assertEqual(actual["draft"]["variety"], "美香占2号")
        self.assertEqual(actual["missing_fields"], ["culti_type"])
        self.assertEqual(actual["pending_message"], "请补充稻作类型。")

    def test_release_compare_detects_candidate_regression(self) -> None:
        from src.eval_platform.release_compare import compare_release_candidates

        def fake_resolve_dataset_specs(*, governance_path=None, profile_name=None, dataset_args=None):
            del governance_path, dataset_args
            if profile_name == "expert_blocking_gate":
                return {
                    "profile": {
                        "policy": {
                            "max_blocking_failures": 0,
                            "min_pass_rate": 1.0,
                            "enforce_exit_code": True,
                        }
                    },
                    "dataset_specs": [{"path": Path("blocking.yaml"), "include_gates": ["blocking"], "line": "expert"}],
                }
            return {
                "profile": {
                    "policy": {
                        "max_blocking_failures": 0,
                        "min_pass_rate": 0.0,
                        "max_latency_regression_ratio": 1.5,
                        "max_total_tokens_regression_ratio": 1.5,
                        "enforce_exit_code": True,
                    }
                },
                "dataset_specs": [{"path": Path("regression.yaml"), "include_gates": ["blocking", "regression"], "line": "expert"}],
            }

        def fake_run_dataset_file(path, *, include_gates=None, dataset_line=None):
            del include_gates, dataset_line
            current_model = os.environ.get("LLM_MODEL")
            is_blocking = str(path) == "blocking.yaml"
            pass_rate = 1.0 if current_model != "candidate" or is_blocking else 0.5
            avg_latency_ms = 100.0 if current_model != "candidate" else 200.0
            estimated_total_tokens = 40 if current_model != "candidate" else 80
            return {
                "dataset": str(path),
                "task": "planner",
                "line": "expert",
                "summary": {
                    "total": 2,
                    "passed": 2 if pass_rate == 1.0 else 1,
                    "failed": 0 if pass_rate == 1.0 else 1,
                    "blocking_failed": 0,
                    "pass_rate": pass_rate,
                    "avg_score": pass_rate,
                    "total_latency_ms": avg_latency_ms * 2,
                    "avg_latency_ms": avg_latency_ms,
                    "p95_latency_ms": avg_latency_ms,
                    "estimated_input_tokens": estimated_total_tokens // 2,
                    "estimated_output_tokens": estimated_total_tokens // 2,
                    "estimated_total_tokens": estimated_total_tokens,
                    "avg_estimated_total_tokens": estimated_total_tokens / 2,
                    "model_names": [current_model or "default"],
                },
                "cases": [],
            }

        with patch("src.eval_platform.release_compare.resolve_dataset_specs", side_effect=fake_resolve_dataset_specs):
            with patch("src.eval_platform.release_compare._dataset_task", return_value="planner"):
                with patch("src.eval_platform.release_compare.run_dataset_file", side_effect=fake_run_dataset_file):
                    result = compare_release_candidates(
                        governance_file="src/eval_assets/governance.yaml",
                        blocking_profile="expert_blocking_gate",
                        regression_profile="expert_regression_gate",
                        baseline_llm_model="baseline",
                        baseline_extractor_model=None,
                        candidate_llm_model="candidate",
                        candidate_extractor_model=None,
                    )

        self.assertFalse(result["passed"])
        self.assertEqual(result["impacted_tasks"], ["planner", "variety_match"])
        self.assertEqual(len(result["comparison"]["regressions"]), 1)

    def test_release_compare_filters_to_impacted_extractor_tasks(self) -> None:
        from src.eval_platform.release_compare import compare_release_candidates

        def fake_resolve_dataset_specs(*, governance_path=None, profile_name=None, dataset_args=None):
            del governance_path, profile_name, dataset_args
            return {
                "profile": {
                    "policy": {
                        "max_blocking_failures": 0,
                        "min_pass_rate": 0.0,
                        "max_latency_regression_ratio": 1.5,
                        "max_total_tokens_regression_ratio": 1.5,
                        "enforce_exit_code": True,
                    }
                },
                "dataset_specs": [
                    {"path": Path("planner.yaml"), "include_gates": ["blocking"], "line": "expert"},
                    {"path": Path("extractor.yaml"), "include_gates": ["blocking"], "line": "expert"},
                    {"path": Path("workflow_extract.yaml"), "include_gates": ["blocking"], "line": "expert"},
                    {"path": Path("session_context.yaml"), "include_gates": ["blocking"], "line": "expert"},
                ],
            }

        tasks_by_path = {
            "planner.yaml": "planner",
            "extractor.yaml": "extractor",
            "workflow_extract.yaml": "workflow_extract",
            "session_context.yaml": "session_context",
        }
        seen: list[str] = []

        def fake_run_dataset_file(path, *, include_gates=None, dataset_line=None):
            del include_gates, dataset_line
            seen.append(str(path))
            return {
                "dataset": str(path),
                "task": tasks_by_path[str(path)],
                "line": "expert",
                "summary": {
                    "total": 1,
                    "passed": 1,
                    "failed": 0,
                    "blocking_failed": 0,
                    "pass_rate": 1.0,
                    "avg_score": 1.0,
                    "total_latency_ms": 1.0,
                    "avg_latency_ms": 1.0,
                    "p95_latency_ms": 1.0,
                    "estimated_input_tokens": 1,
                    "estimated_output_tokens": 1,
                    "estimated_total_tokens": 2,
                    "avg_estimated_total_tokens": 2.0,
                    "model_names": ["fake"],
                },
                "cases": [],
            }

        with patch("src.eval_platform.release_compare.resolve_dataset_specs", side_effect=fake_resolve_dataset_specs):
            with patch(
                "src.eval_platform.release_compare._dataset_task",
                side_effect=lambda path: tasks_by_path[str(path)],
            ):
                with patch("src.eval_platform.release_compare.run_dataset_file", side_effect=fake_run_dataset_file):
                    result = compare_release_candidates(
                        governance_file="src/eval_assets/governance.yaml",
                        blocking_profile="expert_blocking_gate",
                        regression_profile="expert_regression_gate",
                        baseline_llm_model="shared-llm",
                        baseline_extractor_model="baseline-extractor",
                        candidate_llm_model="shared-llm",
                        candidate_extractor_model="candidate-extractor",
                    )

        self.assertTrue(result["passed"])
        self.assertEqual(result["impacted_tasks"], ["extractor", "workflow_extract"])
        self.assertTrue(all(path in {"extractor.yaml", "workflow_extract.yaml"} for path in seen))

    def test_release_compare_detects_latency_and_token_regression(self) -> None:
        from src.eval_platform.release_compare import compare_release_candidates

        def fake_resolve_dataset_specs(*, governance_path=None, profile_name=None, dataset_args=None):
            del governance_path, dataset_args
            profile = {
                "policy": {
                    "max_blocking_failures": 0,
                    "min_pass_rate": 1.0,
                    "max_latency_regression_ratio": 1.2,
                    "max_total_tokens_regression_ratio": 1.1,
                    "enforce_exit_code": True,
                }
            }
            dataset_specs = [{"path": Path("regression.yaml"), "include_gates": ["blocking", "regression"], "line": "expert"}]
            if profile_name == "expert_blocking_gate":
                return {"profile": profile, "dataset_specs": [{"path": Path("blocking.yaml"), "include_gates": ["blocking"], "line": "expert"}]}
            return {"profile": profile, "dataset_specs": dataset_specs}

        def fake_run_dataset_file(path, *, include_gates=None, dataset_line=None):
            del include_gates, dataset_line
            current_model = os.environ.get("LLM_MODEL")
            avg_latency_ms = 100.0 if current_model != "candidate" else 130.0
            estimated_total_tokens = 100 if current_model != "candidate" else 120
            return {
                "dataset": str(path),
                "task": "planner",
                "line": "expert",
                "summary": {
                    "total": 2,
                    "passed": 2,
                    "failed": 0,
                    "blocking_failed": 0,
                    "pass_rate": 1.0,
                    "avg_score": 1.0,
                    "total_latency_ms": avg_latency_ms * 2,
                    "avg_latency_ms": avg_latency_ms,
                    "p95_latency_ms": avg_latency_ms,
                    "estimated_input_tokens": estimated_total_tokens // 2,
                    "estimated_output_tokens": estimated_total_tokens // 2,
                    "estimated_total_tokens": estimated_total_tokens,
                    "avg_estimated_total_tokens": estimated_total_tokens / 2,
                    "model_names": [current_model or "default"],
                },
                "cases": [],
            }

        with patch("src.eval_platform.release_compare.resolve_dataset_specs", side_effect=fake_resolve_dataset_specs):
            with patch("src.eval_platform.release_compare._dataset_task", return_value="planner"):
                with patch("src.eval_platform.release_compare.run_dataset_file", side_effect=fake_run_dataset_file):
                    result = compare_release_candidates(
                        governance_file="src/eval_assets/governance.yaml",
                        blocking_profile="expert_blocking_gate",
                        regression_profile="expert_regression_gate",
                        baseline_llm_model="baseline",
                        baseline_extractor_model=None,
                        candidate_llm_model="candidate",
                        candidate_extractor_model=None,
                    )

        self.assertFalse(result["passed"])
        reasons = result["comparison"]["regressions"][0]["reasons"]
        self.assertTrue(any("avg_latency_ms exceeded regression ratio" in item for item in reasons))
        self.assertTrue(any("estimated_total_tokens exceeded regression ratio" in item for item in reasons))

    def test_release_compare_skips_when_no_model_difference_exists(self) -> None:
        from src.eval_platform.release_compare import compare_release_candidates

        result = compare_release_candidates(
            governance_file="src/eval_assets/governance.yaml",
            blocking_profile="expert_blocking_gate",
            regression_profile="expert_regression_gate",
            baseline_llm_model="same-llm",
            baseline_extractor_model="same-extractor",
            candidate_llm_model="same-llm",
            candidate_extractor_model="same-extractor",
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["impacted_tasks"], [])
        self.assertTrue(result["candidate_blocking"]["profile_result"]["skipped"])

    def test_release_compare_main_writes_default_json_output(self) -> None:
        from src.eval_platform import release_compare

        tmp = self._make_tmpdir()
        try:
            out_path = tmp / "latest.json"
            payload = {
                "passed": True,
                "impacted_tasks": ["extractor"],
                "candidate_blocking": {"profile_result": {"passed": True}},
                "baseline_regression": {"profile_result": {"passed": True}},
                "candidate_regression": {"profile_result": {"passed": True}},
                "comparison": {"no_regressions": True, "regressions": []},
            }
            with patch(
                "src.eval_platform.release_compare.compare_release_candidates",
                return_value=payload,
            ):
                with patch(
                    "src.eval_platform.release_compare._default_json_out_path",
                    return_value=out_path,
                ):
                    exit_code = release_compare.main([])

            self.assertEqual(exit_code, 0)
            saved = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(saved["passed"])
            self.assertEqual(saved["impacted_tasks"], ["extractor"])
            self.assertIn("generated_at", saved)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_release_compare_main_prints_summary_only(self) -> None:
        from src.eval_platform import release_compare

        tmp = self._make_tmpdir()
        try:
            out_path = tmp / "latest.json"
            payload = {
                "passed": False,
                "impacted_tasks": ["planner", "variety_match"],
                "candidate_blocking": {"profile_result": {"passed": True}},
                "baseline_regression": {"profile_result": {"passed": True}},
                "candidate_regression": {"profile_result": {"passed": False}},
                "comparison": {
                    "no_regressions": False,
                    "regressions": [
                        {
                            "task": "planner",
                            "dataset": "src/eval_assets/expert/planner.yaml",
                            "reasons": ["pass_rate dropped"],
                        }
                    ],
                },
            }
            with patch(
                "src.eval_platform.release_compare.compare_release_candidates",
                return_value=payload,
            ):
                with patch(
                    "src.eval_platform.release_compare._default_json_out_path",
                    return_value=out_path,
                ):
                    with patch("builtins.print") as mocked_print:
                        exit_code = release_compare.main([])

            self.assertEqual(exit_code, 1)
            printed = "\n".join(
                " ".join(str(part) for part in call.args) for call in mocked_print.call_args_list
            )
            self.assertIn("compare passed=False", printed)
            self.assertIn("impacted_tasks=planner, variety_match", printed)
            self.assertIn("regressions=1", printed)
            self.assertNotIn("[planner]", printed)
            self.assertNotIn("pass 1/1", printed)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_release_compare_main_stops_on_invalid_model(self) -> None:
        from src.eval_platform import release_compare

        with patch(
            "src.eval_platform.release_compare._validate_compare_models",
            side_effect=ValueError("llm model does not exist or is not accessible: bad-model"),
        ):
            with patch("builtins.print") as mocked_print:
                exit_code = release_compare.main(
                    [
                        "--candidate-llm-model",
                        "bad-model",
                    ]
                )

        self.assertEqual(exit_code, 2)
        printed = "\n".join(
            " ".join(str(part) for part in call.args) for call in mocked_print.call_args_list
        )
        self.assertIn("model_validation_error=", printed)

    def test_import_promotion_payloads_upserts_into_expert_dataset(self) -> None:
        from src.eval_platform.promotion_import import import_promotion_payloads

        tmp = self._make_tmpdir()
        try:
            expert_root = tmp / "expert"
            expert_root.mkdir(parents=True, exist_ok=True)
            target = expert_root / "planner.yaml"
            target.write_text(
                "task: planner\nline: expert\nowner: business_expert\ncases:\n"
                "  - id: promoted_case\n"
                "    gate: regression\n"
                "    input:\n"
                "      prompt: old\n"
                "    expected:\n"
                "      action: none\n",
                encoding="utf-8",
            )
            promotion = tmp / "planner.promotion.yaml"
            promotion.write_text(
                "task: planner\nline: expert\nowner: production_audit_promotion\ncases:\n"
                "  - id: promoted_case\n"
                "    gate: blocking\n"
                "    input:\n"
                "      prompt: new\n"
                "    expected:\n"
                "      action: tool\n",
                encoding="utf-8",
            )

            result = import_promotion_payloads(
                promotion_files=[str(promotion)],
                expert_root=str(expert_root),
            )

            payload = json.loads(json.dumps(result))
            self.assertEqual(payload["imported"][0]["upserted_cases"], 1)
            updated = target.read_text(encoding="utf-8")
            self.assertIn("prompt: new", updated)
            self.assertIn("gate: blocking", updated)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_import_promotion_payloads_prunes_matching_production_audit_cases(self) -> None:
        from src.eval_platform.promotion_import import import_promotion_payloads

        tmp = self._make_tmpdir()
        try:
            expert_root = tmp / "expert"
            production_root = tmp / "production_audit"
            expert_root.mkdir(parents=True, exist_ok=True)
            production_root.mkdir(parents=True, exist_ok=True)

            (production_root / "planner.yaml").write_text(
                "task: planner\nline: production_audit\nowner: audit\ncases:\n"
                "  - id: interaction_240_sowing_query\n"
                "    gate: audit\n"
                "    source:\n"
                "      interaction_id: 240\n"
                "    input:\n"
                "      prompt: 我在湖南省常德种植早稻湘早籼24号，移栽什么时候播种合适\n"
                "    expected:\n"
                "      action: tool\n"
                "      name: sowing_suitability_lookup\n"
                "  - id: interaction_999_keep\n"
                "    gate: audit\n"
                "    source:\n"
                "      interaction_id: 999\n"
                "    input:\n"
                "      prompt: 你是？\n"
                "    expected:\n"
                "      action: none\n",
                encoding="utf-8",
            )

            promotion = tmp / "planner.promotion.yaml"
            promotion.write_text(
                "task: planner\nline: expert\nowner: production_audit_promotion\ncases:\n"
                "  - id: promoted_interaction_240_sowing_query\n"
                "    gate: regression\n"
                "    source:\n"
                "      interaction_id: 240\n"
                "      promotion_source: production_audit_review\n"
                "    input:\n"
                "      prompt: 我在湖南省常德种植早稻湘早籼24号，移栽什么时候播种合适\n"
                "    expected:\n"
                "      action: tool\n"
                "      name: sowing_suitability_lookup\n",
                encoding="utf-8",
            )

            result = import_promotion_payloads(
                promotion_files=[str(promotion)],
                expert_root=str(expert_root),
                production_audit_root=str(production_root),
            )

            self.assertEqual(result["pruned_production_audit"][0]["removed_cases"], 1)
            pruned = (production_root / "planner.yaml").read_text(encoding="utf-8")
            self.assertNotIn("interaction_240_sowing_query", pruned)
            self.assertIn("interaction_999_keep", pruned)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_audit_run_latest_writes_review_and_queue_dirs(self) -> None:
        from src.eval_platform.audit import main

        tmp = self._make_tmpdir()
        try:
            out_dir = tmp / "audit-run"

            with patch("src.eval_platform.audit.get_config") as mocked_cfg:
                mocked_cfg.return_value.interaction_store = "postgres"
                with patch("src.eval_platform.audit.load_interactions", return_value=[{"id": 1}]):
                    with patch(
                        "src.eval_platform.audit.build_production_audit_batches",
                        return_value={"planner": {"task": "planner", "cases": []}},
                    ):
                        batch_file = out_dir / "planner.yaml"
                        with patch(
                            "src.eval_platform.audit.save_production_audit_batches",
                            return_value=[batch_file],
                        ):
                            with patch(
                                "src.eval_platform.audit.build_review_records_from_batch",
                                return_value={"task": "planner", "records": []},
                            ):
                                with patch(
                                    "src.eval_platform.audit.build_human_review_queue",
                                    return_value={"records": []},
                                ):
                                    exit_code = main(
                                        [
                                            "run-latest",
                                            "--out-dir",
                                            str(out_dir),
                                        ]
                                    )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "reviews" / "planner.review.yaml").exists())
            self.assertTrue((out_dir / "queues" / "planner.review.queue.yaml").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_audit_promote_marks_review_and_refreshes_queue(self) -> None:
        from src.eval_platform.audit import main
        from src.eval_platform.audit_pipeline import yaml_load

        tmp = self._make_tmpdir()
        try:
            review_dir = tmp / "reviews"
            queue_dir = tmp / "queues"
            promotion_dir = tmp / "promotions"
            review_dir.mkdir(parents=True, exist_ok=True)
            queue_dir.mkdir(parents=True, exist_ok=True)
            review_path = review_dir / "planner.review.yaml"
            queue_path = queue_dir / "planner.review.queue.yaml"
            review_path.write_text(
                "task: planner\nrecords:\n"
                "  - id: interaction_240_sowing_query\n"
                "    task: planner\n"
                "    source:\n"
                "      interaction_id: 240\n"
                "    input:\n"
                "      prompt: 我在湖南省常德种植早稻湘早籼24号，移栽什么时候播种合适\n"
                "    expected:\n"
                "      action: tool\n"
                "      name: sowing_suitability_lookup\n"
                "    ai_judge:\n"
                "      verdict: fail\n"
                "      confidence: 0.2\n"
                "      risk: high\n"
                "      suggested_gate: blocking\n"
                "    human_review:\n"
                "      status: promote_to_expert\n"
                "      notes: confirmed by reviewer\n",
                encoding="utf-8",
            )
            queue_path.write_text(
                "line: production_audit\nrecords:\n"
                "  - id: interaction_240_sowing_query\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "promote",
                    "--review",
                    str(review_path),
                    "--out-dir",
                    str(promotion_dir),
                ]
            )

            self.assertEqual(exit_code, 0)
            updated_review = yaml_load(review_path)
            human_review = updated_review["records"][0]["human_review"]
            self.assertEqual(human_review["status"], "promote_to_expert")
            self.assertIn("promotion_exported_at", human_review)
            self.assertIn("promotion_file", human_review)

            updated_queue = yaml_load(queue_path)
            self.assertEqual(updated_queue["records"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_audit_export_csv_and_import_csv_round_trip(self) -> None:
        from src.eval_platform.audit import main
        from src.eval_platform.audit_pipeline import yaml_load

        tmp = self._make_tmpdir()
        try:
            review_dir = tmp / "reviews"
            queue_dir = tmp / "queues"
            csv_dir = tmp / "csv"
            review_dir.mkdir(parents=True, exist_ok=True)
            queue_dir.mkdir(parents=True, exist_ok=True)
            review_path = review_dir / "planner.review.yaml"
            queue_path = queue_dir / "planner.review.queue.yaml"
            review_path.write_text(
                "task: planner\nrecords:\n"
                "  - id: interaction_240_sowing_query\n"
                "    task: planner\n"
                "    source:\n"
                "      interaction_id: 240\n"
                "    input:\n"
                "      prompt: 我在湖南省常德种植早稻湘早籼24号，移栽什么时候播种合适\n"
                "    expected:\n"
                "      action: tool\n"
                "      name: sowing_suitability_lookup\n"
                "    observed_output:\n"
                "      tool_name: sowing_suitability_lookup\n"
                "    ai_judge:\n"
                "      verdict: fail\n"
                "      confidence: 0.2\n"
                "      risk: high\n"
                "      rationale: missing field\n"
                "      findings: [bad]\n"
                "    human_review:\n"
                "      status: pending\n",
                encoding="utf-8",
            )
            queue_path.write_text(
                "line: production_audit\n"
                f"source_review_file: {review_path.as_posix()}\n"
                "records:\n"
                "  - id: interaction_240_sowing_query\n"
                "    task: planner\n"
                "    source:\n"
                "      interaction_id: 240\n"
                "    input:\n"
                "      prompt: 我在湖南省常德种植早稻湘早籼24号，移栽什么时候播种合适\n"
                "    expected:\n"
                "      action: tool\n"
                "      name: sowing_suitability_lookup\n"
                "    observed_output:\n"
                "      tool_name: sowing_suitability_lookup\n"
                "    ai_judge:\n"
                "      verdict: fail\n"
                "      confidence: 0.2\n"
                "      risk: high\n"
                "      rationale: missing field\n"
                "      findings: [bad]\n"
                "    human_review:\n"
                "      status: pending\n",
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "export-csv",
                    "--queue",
                    str(queue_path),
                    "--out-dir",
                    str(csv_dir),
                ]
            )
            self.assertEqual(exit_code, 0)

            csv_path = csv_dir / "planner.review.queue.csv"
            self.assertTrue(csv_path.exists())

            with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            rows[0]["human_status"] = "promote_to_expert"
            rows[0]["reviewer"] = "expert_a"
            rows[0]["target_gate"] = "blocking"
            rows[0]["notes"] = "confirmed in manual review"
            rows[0]["corrected_expected_json"] = json.dumps(
                {"action": "tool", "name": "sowing_suitability_lookup"},
                ensure_ascii=False,
            )
            with csv_path.open("w", encoding="utf-8-sig", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            exit_code = main(["import-csv", "--csv", str(csv_path)])
            self.assertEqual(exit_code, 0)

            updated_review = yaml_load(review_path)
            human_review = updated_review["records"][0]["human_review"]
            self.assertEqual(human_review["status"], "promote_to_expert")
            self.assertEqual(human_review["reviewer"], "expert_a")
            self.assertEqual(human_review["target_gate"], "blocking")
            self.assertEqual(
                human_review["corrected_expected"]["name"],
                "sowing_suitability_lookup",
            )

            updated_queue = yaml_load(queue_path)
            self.assertEqual(updated_queue["records"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
