import importlib.util
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TMP_ROOT = ROOT / ".cache" / "test_tmp"

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class AuditPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        TMP_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = TMP_ROOT / "audit-pipeline"
        self.tmp.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_batches_deidentifies_and_splits_tasks(self) -> None:
        from src.eval_platform.audit_pipeline import build_production_audit_batches

        rows = [
            {
                "id": 1,
                "created_at": 1,
                "session_id": "s1",
                "mode": "tool",
                "prompt": "今天适合施肥吗，我邮箱是 a@test.com",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {
                                "name": "weather_lookup",
                                "message": "ok",
                                "data": {},
                            },
                        }
                    }
                },
                "response_json": {},
            },
            {
                "id": 2,
                "created_at": 2,
                "session_id": "s2",
                "mode": "tool",
                "prompt": "我在常德种早稻湘早籼24号，移栽什么时候播种合适",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {
                                "name": "sowing_suitability_lookup",
                                "data": {
                                    "resolved": {
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
                },
                "response_json": {},
            },
            {
                "id": 3,
                "created_at": 3,
                "session_id": "s3",
                "mode": "tool",
                "prompt": "1",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {
                                "name": "variety_lookup",
                                "data": {
                                    "query": "美香占2号",
                                    "region_choice": "湖南",
                                    "raw_matches": [
                                        {
                                            "variety_name": "美香占2号",
                                            "approval_region": "湖南",
                                        },
                                        {
                                            "variety_name": "美香占2号",
                                            "approval_region": "芜湖",
                                        },
                                    ],
                                    "raw_selected": {
                                        "variety_name": "美香占2号",
                                        "approval_region": "湖南",
                                    },
                                },
                            },
                        }
                    }
                },
                "response_json": {},
            },
        ]
        batches = build_production_audit_batches(rows, store_name="postgres")

        self.assertEqual(len(batches["planner"]["cases"]), 3)
        self.assertEqual(len(batches["extractor"]["cases"]), 1)
        self.assertEqual(len(batches["variety_match"]["cases"]), 1)
        self.assertEqual(len(batches["planner.context_dependent"]["cases"]), 0)
        planner_prompt = batches["planner"]["cases"][0]["input"]["prompt"]
        self.assertIn("<email>", planner_prompt)

    def test_build_batches_includes_workflow_planner_cases(self) -> None:
        from src.eval_platform.audit_pipeline import build_production_audit_batches

        rows = [
            {
                "id": 21,
                "created_at": 21,
                "session_id": "s-workflow",
                "mode": "workflow",
                "prompt": "帮我生成一个水稻种植计划，品种是美香占2号",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "workflow",
                            "plan": {
                                "message": "已生成农事方案。",
                                "data": {
                                    "workflow_name": "crop_calendar_workflow",
                                    "planting": {
                                        "crop": "水稻",
                                        "variety": "美香占2号",
                                    },
                                },
                            },
                        }
                    }
                },
                "response_json": {},
            }
        ]

        batches = build_production_audit_batches(rows, store_name="postgres")

        self.assertEqual(len(batches["planner"]["cases"]), 1)
        case = batches["planner"]["cases"][0]
        self.assertEqual(case["expected"]["action"], "workflow")
        self.assertEqual(case["expected"]["name"], "crop_calendar_workflow")
        self.assertEqual(case["observed_output"]["workflow_name"], "crop_calendar_workflow")

    def test_workflow_planner_case_keeps_thread_context_and_state(self) -> None:
        from src.eval_platform.audit_pipeline import build_production_audit_batches

        rows = [
            {
                "id": 40,
                "created_at": 40,
                "session_id": "s-workflow-thread",
                "mode": "workflow",
                "prompt": "帮我创建水稻种植计划",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "workflow",
                            "plan": {
                                "message": "还缺少地区信息。",
                                "data": {
                                    "workflow_name": "crop_calendar_workflow",
                                    "workflow_state": {
                                        "draft": {"crop": "水稻"},
                                        "missing_fields": ["region_id"],
                                        "pending_message": "请补充地区。",
                                    },
                                },
                            },
                        }
                    }
                },
                "response_json": {},
            },
            {
                "id": 41,
                "created_at": 41,
                "session_id": "s-workflow-thread",
                "mode": "workflow",
                "prompt": "地区是常德，继续生成计划",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "workflow",
                            "plan": {
                                "message": "已生成农事方案。",
                                "data": {
                                    "workflow_name": "crop_calendar_workflow",
                                    "workflow_state": {
                                        "draft": {"crop": "水稻", "region_id": "常德"},
                                        "missing_fields": [],
                                    },
                                },
                            },
                        }
                    }
                },
                "response_json": {},
            },
        ]

        batches = build_production_audit_batches(rows, store_name="postgres")

        self.assertEqual(len(batches["planner"]["cases"]), 2)
        case = batches["planner"]["cases"][1]
        self.assertEqual(case["source"]["context_window_kind"], "workflow_thread")
        self.assertEqual(len(case["source"]["context_window"]), 1)
        self.assertEqual(case["source"]["context_window"][0]["prompt"], "帮我创建水稻种植计划")
        self.assertEqual(case["observed_output"]["workflow_state"]["draft"]["region_id"], "常德")

    def test_build_batches_splits_context_dependent_followups(self) -> None:
        from src.eval_platform.audit_pipeline import build_production_audit_batches

        rows = [
            {
                "id": 10,
                "created_at": 10,
                "session_id": "s-followup",
                "mode": "tool",
                "prompt": "我在常德种早稻湘早籼24号，移栽什么时候播种合适",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {
                                "name": "sowing_suitability_lookup",
                                "message": "success",
                                "data": {
                                    "resolved": {
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
                },
                "response_json": {},
            },
            {
                "id": 11,
                "created_at": 11,
                "session_id": "s-followup",
                "mode": "tool",
                "prompt": "芜湖呢",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {
                                "name": "sowing_suitability_lookup",
                                "message": "success",
                                "data": {
                                    "resolved": {
                                        "crop": "水稻",
                                        "variety": "湘早籼24号",
                                        "region_id": "芜湖",
                                        "culti_type": "早稻",
                                        "planting_method": "transplanting",
                                    }
                                },
                            },
                        }
                    }
                },
                "response_json": {},
            },
        ]

        batches = build_production_audit_batches(rows, store_name="postgres")

        self.assertEqual(len(batches["planner"]["cases"]), 1)
        self.assertEqual(len(batches["planner.context_dependent"]["cases"]), 1)
        context_case = batches["planner.context_dependent"]["cases"][0]
        self.assertEqual(context_case["source"]["sampling_scope"], "context_dependent")
        self.assertEqual(
            context_case["source"]["context_window"][0]["prompt"],
            "我在常德种早稻湘早籼24号，移栽什么时候播种合适",
        )

    def test_build_batches_context_window_keeps_multiple_prior_turns(self) -> None:
        from src.eval_platform.audit_pipeline import build_production_audit_batches

        rows = [
            {
                "id": 30,
                "created_at": 30,
                "session_id": "s-window",
                "mode": "tool",
                "prompt": "今天适合施肥吗",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {"name": "weather_lookup", "message": "ok", "data": {}},
                        }
                    }
                },
                "response_json": {},
            },
            {
                "id": 31,
                "created_at": 31,
                "session_id": "s-window",
                "mode": "tool",
                "prompt": "芜湖呢",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {"name": "weather_lookup", "message": "ok", "data": {}},
                        }
                    }
                },
                "response_json": {},
            },
            {
                "id": 32,
                "created_at": 32,
                "session_id": "s-window",
                "mode": "tool",
                "prompt": "下周呢",
                "request_json": {
                    "raw": {
                        "response": {
                            "mode": "tool",
                            "tool": {"name": "weather_lookup", "message": "ok", "data": {}},
                        }
                    }
                },
                "response_json": {},
            },
        ]

        batches = build_production_audit_batches(rows, store_name="postgres")

        context_case = batches["planner.context_dependent"]["cases"][-1]
        prompts = [item["prompt"] for item in context_case["source"]["context_window"]]
        self.assertEqual(prompts, ["今天适合施肥吗", "芜湖呢"])

    def test_build_sampling_watermark_tracks_latest_row(self) -> None:
        from src.eval_platform.audit_pipeline import build_sampling_watermark

        watermark = build_sampling_watermark(
            [
                {"id": 4, "created_at": 100},
                {"id": 6, "created_at": 100},
                {"id": 5, "created_at": 99},
            ]
        )

        self.assertEqual(watermark["last_created_at"], 100)
        self.assertEqual(watermark["last_id"], 6)

    def test_default_sampling_state_path_uses_state_directory(self) -> None:
        from src.eval_platform.audit import _default_state_path

        self.assertEqual(
            _default_state_path().as_posix(),
            ".state/eval/production_audit/sampling_state.json",
        )

    def test_default_audit_output_dirs_are_separated_by_step(self) -> None:
        from src.eval_platform.audit import (
            _default_batch_dir,
            _default_run_dir,
            _default_step_dir,
        )

        self.assertIn("/production_audit/batches/", _default_batch_dir().as_posix())
        self.assertIn("/production_audit/runs/", _default_run_dir().as_posix())
        self.assertEqual(
            _default_step_dir("reviews").as_posix(),
            ".cache/eval/production_audit/reviews",
        )
        self.assertEqual(
            _default_step_dir("queues").as_posix(),
            ".cache/eval/production_audit/queues",
        )
        self.assertEqual(
            _default_step_dir("promotions").as_posix(),
            ".cache/eval/production_audit/promotions",
        )

    def test_build_human_review_queue_filters_low_confidence_or_failures(self) -> None:
        from src.eval_platform.audit_pipeline import build_human_review_queue

        review_payload = {
            "records": [
                {
                    "id": "a",
                    "task": "planner",
                    "source": {},
                    "input": {"prompt": "a"},
                    "expected": {},
                    "observed_output": {},
                    "ai_judge": {"verdict": "pass", "confidence": 0.95},
                    "human_review": {"status": "pending"},
                },
                {
                    "id": "b",
                    "task": "planner",
                    "source": {},
                    "input": {"prompt": "b"},
                    "expected": {},
                    "observed_output": {},
                    "ai_judge": {"verdict": "needs_human_review", "confidence": 0.8},
                    "human_review": {"status": "pending"},
                },
            ]
        }
        queue = build_human_review_queue(review_payload, max_confidence_auto_pass=0.9)
        self.assertEqual(len(queue["records"]), 1)
        self.assertEqual(queue["records"][0]["id"], "b")

    def test_build_human_review_queue_skips_non_pending_records(self) -> None:
        from src.eval_platform.audit_pipeline import build_human_review_queue

        review_payload = {
            "records": [
                {
                    "id": "promoted",
                    "task": "planner",
                    "source": {},
                    "input": {"prompt": "a"},
                    "expected": {},
                    "observed_output": {},
                    "ai_judge": {"verdict": "fail", "confidence": 0.2},
                    "human_review": {
                        "status": "promote_to_expert",
                        "promotion_exported_at": "2026-03-24T10:00:00Z",
                    },
                },
                {
                    "id": "pending",
                    "task": "planner",
                    "source": {},
                    "input": {"prompt": "b"},
                    "expected": {},
                    "observed_output": {},
                    "ai_judge": {"verdict": "fail", "confidence": 0.2},
                    "human_review": {"status": "pending"},
                },
            ]
        }

        queue = build_human_review_queue(review_payload, max_confidence_auto_pass=0.9)
        self.assertEqual(len(queue["records"]), 1)
        self.assertEqual(queue["records"][0]["id"], "pending")

    def test_build_promotion_candidates_requires_human_promotion(self) -> None:
        from src.eval_platform.audit_pipeline import build_promotion_candidates

        review_payload = {
            "records": [
                {
                    "id": "x",
                    "task": "extractor",
                    "source": {"interaction_id": 12},
                    "input": {"prompt": "foo"},
                    "expected": {"crop": "水稻"},
                    "ai_judge": {"suggested_gate": "blocking"},
                    "human_review": {
                        "status": "promote_to_expert",
                        "notes": "expert confirmed",
                        "corrected_expected": {
                            "crop": "水稻",
                            "variety": "南粳46",
                        },
                    },
                }
            ]
        }
        grouped = build_promotion_candidates(review_payload)
        self.assertIn("extractor", grouped)
        case = grouped["extractor"]["cases"][0]
        self.assertEqual(case["gate"], "blocking")
        self.assertEqual(case["expected"]["variety"], "南粳46")
