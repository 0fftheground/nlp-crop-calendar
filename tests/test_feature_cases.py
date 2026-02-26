import calendar
import importlib.util
import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from src.agent.tools.weather import weather_lookup
    from src.agent.workflows.growth_stage_graph import build_growth_stage_graph
    from src.application.services import variety_service
    from src.infra.config import get_config
    from src.infra.tool_cache import get_tool_result_cache
    from src.schemas import (
        GrowthStageResult,
        PlantingDetailsDraft,
    )


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class FeatureCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "WEATHER_PROVIDER": os.environ.get("WEATHER_PROVIDER"),
            "TOOL_CACHE_STORE": os.environ.get("TOOL_CACHE_STORE"),
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
        }
        os.environ["WEATHER_PROVIDER"] = "mock"
        os.environ["TOOL_CACHE_STORE"] = "memory"
        os.environ["PENDING_STORE"] = "memory"
        get_config.cache_clear()
        get_tool_result_cache.cache_clear()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_config.cache_clear()
        get_tool_result_cache.cache_clear()

    def test_weather_lookup_returns_series(self) -> None:
        payload = json.dumps(
            {
                "region": "长沙",
                "start_date": "2025-01-01",
                "end_date": "2025-01-03",
            }
        )
        result = weather_lookup(payload)
        self.assertEqual(result.name, "growth_weather_lookup")
        data = result.data or {}
        self.assertEqual(data.get("region"), "farm:1")
        points = data.get("points") or []
        self.assertEqual(len(points), 3)
        self.assertEqual(data.get("start_date"), "2025-01-01")
        self.assertEqual(data.get("end_date"), "2025-01-03")

    def test_variety_lookup_requires_exact_name(self) -> None:
        prompt = json.dumps(
            {"prompt": "美香占", "planting": {"crop": "水稻"}}
        )
        records = [{"品种名称": "美香占2号", "审定区域": "湖南"}]
        raw_records = [{"variety_name": "美香占2号", "approval_region": "湖南"}]
        with patch(
            "src.application.services.variety_service._lookup_variety_records",
            return_value=(records, raw_records),
        ), patch(
            "src.application.services.variety_service.retrieve_variety_candidates",
            return_value=["美香占2号"],
        ):
            result = variety_service.lookup_variety(prompt)
        self.assertEqual(result.name, "variety_lookup")
        self.assertIn("未找到完全匹配的品种", result.message)
        self.assertIn("美香占2号", result.message)
        self.assertEqual(result.data.get("missing_fields"), ["variety"])

    def test_variety_followup_region_choice_keeps_selected_variety(self) -> None:
        original_query = "帮我查询品种美香占"
        candidate_followup_prompt = json.dumps(
            {
                "user_id": "u1",
                "query": original_query,
                "followup": {
                    "prompt": "1",
                    "draft": {
                        "candidates": [
                            "美香占2号",
                            "华两优美香新占",
                            "荃优美香银占3号",
                            "Q两优银泰香占",
                            "荃优美香丝苗1号",
                        ]
                    },
                    "missing_fields": ["variety"],
                    "followup_count": 0,
                },
            },
            ensure_ascii=False,
        )

        def fake_lookup(prompt, *, limit=5, confirmed_candidate=None):
            prompt_text = str(prompt)
            if confirmed_candidate == "美香占2号":
                records = [
                    {"品种名称": "美香占2号", "审定区域": "湖南", "审定年份": "2020"},
                    {"品种名称": "美香占2号", "审定区域": "芜湖", "审定年份": "2021"},
                ]
                raw = [
                    {"variety_name": "美香占2号", "approval_region": "湖南"},
                    {"variety_name": "美香占2号", "approval_region": "芜湖"},
                ]
                return records, raw
            # Simulate the old bug trigger: nested candidate JSON pollutes the next query.
            if "荃优美香丝苗1号" in prompt_text:
                records = [{"品种名称": "荃优美香丝苗1号", "审定区域": "湖南"}]
                raw = [{"variety_name": "荃优美香丝苗1号", "approval_region": "湖南"}]
                return records, raw
            records = [
                {"品种名称": "美香占2号", "审定区域": "湖南", "审定年份": "2020"},
                {"品种名称": "美香占2号", "审定区域": "芜湖", "审定年份": "2021"},
            ]
            raw = [
                {"variety_name": "美香占2号", "approval_region": "湖南"},
                {"variety_name": "美香占2号", "approval_region": "芜湖"},
            ]
            return records, raw

        with patch(
            "src.application.services.variety_service._lookup_variety_records",
            side_effect=fake_lookup,
        ):
            step1 = variety_service.lookup_variety(candidate_followup_prompt)
            self.assertEqual(step1.name, "variety_lookup")
            self.assertIn("请选择要查看的区域", step1.message)
            self.assertEqual(step1.data.get("query"), original_query)

            region_followup_prompt = json.dumps(
                {
                    "user_id": "u1",
                    "query": step1.data.get("query"),
                    "followup": {
                        "prompt": "1",
                        "draft": step1.data.get("draft"),
                        "missing_fields": step1.data.get("missing_fields"),
                        "followup_count": 1,
                    },
                },
                ensure_ascii=False,
            )
            step2 = variety_service.lookup_variety(region_followup_prompt)

        self.assertEqual(step2.name, "variety_lookup")
        self.assertIn("已返回品种 美香占2号 的审定信息", step2.message)
        self.assertEqual((step2.data or {}).get("variety"), "美香占2号")

    def test_growth_stage_prediction_output(self) -> None:
        draft = PlantingDetailsDraft(
            crop="水稻",
            planting_method="direct_seeding",
            sowing_date=date(2025, 4, 3),
        )
        planting = draft.to_canonical()
        # weather lookup is no longer required for growth stage workflow
        stages = {
            "stage_dates": json.dumps(
                {"三叶一心": "2025-05-01", "成熟期": "2025-08-09"},
                ensure_ascii=False,
            )
        }
        growth_payload = GrowthStageResult(stages=stages)
        with patch(
            "src.agent.workflows.growth_stage_graph.extract_planting_details",
            return_value=draft,
        ), patch(
            "src.agent.workflows.growth_stage_graph.search_planting_plans",
            return_value=([{"id": 1}], "id", ["id"]),
        ), patch(
            "src.agent.workflows.growth_stage_graph.resolve_planting_from_plan_id",
            return_value=planting,
        ), patch(
            "src.agent.workflows.growth_stage_graph.query_growth_stage_from_plan_id",
            return_value=growth_payload,
        ):
            graph = build_growth_stage_graph()
            state = graph.invoke(
                {
                    "user_prompt": "在常德鼎城区种水稻，播种日期2025-04-03，直播",
                    "trace": [],
                    "user_id": "u1",
                }
            )
        message = state.get("message", "")
        self.assertIn("种植信息", message)
        self.assertIn("三叶一心：2025-05-01", message)
        self.assertIn("成熟期：2025-08-09", message)
        self.assertNotIn("积温", message)
        self.assertNotIn("气象信息", message)
        self.assertNotIn("品种信息", message)


if __name__ == "__main__":
    unittest.main()
