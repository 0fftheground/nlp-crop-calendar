import calendar
import importlib.util
import json
import os
import sys
import unittest
from datetime import date, datetime, time
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
        ToolInvocation,
        WeatherDataPoint,
        WeatherSeries,
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
        payload = json.dumps({"region": "长沙", "year": 2025})
        result = weather_lookup(payload)
        self.assertEqual(result.name, "weather_lookup")
        data = result.data or {}
        self.assertEqual(data.get("region"), "长沙")
        points = data.get("points") or []
        expected_days = 366 if calendar.isleap(2025) else 365
        self.assertEqual(len(points), expected_days)
        self.assertEqual(data.get("start_date"), "2025-01-01")
        self.assertEqual(data.get("end_date"), "2025-12-31")

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
            return_value=[],
        ):
            result = variety_service.lookup_variety(prompt)
        self.assertEqual(result.name, "variety_lookup")
        self.assertIn("补充完整品种名称", result.message)
        self.assertEqual(result.data.get("missing_fields"), ["variety"])

    def test_growth_stage_prediction_output(self) -> None:
        draft = PlantingDetailsDraft(
            crop="水稻",
            planting_method="direct_seeding",
            sowing_date=date(2025, 4, 3),
            region="常德鼎城区",
        )
        series = WeatherSeries(
            region="常德鼎城区",
            granularity="daily",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
            points=[
                WeatherDataPoint(
                    timestamp=datetime.combine(date(2025, 1, 1), time.min),
                    temperature=20.0,
                )
            ],
            source="test",
        )
        weather_payload = ToolInvocation(
            name="growth_weather_lookup",
            message="ok",
            data=series.model_dump(mode="json"),
        )
        variety_payload = ToolInvocation(
            name="variety_lookup",
            message="ok",
            data={"variety": "美香占", "raw_selected": {}, "missing_fields": []},
        )
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
            "src.agent.workflows.growth_stage_graph.execute_tool",
            return_value=variety_payload,
        ), patch(
            "src.agent.workflows.growth_stage_graph.lookup_goso_weather",
            return_value=weather_payload,
        ), patch(
            "src.agent.workflows.growth_stage_graph.predict_growth_stage_local",
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
        self.assertIn("三叶一心: 2025-05-01", message)
        self.assertIn("成熟期: 2025-08-09", message)
        self.assertNotIn("积温", message)
        self.assertNotIn("气象信息", message)
        self.assertNotIn("品种信息", message)


if __name__ == "__main__":
    unittest.main()
