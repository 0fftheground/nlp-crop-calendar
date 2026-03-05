import calendar
import importlib.util
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
    from src.agent.workflows.common import build_fallback_planting
    from src.application.services.crop_calendar_service import (
        _build_crop_calendar_payload,
        _build_operation_plan_from_farmworks,
        build_operation_plan,
        fetch_weather,
    )
    from src.domain.planting import extract_planting_details
    from src.infra.config import get_config
    from src.schemas import (
        PlantingDetails,
        PlantingDetailsDraft,
        WeatherQueryInput,
        WeatherSeries,
    )


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class DomainServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["WEATHER_PROVIDER"] = "mock"
        os.environ["CROP_CALENDAR_PROVIDER"] = "mock"
        get_config.cache_clear()

    def test_fetch_weather_returns_series(self) -> None:
        query = WeatherQueryInput(
            region="test",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 30),
            year=2025,
        )
        series = fetch_weather(query)
        self.assertEqual(series.region, "test")
        expected_days = 30
        self.assertEqual(len(series.points), expected_days)
        self.assertEqual(series.start_date, date(2025, 1, 1))
        self.assertEqual(series.end_date, date(2025, 1, 30))

    def test_build_operation_plan_returns_ops(self) -> None:
        planting = PlantingDetails(
            crop="水稻",
            planting_method="direct_seeding",
            sowing_date=date(2025, 1, 1),
        )
        weather_series = WeatherSeries(
            region="test",
            granularity="daily",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 3),
            points=[],
            source="synthetic",
        )
        plan = build_operation_plan(planting, weather_series)
        self.assertTrue(plan.operations)

    def test_build_operation_plan_from_farmworks_sorts_by_date(self) -> None:
        farmworks = {
            "追肥": "2025-05-20",
            "整地": "2025-05-01",
            "收获": "2025-09-30",
        }
        plan = _build_operation_plan_from_farmworks(farmworks, crop="水稻")
        titles = [op.title for op in plan.operations]
        self.assertEqual(titles, ["整地", "追肥", "收获"])

    def test_extract_planting_details_prefers_specific_culti_type(self) -> None:
        draft = extract_planting_details("4月20日播种，一季晚稻")
        self.assertEqual(draft.culti_type, "一季晚稻")

    def test_extract_planting_details_upgrades_llm_culti_type_alias(self) -> None:
        draft = extract_planting_details(
            "一季晚稻",
            llm_extract=lambda _: {"culti_type": "晚稻"},
        )
        self.assertEqual(draft.culti_type, "一季晚稻")

    def test_extract_planting_details_infers_region_hint(self) -> None:
        draft = extract_planting_details("我想在湖南省种植水稻，5月1日播种")
        self.assertEqual(draft.region_id, "湖南省")

    def test_build_crop_calendar_payload_allows_empty_transplant_date(self) -> None:
        os.environ["DEFAULT_FARM_ID"] = "1"
        get_config.cache_clear()
        planting = PlantingDetails(
            crop="水稻",
            variety="美香占2号",
            planting_method="transplanting",
            sowing_date=date(2025, 5, 1),
            transplant_date=None,
        )
        with patch(
            "src.application.services.crop_calendar_service._fetch_variety_id_by_name",
            return_value=1001,
        ):
            payload = _build_crop_calendar_payload(planting)
        self.assertEqual(payload.get("transp_date"), "")

    def test_build_fallback_planting_keeps_transplant_date_empty(self) -> None:
        draft = PlantingDetailsDraft(
            crop="水稻",
            planting_method="transplanting",
            sowing_date=date(2025, 5, 1),
            transplant_date=None,
        )
        planting = build_fallback_planting(draft)
        self.assertIsNone(planting.transplant_date)

    def test_planting_draft_to_canonical_keeps_farm_and_region_id(self) -> None:
        draft = PlantingDetailsDraft(
            farm_id="88",
            region_id="320100",
            crop="水稻",
            planting_method="direct_seeding",
            sowing_date=date(2025, 5, 1),
        )
        planting = draft.to_canonical()
        self.assertEqual(planting.farm_id, "88")
        self.assertEqual(planting.region_id, "320100")

    def test_build_crop_calendar_payload_uses_default_farm_id(self) -> None:
        os.environ["DEFAULT_FARM_ID"] = "1"
        get_config.cache_clear()
        planting = PlantingDetails(
            farm_id="99",
            crop="水稻",
            variety="美香占2号",
            planting_method="transplanting",
            sowing_date=date(2025, 5, 1),
            transplant_date=None,
        )
        with patch(
            "src.application.services.crop_calendar_service._fetch_variety_id_by_name",
            return_value=1001,
        ):
            payload = _build_crop_calendar_payload(planting)
        self.assertEqual(payload.get("farm_id"), 1)

    def test_build_crop_calendar_payload_prefers_region_id(self) -> None:
        os.environ["DEFAULT_FARM_ID"] = "1"
        get_config.cache_clear()
        planting = PlantingDetails(
            farm_id="99",
            region_id="320100",
            crop="水稻",
            variety="美香占2号",
            planting_method="transplanting",
            sowing_date=date(2025, 5, 1),
            transplant_date=None,
        )
        with patch(
            "src.application.services.crop_calendar_service._fetch_variety_id_by_name",
            return_value=1001,
        ):
            payload = _build_crop_calendar_payload(planting)
        self.assertEqual(payload.get("region_id"), 320100)
        self.assertNotIn("farm_id", payload)

    def test_build_crop_calendar_payload_resolves_region_name(self) -> None:
        os.environ["DEFAULT_FARM_ID"] = "1"
        get_config.cache_clear()
        planting = PlantingDetails(
            region_id="湖南省",
            crop="水稻",
            variety="美香占2号",
            planting_method="transplanting",
            sowing_date=date(2025, 5, 1),
            transplant_date=None,
        )
        with patch(
            "src.application.services.crop_calendar_service._fetch_variety_id_by_name",
            return_value=1001,
        ), patch(
            "src.application.services.crop_calendar_service._resolve_region_id_for_payload",
            return_value=430000,
        ):
            payload = _build_crop_calendar_payload(planting)
        self.assertEqual(payload.get("region_id"), 430000)
        self.assertNotIn("farm_id", payload)

    def test_build_crop_calendar_payload_errors_when_region_unmatched(self) -> None:
        os.environ["DEFAULT_FARM_ID"] = "1"
        get_config.cache_clear()
        planting = PlantingDetails(
            region_id="火星基地",
            crop="水稻",
            variety="美香占2号",
            planting_method="transplanting",
            sowing_date=date(2025, 5, 1),
            transplant_date=None,
        )
        with patch(
            "src.application.services.crop_calendar_service._fetch_variety_id_by_name",
            return_value=1001,
        ), patch(
            "src.application.services.crop_calendar_service._resolve_region_id_for_payload",
            return_value=None,
        ):
            with self.assertRaises(RuntimeError):
                _build_crop_calendar_payload(planting)


if __name__ == "__main__":
    unittest.main()
