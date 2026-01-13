import calendar
import importlib.util
import os
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from src.application.services.crop_calendar_service import (
        build_operation_plan,
        fetch_weather,
    )
    from src.infra.config import get_config
    from src.schemas import PlantingDetails, WeatherQueryInput, WeatherSeries


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class DomainServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ["WEATHER_PROVIDER"] = "mock"
        os.environ["RECOMMENDATION_PROVIDER"] = "mock"
        get_config.cache_clear()

    def test_fetch_weather_returns_series(self) -> None:
        query = WeatherQueryInput(
            region="test",
            year=2025,
        )
        series = fetch_weather(query)
        self.assertEqual(series.region, "test")
        expected_days = 366 if calendar.isleap(2025) else 365
        self.assertEqual(len(series.points), expected_days)
        self.assertEqual(series.start_date, date(2025, 1, 1))
        self.assertEqual(series.end_date, date(2025, 12, 31))

    def test_build_operation_plan_returns_ops(self) -> None:
        planting = PlantingDetails(
            crop="水稻",
            planting_method="direct_seeding",
            sowing_date=date(2025, 1, 1),
            region="test",
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


if __name__ == "__main__":
    unittest.main()
