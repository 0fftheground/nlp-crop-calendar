import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from src.agent.tools.weather import weather_lookup
    from src.infra.config import get_config
    from src.infra.tool_cache import get_tool_result_cache
    from src.schemas import ToolInvocation


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class WeatherFeatureCaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "TOOL_CACHE_STORE": os.environ.get("TOOL_CACHE_STORE"),
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
        }
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
        self.assertEqual(result.name, "weather_lookup")
        data = result.data or {}
        self.assertEqual(data.get("region"), "farm:1")
        points = data.get("points") or []
        self.assertEqual(len(points), 3)
        self.assertEqual(data.get("start_date"), "2025-01-01")
        self.assertEqual(data.get("end_date"), "2025-01-03")

    def test_weather_lookup_reuses_cached_series_across_operations(self) -> None:
        first_prompt = json.dumps({"query": "最近适合打药吗"}, ensure_ascii=False)
        second_prompt = json.dumps({"query": "最近适合施肥吗"}, ensure_ascii=False)
        fake_result = ToolInvocation(
            name="weather_lookup",
            message="ok",
            data={
                "region": "farm:1",
                "start_date": "2026-03-12",
                "end_date": "2026-03-18",
                "points": [
                    {
                        "timestamp": "2026-03-12T00:00:00",
                        "dy_ws": 0.9,
                        "sf_ws": 0.5,
                    }
                ],
            },
        )
        call_count = {"value": 0}

        def fake_lookup(prompt: str, *, cache_prompt=None, query=None):
            call_count["value"] += 1
            return fake_result

        with patch("src.agent.tools.weather.lookup_weather", side_effect=fake_lookup):
            first = weather_lookup(first_prompt)
            second = weather_lookup(second_prompt)

        self.assertEqual(call_count["value"], 1)
        self.assertEqual(first.data.get("requested_operations"), ["打药"])
        self.assertEqual(second.data.get("requested_operations"), ["施肥"])

    def test_weather_lookup_uses_requested_operations_from_payload_when_prompt_is_relative(self) -> None:
        prompt = json.dumps(
            {
                "query": "下周呢",
                "region": "长沙",
                "start_date": "2026-03-16",
                "end_date": "2026-03-22",
                "requested_operations": ["施肥"],
            },
            ensure_ascii=False,
        )
        fake_result = ToolInvocation(
            name="weather_lookup",
            message="ok",
            data={
                "region": "长沙",
                "start_date": "2026-03-16",
                "end_date": "2026-03-22",
                "points": [
                    {
                        "timestamp": "2026-03-16T00:00:00",
                        "dy_ws": 0.9,
                        "sf_ws": 0.2,
                    }
                ],
            },
        )

        with patch("src.agent.tools.weather.lookup_weather", return_value=fake_result):
            result = weather_lookup(prompt)

        self.assertEqual(result.data.get("requested_operations"), ["施肥"])
        points = result.data.get("points") or []
        self.assertEqual(points[0].get("sf_ws"), 0.2)
        self.assertNotIn("dy_ws", points[0])


if __name__ == "__main__":
    unittest.main()
