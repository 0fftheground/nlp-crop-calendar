import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MISSING_CHAINLIT = importlib.util.find_spec("chainlit") is None

if not _MISSING_CHAINLIT:
    from chainlit_app import _format_weather_tool_details


@unittest.skipUnless(not _MISSING_CHAINLIT, "chainlit is not installed")
class ChainlitFormattingTests(unittest.TestCase):
    def test_weather_details_include_operation_suitability(self) -> None:
        detail = _format_weather_tool_details(
            "weather_lookup",
            {
                "region": "farm:1",
                "start_date": "2026-03-03",
                "end_date": "2026-03-12",
                "points": [
                    {
                        "timestamp": "2026-03-12T00:00:00",
                        "temperature_max": 29.3,
                        "temperature_min": 19.73,
                        "precipitation": 0.0,
                        "dy_ws": 0.85,
                        "dy_reason": "风小无雨，适合打药。",
                    }
                ],
            },
        )
        self.assertIn("打药适宜度 0.85", detail)
        self.assertIn("风小无雨，适合打药。", detail)

    def test_weather_details_keep_operation_labels_consistent(self) -> None:
        detail = _format_weather_tool_details(
            "weather_lookup",
            {
                "region": "farm:1",
                "start_date": "2026-03-10",
                "end_date": "2026-03-10",
                "points": [
                    {
                        "timestamp": "2026-03-10T00:00:00",
                        "temperature_max": 24.0,
                        "temperature_min": 16.0,
                        "precipitation": 1.5,
                        "dy_ws": 0.8,
                        "dy_reason": "风力较小，适合打药。",
                        "sf_reason": "降水偏多，不建议施肥。",
                    }
                ],
            },
        )
        self.assertIn("打药适宜度 0.8", detail)
        self.assertIn("施肥适宜度（降水偏多，不建议施肥。）", detail)
        self.assertNotIn("施肥建议", detail)


if __name__ == "__main__":
    unittest.main()
