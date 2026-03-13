import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_CHAINLIT = importlib.util.find_spec("chainlit") is None

if not _MISSING_CHAINLIT:
    from tests.scenario_loader import load_yaml_scenarios
    from chainlit_app import _format_weather_tool_details


@unittest.skipUnless(not _MISSING_CHAINLIT, "chainlit is not installed")
class ChainlitFormattingTests(unittest.TestCase):
    def test_weather_formatting_scenarios(self) -> None:
        for scenario in load_yaml_scenarios("weather/ui.yaml"):
            with self.subTest(scenario=scenario["id"]):
                detail = _format_weather_tool_details(
                    "weather_lookup",
                    scenario["data"],
                )
                for fragment in scenario.get("contains", []):
                    self.assertIn(fragment, detail)
                for fragment in scenario.get("not_contains", []):
                    self.assertNotIn(fragment, detail)


if __name__ == "__main__":
    unittest.main()
