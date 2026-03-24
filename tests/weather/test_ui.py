import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_CHAINLIT = importlib.util.find_spec("chainlit") is None

if _MISSING_CHAINLIT:
    fake_chainlit = types.ModuleType("chainlit")

    class _FakeUserSession:
        def __init__(self) -> None:
            self._values: dict[str, object] = {}

        def get(self, key: str, default: object = None) -> object:
            return self._values.get(key, default)

        def set(self, key: str, value: object) -> None:
            self._values[key] = value

    class _FakeMessage:
        def __init__(self, content: str = "", author: str | None = None) -> None:
            self.content = content
            self.author = author

        async def send(self):
            return self

        async def remove(self) -> None:
            return None

        async def update(self, content: str = "") -> None:
            self.content = content
            return None

    class _FakeUser:
        def __init__(self, identifier: str) -> None:
            self.identifier = identifier

    def _identity_decorator(func):
        return func

    fake_chainlit.on_chat_start = _identity_decorator
    fake_chainlit.on_chat_resume = _identity_decorator
    fake_chainlit.on_message = _identity_decorator
    fake_chainlit.password_auth_callback = _identity_decorator
    fake_chainlit.Message = _FakeMessage
    fake_chainlit.User = _FakeUser
    fake_chainlit.user_session = _FakeUserSession()
    fake_chainlit.context = None
    sys.modules["chainlit"] = fake_chainlit

from tests.scenario_loader import load_yaml_scenarios
from chainlit_app import (
    _build_capability_guide,
    _format_recent_farm_work_summary,
    _format_weather_tool_details,
)

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

    def test_recent_farm_work_summary_lists_plan_items(self) -> None:
        summary = _format_recent_farm_work_summary(
            {
                "code": 200,
                "data": {
                    "farm_id": 1,
                    "plans": [
                        {
                            "plan_name": "早稻一号田",
                            "farm_works": [
                                {"name": "施肥", "date": "2026-03-18"},
                                {"name": "灌溉", "date": "2026-03-20"},
                            ],
                        },
                        {
                            "name": "晚稻二号田",
                            "tasks": {"病虫害巡查": "2026-03-21"},
                        },
                    ],
                },
            },
            farm_id=1,
        )
        self.assertIn("默认农场（farm_id=1）未来 7 天农事", summary)
        self.assertIn("早稻一号田", summary)
        self.assertIn("2026-03-18 施肥", summary)
        self.assertIn("晚稻二号田", summary)
        self.assertIn("2026-03-21 病虫害巡查", summary)

    def test_capability_guide_is_standalone_message(self) -> None:
        guide = _build_capability_guide()
        self.assertIn("欢迎使用农事助手。", guide)
        self.assertIn("支持的 Tool：", guide)
        self.assertIn("weather_lookup", guide)
        self.assertIn("plant_plan_delete", guide)
        self.assertIn("growth_stage_lookup", guide)
        self.assertIn("支持的 Workflow：", guide)
        self.assertIn("crop_calendar_workflow", guide)
        self.assertNotIn("growth_stage_query_workflow", guide)
        self.assertIn("你可以直接这样问：", guide)

    def test_recent_farm_work_summary_handles_request_failure(self) -> None:
        summary = _format_recent_farm_work_summary(None, farm_id=1)
        self.assertIn("farm_id=1", summary)
        self.assertIn("暂时无法加载", summary)


if __name__ == "__main__":
    unittest.main()
