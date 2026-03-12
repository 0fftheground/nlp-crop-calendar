import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None


class _DummyLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return {"action": "none", "response": "noop"}


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class IntentRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
            "INTENT_RULES_PATH": os.environ.get("INTENT_RULES_PATH"),
            "INTENT_ROUTING_MODE": os.environ.get("INTENT_ROUTING_MODE"),
        }
        os.environ["PENDING_STORE"] = "memory"
        os.environ["INTENT_RULES_PATH"] = str(
            ROOT / "resources" / "intent_rules.json"
        )
        os.environ["INTENT_ROUTING_MODE"] = "hybrid"
        from src.infra.config import get_config

        get_config.cache_clear()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from src.infra.config import get_config

        get_config.cache_clear()

    def test_intent_rule_engine_match(self) -> None:
        from src.agent.intent_rules import IntentRuleEngine

        engine = IntentRuleEngine(Path(os.environ["INTENT_RULES_PATH"]))
        rule = engine.match("请帮我清除历史记录")
        self.assertIsNotNone(rule)
        self.assertEqual(rule.action, "tool")
        self.assertEqual(rule.name, "memory_clear")

    def test_router_rule_weather_payload(self) -> None:
        from src.agent.router import RequestRouter

        with patch("src.agent.planner.get_chat_model", return_value=_DummyLLM()):
            with patch(
                "src.agent.fast_intent.get_extractor_model", return_value=_DummyLLM()
            ):
                router = RequestRouter()
        plan = router._rule_route("广州天气 2024年")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, "tool")
        self.assertEqual(plan.name, "weather_lookup")
        self.assertIsInstance(plan.input, dict)
        self.assertEqual(plan.input.get("region"), "广州")
        self.assertEqual(plan.input.get("year"), 2024)

    def test_router_rule_sowing_suitability_payload(self) -> None:
        from src.agent.router import RequestRouter

        with patch("src.agent.planner.get_chat_model", return_value=_DummyLLM()):
            with patch(
                "src.agent.fast_intent.get_extractor_model", return_value=_DummyLLM()
            ):
                router = RequestRouter()
        plan = router._intent_router._rule_route("帮我查美香占2号一季晚稻直播的播种适宜期")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, "tool")
        self.assertEqual(plan.name, "sowing_suitability_lookup")
        self.assertIsInstance(plan.input, dict)
        self.assertEqual(plan.input.get("query"), "帮我查美香占2号一季晚稻直播的播种适宜期")

    def test_router_rule_crop_calendar_generate_plan(self) -> None:
        from src.agent.router import RequestRouter

        with patch("src.agent.planner.get_chat_model", return_value=_DummyLLM()):
            with patch(
                "src.agent.fast_intent.get_extractor_model", return_value=_DummyLLM()
            ):
                router = RequestRouter()
        plan = router._intent_router._rule_route("帮我生成一个种植计划")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.action, "workflow")
        self.assertEqual(plan.name, "crop_calendar_workflow")
        self.assertEqual(plan.input, {"prompt": "帮我生成一个种植计划"})


if __name__ == "__main__":
    unittest.main()
