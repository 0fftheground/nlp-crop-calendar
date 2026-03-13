import importlib.util
import json
import os
import sys
import unittest
from contextlib import ExitStack
from datetime import date as _date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from tests.scenario_loader import load_yaml_scenarios
    from src.schemas.models import ToolInvocation


class _DummyLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return {"action": "none", "response": "noop"}


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class WeatherRegressionFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
            "INTENT_ROUTING_MODE": os.environ.get("INTENT_ROUTING_MODE"),
        }
        os.environ["PENDING_STORE"] = "memory"
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

    def _build_router(self):
        from src.agent.router import RequestRouter

        with patch("src.agent.planner.get_chat_model", return_value=_DummyLLM()):
            with patch(
                "src.agent.fast_intent.get_extractor_model", return_value=_DummyLLM()
            ):
                return RequestRouter()

    def _make_tool_invocation(self, payload: dict) -> ToolInvocation:
        return ToolInvocation(
            name="weather_lookup",
            message=str(payload.get("message") or ""),
            data=dict(payload.get("data") or {}),
        )

    def _build_fake_date(self, iso_date: str):
        year, month, day = (int(part) for part in iso_date.split("-"))

        class FakeDate(_date):
            @classmethod
            def today(cls):
                return cls(year, month, day)

        return FakeDate

    def test_weather_regression_scenarios(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        for scenario in load_yaml_scenarios("weather/session.yaml"):
            with self.subTest(scenario=scenario["id"]):
                router = self._build_router()
                session_id = f"scenario-{scenario['id']}"
                seed_turn = scenario["seed_turn"]
                seed_plan = ActionPlan(
                    action="tool",
                    name="weather_lookup",
                    input=dict(seed_turn["plan_input"]),
                )
                seed_payload = self._make_tool_invocation(seed_turn["tool_result"])
                with patch.object(router._intent_router, "plan", return_value=seed_plan):
                    with patch(
                        "src.agent.router.execute_tool", return_value=seed_payload
                    ):
                        router.handle(
                            UserRequest(prompt=seed_turn["prompt"], session_id=session_id)
                        )

                followup_turn = scenario["followup_turn"]
                followup_payload = self._make_tool_invocation(
                    followup_turn["tool_result"]
                )
                patches = []
                today_value = scenario.get("today")
                if today_value:
                    fake_date = self._build_fake_date(today_value)
                    patches.extend(
                        [
                            patch("src.application.services.weather_service.date", fake_date),
                            patch("src.agent.session_context.date", fake_date),
                        ]
                    )
                with patch.object(router._intent_router, "plan", return_value=None) as mocked_plan:
                    with patch(
                        "src.agent.router.execute_tool", return_value=followup_payload
                    ) as mocked_execute:
                        with ExitStack() as stack:
                            for item in patches:
                                stack.enter_context(item)
                            result = router.handle(
                                UserRequest(
                                    prompt=followup_turn["prompt"], session_id=session_id
                                )
                            )

                expected = scenario["expected"]
                self.assertEqual(result.mode, expected.get("mode", "tool"))
                mocked_plan.assert_not_called()
                if "requested_operations" in expected:
                    self.assertEqual(
                        result.tool.data.get("requested_operations"),
                        expected["requested_operations"],
                    )
                if "message_contains" in expected:
                    self.assertIn(expected["message_contains"], result.tool.message)
                payload = json.loads(mocked_execute.call_args[0][1])
                for key, value in expected["payload"].items():
                    self.assertEqual(payload.get(key), value)


if __name__ == "__main__":
    unittest.main()
