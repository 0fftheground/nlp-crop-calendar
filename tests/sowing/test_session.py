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
    from tests.scenario_loader import load_yaml_scenarios


class _DummyLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return {"action": "none", "response": "noop"}


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class SowingSessionContextTests(unittest.TestCase):
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

    def test_sowing_session_context_scenarios(self) -> None:
        from src.agent.router import RequestRouter
        from src.schemas.models import ToolInvocation, UserRequest

        for scenario in load_yaml_scenarios("sowing/session.yaml")[
            "sowing_session_context"
        ]:
            with self.subTest(scenario=scenario["id"]):
                with patch("src.agent.planner.get_chat_model", return_value=_DummyLLM()):
                    with patch(
                        "src.agent.fast_intent.get_extractor_model",
                        return_value=_DummyLLM(),
                    ):
                        router = RequestRouter()

                router._session_context_store.set(
                    scenario["session_id"], scenario["initial_context"]
                )

                tool_result = scenario["tool_result"]
                tool_payload = ToolInvocation(
                    name=tool_result["name"],
                    message=tool_result["message"],
                    data=tool_result["data"],
                )
                with patch(
                    "src.agent.router.execute_tool", return_value=tool_payload
                ) as mocked_execute:
                    result = router.handle(
                        UserRequest(
                            prompt=scenario["followup_prompt"],
                            session_id=scenario["session_id"],
                        )
                    )

                self.assertEqual(result.mode, "tool")
                self.assertIsNotNone(result.tool)
                self.assertEqual(result.tool.name, scenario["expected"]["tool_name"])
                mocked_execute.assert_called_once()
                args = mocked_execute.call_args[0]
                self.assertEqual(args[0], scenario["expected"]["tool_name"])
                payload = json.loads(args[1])
                for key, value in scenario["expected"]["payload"].items():
                    self.assertEqual(payload.get(key), value)


if __name__ == "__main__":
    unittest.main()
