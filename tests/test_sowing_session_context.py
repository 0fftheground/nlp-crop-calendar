import importlib.util
import json
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

    def test_followup_region_reuses_latest_sowing_context(self) -> None:
        from src.agent.router import RequestRouter
        from src.schemas.models import ToolInvocation, UserRequest

        with patch("src.agent.planner.get_chat_model", return_value=_DummyLLM()):
            with patch(
                "src.agent.fast_intent.get_extractor_model", return_value=_DummyLLM()
            ):
                router = RequestRouter()

        router._session_context_store.set(
            "s1",
            {
                "tool_contexts": {
                    "sowing_suitability_lookup": {
                        "variety": "美香占2号",
                        "culti_type": "早稻",
                        "planting_method": "direct_seeding",
                        "region_id": "湖南",
                        "crop": "水稻",
                    }
                },
                "last_context": {
                    "kind": "tool",
                    "name": "sowing_suitability_lookup",
                },
            },
        )

        tool_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="success",
            data={"resolved": {"region_id": "芜湖"}},
        )
        with patch(
            "src.agent.router.execute_tool", return_value=tool_payload
        ) as mocked_execute:
            result = router.handle(UserRequest(prompt="在芜湖呢", session_id="s1"))

        self.assertEqual(result.mode, "tool")
        self.assertIsNotNone(result.tool)
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        mocked_execute.assert_called_once()
        args = mocked_execute.call_args[0]
        self.assertEqual(args[0], "sowing_suitability_lookup")
        payload = json.loads(args[1])
        self.assertEqual(payload.get("variety"), "美香占2号")
        self.assertEqual(payload.get("culti_type"), "早稻")
        self.assertEqual(payload.get("planting_method"), "direct_seeding")
        self.assertEqual(payload.get("region_id"), "芜湖")


if __name__ == "__main__":
    unittest.main()
