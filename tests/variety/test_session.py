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
    from tests.support import build_test_router


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class VarietySessionContextTests(unittest.TestCase):
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

    def _make_tool_invocation(self, payload: dict):
        from src.schemas.models import ToolInvocation

        return ToolInvocation(
            name=str(payload.get("name") or "variety_lookup"),
            message=str(payload.get("message") or ""),
            data=dict(payload.get("data") or {}),
        )

    def test_variety_session_context_scenarios(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        scenarios = load_yaml_scenarios("variety/service.yaml")["session_context"]
        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                router = build_test_router()
                seed_plan = ActionPlan(
                    action="tool",
                    name="variety_lookup",
                    input=dict(scenario["seed_turn"]["plan_input"]),
                )
                seed_payload = self._make_tool_invocation(
                    scenario["seed_turn"]["tool_result"]
                )
                with patch.object(router._intent_router, "plan", return_value=seed_plan):
                    with patch(
                        "src.agent.router.execute_tool", return_value=seed_payload
                    ):
                        router.handle(
                            UserRequest(
                                prompt=scenario["seed_turn"]["prompt"],
                                session_id=scenario["session_id"],
                            )
                        )

                followup_payload = self._make_tool_invocation(
                    scenario["followup_turn"]["tool_result"]
                )
                with patch.object(
                    router._intent_router, "plan", return_value=None
                ) as mocked_plan:
                    with patch(
                        "src.agent.router.execute_tool", return_value=followup_payload
                    ) as mocked_execute:
                        result = router.handle(
                            UserRequest(
                                prompt=scenario["followup_turn"]["prompt"],
                                session_id=scenario["session_id"],
                            )
                        )

                self.assertEqual(result.mode, "tool")
                self.assertEqual(result.tool.name, "variety_lookup")
                mocked_plan.assert_called_once()
                payload = json.loads(mocked_execute.call_args[0][1])
                merged_prompt = str(payload.get("prompt") or "")
                for snippet in scenario["expected"]["prompt_contains"]:
                    self.assertIn(snippet, merged_prompt)

    def test_supported_variety_attribute_followup_reuses_context(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "v-supported",
            {
                "tool_contexts": {
                    "variety_lookup": {
                        "crop": "水稻",
                        "variety": "美香占2号",
                        "region_choice": "湖南",
                        "selected": {"品种名称": "美香占2号"},
                    }
                },
                "last_context": {"kind": "tool", "name": "variety_lookup"},
            },
        )
        tool_payload = ToolInvocation(name="variety_lookup", message="ok", data={})
        with patch.object(router._intent_router, "plan", return_value=None) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="生育期天数呢", session_id="v-supported")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "variety_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertIn("美香占2号", str(payload.get("prompt") or ""))
        self.assertIn("生育期天数呢", str(payload.get("prompt") or ""))

    def test_unsupported_variety_attribute_followup_reuses_context(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "v-unsupported",
            {
                "tool_contexts": {
                    "variety_lookup": {
                        "crop": "水稻",
                        "variety": "美香占2号",
                        "region_choice": "湖南",
                        "selected": {"品种名称": "美香占2号"},
                    }
                },
                "last_context": {"kind": "tool", "name": "variety_lookup"},
            },
        )
        tool_payload = ToolInvocation(name="variety_lookup", message="暂无信息", data={})
        with patch.object(router._intent_router, "plan", return_value=None) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="抗病性怎么样", session_id="v-unsupported")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "variety_lookup")
        mocked_plan.assert_called_once()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertIn("美香占2号", str(payload.get("prompt") or ""))
        self.assertIn("抗病性怎么样", str(payload.get("prompt") or ""))


if __name__ == "__main__":
    unittest.main()
