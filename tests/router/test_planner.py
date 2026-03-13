import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
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
class PlannerRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
            "INTENT_ROUTING_MODE": os.environ.get("INTENT_ROUTING_MODE"),
        }
        os.environ["PENDING_STORE"] = "memory"
        os.environ["INTENT_ROUTING_MODE"] = "hybrid"
        from src.infra.config import get_config

        get_config.cache_clear()
        self._llm_patch = patch(
            "src.agent.planner.get_chat_model", return_value=_DummyLLM()
        )
        self._llm_patch.start()
        self._fast_intent_patch = patch(
            "src.agent.fast_intent.get_extractor_model", return_value=_DummyLLM()
        )
        self._fast_intent_patch.start()
        from src.agent.router import RequestRouter

        self.router = RequestRouter()

    def tearDown(self) -> None:
        self._llm_patch.stop()
        self._fast_intent_patch.stop()
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from src.infra.config import get_config

        get_config.cache_clear()

    def test_none_action_returns_response(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        plan = ActionPlan(action="none", response="ok")
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            result = self.router.handle(UserRequest(prompt="hello", session_id="s1"))

        self.assertEqual(result.mode, "none")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.message, "ok")

    def test_none_action_does_not_fallback_to_rule_tool(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        plan = ActionPlan(action="none")
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            with patch("src.agent.router.execute_tool") as mocked_execute:
                result = self.router.handle(
                    UserRequest(prompt="水稻品种美香占2号", session_id="s6")
                )

        self.assertEqual(result.mode, "none")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.message, "未识别到与农事相关的需求。")
        mocked_execute.assert_not_called()

    def test_tool_action_sets_pending(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={
                "region": "长沙",
                "start_date": "2025-03-13",
                "end_date": "2025-03-19",
                "year": 2025,
            },
        )
        tool_payload = ToolInvocation(
            name="weather_lookup",
            message="need followup",
            data={
                "missing_fields": ["region"],
                "draft": {"crop": "水稻"},
                "followup_count": 0,
            },
        )
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = self.router.handle(UserRequest(prompt="查天气", session_id="s2"))

        self.assertEqual(result.mode, "tool")
        self.assertTrue(mocked_execute.called)
        pending = self.router._pending_store.get("s2")
        self.assertIsNotNone(pending)
        self.assertEqual(pending.get("mode"), "tool")
        self.assertEqual(pending.get("tool_name"), "weather_lookup")

    def test_memory_clear_tool_clears_session_context(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        self.router._session_context_store.set(
            "s7",
            {
                "tool_contexts": {
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-13",
                        "end_date": "2026-03-19",
                    }
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        plan = ActionPlan(action="tool", name="memory_clear", input={})
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            with patch("src.agent.router.execute_tool") as mocked_execute:
                result = self.router.handle(
                    UserRequest(prompt="clear memory", session_id="s7", user_id="u7")
                )

        mocked_execute.assert_not_called()
        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "memory_clear")
        session_context = self.router._session_context_store.get("s7") or {}
        self.assertEqual(session_context, {})

    def test_workflow_action_invokes_runner(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest, WorkflowResponse

        plan = ActionPlan(
            action="workflow",
            name="crop_calendar_workflow",
            input={"prompt": "种水稻"},
        )
        plan_payload = WorkflowResponse(message="done")
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            with patch.object(
                self.router,
                "_run_named_workflow",
                return_value=plan_payload,
            ) as mocked_run:
                result = self.router.handle(UserRequest(prompt="种水稻", session_id="s3"))

        self.assertEqual(result.mode, "workflow")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.message, "done")
        mocked_run.assert_called_once()

    def test_none_action_clears_pending(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        self.router._pending_store.set(
            "s4",
            {
                "mode": "tool",
                "tool_name": "weather_lookup",
                "draft": {},
                "missing_fields": ["region"],
                "followup_count": 0,
            },
        )
        plan = ActionPlan(action="none", response="ok")
        with patch.object(self.router._intent_router, "plan", return_value=plan) as mocked_plan:
            result = self.router.handle(UserRequest(prompt="取消追问", session_id="s4"))

        mocked_plan.assert_called_once()
        self.assertEqual(result.mode, "none")
        self.assertIsNone(self.router._pending_store.get("s4"))
        self.assertEqual(result.plan.message, "ok")

    def test_rule_interrupts_pending_for_plan_list(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        self.router._pending_store.set(
            "s8",
            {
                "mode": "workflow",
                "workflow_name": "crop_calendar_workflow",
                "draft": {},
                "missing_fields": ["variety"],
                "followup_count": 1,
            },
        )
        tool_payload = ToolInvocation(
            name="plant_plan_list_active",
            message="ok",
            data={},
        )
        with patch(
            "src.agent.router.execute_tool", return_value=tool_payload
        ) as mocked_execute:
            result = self.router.handle(
                UserRequest(prompt="查询所有种植计划", session_id="s8")
            )

        self.assertEqual(result.mode, "tool")
        self.assertIsNotNone(result.tool)
        self.assertEqual(result.tool.name, "plant_plan_list_active")
        self.assertTrue(mocked_execute.called)
        self.assertIsNone(self.router._pending_store.get("s8"))

    def test_llm_only_mode_rewrites_sowing_query_away_from_weather(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        self.router._intent_mode = "llm_only"
        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={"start_date": "2026-03-13", "end_date": "2026-03-19"},
            reason="llm:weather",
        )
        tool_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="ok",
            data={},
        )
        with patch.object(self.router._planner, "plan", return_value=plan):
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = self.router.handle(
                    UserRequest(prompt="最近适合播种嘛", session_id="s-llm-sowing")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")

    def test_input_validation_reports_invalid_field_format_instead_of_missing(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={"start_date": {"bad": 1}, "end_date": "2026-03-19"},
        )
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            result = self.router.handle(
                UserRequest(prompt="查天气", session_id="s-invalid-weather")
            )

        self.assertEqual(result.mode, "none")
        self.assertIsNotNone(result.plan)
        self.assertIn("请检查这些字段的格式", result.plan.message)
        self.assertIn("起始日期(YYYY-MM-DD)", result.plan.message)


if __name__ == "__main__":
    unittest.main()
