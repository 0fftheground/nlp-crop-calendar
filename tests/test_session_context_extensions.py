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
class SessionContextExtensionTests(unittest.TestCase):
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

    def test_weather_lookup_reuses_latest_session_context(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        router = self._build_router()
        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={
                "region": "长沙",
                "start_date": "2026-03-25",
                "end_date": "2026-03-31",
            },
        )
        initial_payload = ToolInvocation(
            name="weather_lookup",
            message="ok",
            data={
                "region": "长沙",
                "start_date": "2026-03-25",
                "end_date": "2026-03-31",
                "granularity": "daily",
                "points": [],
            },
        )
        with patch.object(router._intent_router, "plan", return_value=plan):
            with patch(
                "src.agent.router.execute_tool", return_value=initial_payload
            ):
                router.handle(
                    UserRequest(prompt="查长沙 2026-03-25 到 2026-03-31 天气", session_id="w1")
                )

        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=initial_payload
            ) as mocked_execute:
                result = router.handle(UserRequest(prompt="在芜湖呢", session_id="w1"))

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_not_called()
        payload = json.loads(mocked_execute.call_args[0][1])
        self.assertEqual(payload.get("region"), "芜湖")
        self.assertEqual(payload.get("start_date"), "2026-03-25")
        self.assertEqual(payload.get("end_date"), "2026-03-31")

    def test_variety_lookup_reuses_latest_session_context(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        router = self._build_router()
        plan = ActionPlan(
            action="tool",
            name="variety_lookup",
            input={"query": "查询美香占2号审定信息"},
        )
        initial_payload = ToolInvocation(
            name="variety_lookup",
            message="ok",
            data={
                "crop": "水稻",
                "variety": "美香占2号",
                "region_choice": "湖南",
                "selected": {"品种名称": "美香占2号"},
            },
        )
        with patch.object(router._intent_router, "plan", return_value=plan):
            with patch(
                "src.agent.router.execute_tool", return_value=initial_payload
            ):
                router.handle(
                    UserRequest(prompt="查询美香占2号审定信息", session_id="v1")
                )

        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=initial_payload
            ) as mocked_execute:
                result = router.handle(UserRequest(prompt="在芜湖呢", session_id="v1"))

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "variety_lookup")
        mocked_plan.assert_not_called()
        payload = json.loads(mocked_execute.call_args[0][1])
        merged_prompt = str(payload.get("prompt") or "")
        self.assertIn("美香占2号", merged_prompt)
        self.assertIn("芜湖", merged_prompt)

    def test_growth_stage_workflow_reuses_latest_session_context(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest, WorkflowResponse

        router = self._build_router()
        plan = ActionPlan(
            action="workflow",
            name="growth_stage_query_workflow",
            input={"prompt": "查询美香占2号的生育期"},
        )
        initial_payload = WorkflowResponse(
            message="ok",
            data={
                "workflow": {
                    "plan_id": "12",
                    "plan_filters": {
                        "variety": "美香占2号",
                        "region_id": "长沙",
                        "planting_method": "direct_seeding",
                        "culti_type": "早稻",
                        "sowing_date": "2026-03-20",
                    },
                },
                "planting": {
                    "variety": "美香占2号",
                    "region_id": "长沙",
                    "planting_method": "direct_seeding",
                    "culti_type": "早稻",
                    "sowing_date": "2026-03-20",
                },
            },
        )
        with patch.object(router._intent_router, "plan", return_value=plan):
            with patch.object(
                router, "_run_named_workflow", return_value=initial_payload
            ):
                router.handle(
                    UserRequest(prompt="查询美香占2号的生育期", session_id="g1")
                )

        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch.object(
                router, "_run_named_workflow", return_value=initial_payload
            ) as mocked_run:
                result = router.handle(UserRequest(prompt="在芜湖呢", session_id="g1"))

        self.assertEqual(result.mode, "workflow")
        mocked_plan.assert_not_called()
        merged_prompt = mocked_run.call_args[0][0]
        self.assertIn("芜湖", merged_prompt)
        self.assertIn("美香占2号", merged_prompt)
        self.assertIn("生育期", merged_prompt)
        self.assertEqual(mocked_run.call_args[0][1], "growth_stage_query_workflow")

    def test_crop_calendar_workflow_reuses_latest_session_context(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest, WorkflowResponse

        router = self._build_router()
        plan = ActionPlan(
            action="workflow",
            name="crop_calendar_workflow",
            input={"prompt": "生成农事方案"},
        )
        initial_payload = WorkflowResponse(
            message="ok",
            data={
                "planting": {
                    "region_id": "长沙",
                    "crop": "水稻",
                    "variety": "美香占2号",
                    "culti_type": "早稻",
                    "planting_method": "transplanting",
                    "sowing_date": "2026-03-20",
                    "transplant_date": "2026-04-05",
                },
                "plant_season_id": 1,
                "resolved_region_id": "430100",
            },
        )
        with patch.object(router._intent_router, "plan", return_value=plan):
            with patch.object(
                router, "_run_named_workflow", return_value=initial_payload
            ):
                router.handle(UserRequest(prompt="生成农事方案", session_id="c1"))

        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch.object(
                router, "_run_named_workflow", return_value=initial_payload
            ) as mocked_run:
                result = router.handle(UserRequest(prompt="改成直播呢", session_id="c1"))

        self.assertEqual(result.mode, "workflow")
        mocked_plan.assert_not_called()
        merged_prompt = mocked_run.call_args[0][0]
        self.assertIn("直播", merged_prompt)
        self.assertIn("美香占2号", merged_prompt)
        self.assertIn("2026-03-20", merged_prompt)
        self.assertEqual(mocked_run.call_args[0][1], "crop_calendar_workflow")

    def test_crop_calendar_save_response_does_not_override_session_context(self) -> None:
        from src.schemas.models import WorkflowResponse

        router = self._build_router()
        router._session_context_store.set(
            "c-save",
            {
                "workflow_contexts": {
                    "crop_calendar_workflow": {
                        "planting": {
                            "region_id": "长沙",
                            "crop": "水稻",
                            "variety": "美香占2号",
                            "culti_type": "早稻",
                            "planting_method": "direct_seeding",
                            "sowing_date": "2026-03-20",
                        },
                        "plant_season_id": 1,
                    }
                },
                "last_context": {"kind": "workflow", "name": "crop_calendar_workflow"},
            },
        )

        session_context = router._session_context_store.get("c-save")
        save_plan = WorkflowResponse(
            message="已保存种植计划。",
            data={"save_response": {"code": "0", "data": {"plant_season_id": 1}}},
        )

        from src.agent.session_context import extract_session_context_from_workflow

        extracted = extract_session_context_from_workflow(
            "crop_calendar_workflow", save_plan
        )
        self.assertIsNone(extracted)
        self.assertEqual(
            session_context.get("last_context"),
            {"kind": "workflow", "name": "crop_calendar_workflow"},
        )

    def test_session_context_only_uses_last_context(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = self._build_router()
        router._session_context_store.set(
            "s-last",
            {
                "tool_contexts": {
                    "sowing_suitability_lookup": {
                        "variety": "美香占2号",
                        "culti_type": "早稻",
                        "planting_method": "direct_seeding",
                        "region_id": "长沙",
                        "crop": "水稻",
                    },
                    "weather_lookup": {
                        "region": "长沙",
                        "start_date": "2026-03-25",
                        "end_date": "2026-03-31",
                        "granularity": "daily",
                    },
                },
                "last_context": {"kind": "tool", "name": "weather_lookup"},
            },
        )

        weather_payload = ToolInvocation(
            name="weather_lookup",
            message="ok",
            data={
                "region": "芜湖",
                "start_date": "2026-03-25",
                "end_date": "2026-03-31",
                "granularity": "daily",
                "points": [],
            },
        )
        with patch.object(
            router._intent_router, "plan", return_value=None
        ) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=weather_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="在芜湖呢", session_id="s-last")
                )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "weather_lookup")
        mocked_plan.assert_not_called()
        self.assertEqual(mocked_execute.call_args[0][0], "weather_lookup")


if __name__ == "__main__":
    unittest.main()
