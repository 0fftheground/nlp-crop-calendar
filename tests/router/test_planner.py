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

    def test_pending_missing_field_reply_still_resumes(self) -> None:
        from src.schemas.models import HandleResponse, ToolInvocation, UserRequest

        self.router._pending_store.set(
            "s-pending-reply",
            {
                "mode": "tool",
                "tool_name": "weather_lookup",
                "draft": {"start_date": "2026-03-23", "end_date": "2026-03-29"},
                "missing_fields": ["region"],
                "followup_count": 0,
            },
        )
        pending_response = HandleResponse(
            mode="tool",
            tool=ToolInvocation(name="weather_lookup", message="ok", data={}),
        )
        with patch.object(
            self.router, "_resume_pending", return_value=pending_response
        ) as mocked_resume:
            with patch.object(self.router._intent_router, "plan") as mocked_plan:
                result = self.router.handle(
                    UserRequest(prompt="芜湖", session_id="s-pending-reply")
                )

        self.assertEqual(result.mode, "tool")
        mocked_resume.assert_called_once()
        mocked_plan.assert_not_called()

    def test_pending_new_question_falls_through_to_standalone_router(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        self.router._pending_store.set(
            "s-pending-new-question",
            {
                "mode": "tool",
                "tool_name": "weather_lookup",
                "draft": {"start_date": "2026-03-23", "end_date": "2026-03-29"},
                "missing_fields": ["region"],
                "followup_count": 0,
            },
        )
        plan = ActionPlan(action="none", response="ok")
        with patch.object(self.router, "_resume_pending") as mocked_resume:
            with patch.object(
                self.router._intent_router, "plan", return_value=plan
            ) as mocked_plan:
                result = self.router.handle(
                    UserRequest(
                        prompt="今天适合施肥吗", session_id="s-pending-new-question"
                    )
                )

        self.assertEqual(result.mode, "none")
        mocked_resume.assert_not_called()
        mocked_plan.assert_called_once()
        self.assertIsNone(self.router._pending_store.get("s-pending-new-question"))

    def test_pending_save_confirmation_new_question_does_not_resume(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        self.router._pending_store.set(
            "s-save-confirm-new-question",
            {
                "mode": "workflow",
                "workflow_name": "crop_calendar_workflow",
                "draft": {"region_id": "常德"},
                "missing_fields": ["save_confirmation"],
                "followup_count": 0,
                "pending_message": "是否保存该方案？请回复“是/否”。",
                "pending_kind": "confirmation",
            },
        )
        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={
                "region": "常德",
                "start_date": "2026-03-23",
                "end_date": "2026-03-29",
                "year": 2026,
            },
            response="ok",
        )
        tool_payload = ToolInvocation(name="weather_lookup", message="ok", data={})
        with patch.object(self.router, "_resume_pending") as mocked_resume:
            with patch.object(
                self.router._intent_router, "plan", return_value=plan
            ) as mocked_plan:
                with patch(
                    "src.agent.router.execute_tool", return_value=tool_payload
                ) as mocked_execute:
                    result = self.router.handle(
                        UserRequest(
                            prompt="下周适合打药嘛",
                            session_id="s-save-confirm-new-question",
                        )
                    )

        self.assertEqual(result.mode, "tool")
        mocked_resume.assert_not_called()
        mocked_plan.assert_called_once()
        mocked_execute.assert_called_once()
        self.assertIsNone(self.router._pending_store.get("s-save-confirm-new-question"))

    def test_pending_save_confirmation_yes_still_resumes(self) -> None:
        from src.schemas.models import HandleResponse, UserRequest, WorkflowResponse

        self.router._pending_store.set(
            "s-save-confirm-yes",
            {
                "mode": "workflow",
                "workflow_name": "crop_calendar_workflow",
                "draft": {"region_id": "常德"},
                "missing_fields": ["save_confirmation"],
                "followup_count": 0,
                "pending_message": "是否保存该方案？请回复“是/否”。",
                "pending_kind": "confirmation",
            },
        )
        pending_response = HandleResponse(
            mode="workflow",
            plan=WorkflowResponse(message="已保存。"),
        )
        with patch.object(
            self.router, "_resume_pending", return_value=pending_response
        ) as mocked_resume:
            with patch.object(self.router._intent_router, "plan") as mocked_plan:
                result = self.router.handle(
                    UserRequest(prompt="是", session_id="s-save-confirm-yes")
                )

        self.assertEqual(result.mode, "workflow")
        mocked_resume.assert_called_once()
        mocked_plan.assert_not_called()

    def test_pending_save_confirmation_phrase_still_resumes(self) -> None:
        from src.schemas.models import HandleResponse, UserRequest, WorkflowResponse

        self.router._pending_store.set(
            "s-save-confirm-phrase",
            {
                "mode": "workflow",
                "workflow_name": "crop_calendar_workflow",
                "draft": {"region_id": "常德"},
                "missing_fields": ["save_confirmation"],
                "followup_count": 0,
                "pending_message": "是否保存该方案？请回复“是/否”。",
                "pending_kind": "confirmation",
            },
        )
        pending_response = HandleResponse(
            mode="workflow",
            plan=WorkflowResponse(message="已保存。"),
        )
        with patch.object(
            self.router, "_resume_pending", return_value=pending_response
        ) as mocked_resume:
            with patch.object(self.router._intent_router, "plan") as mocked_plan:
                result = self.router.handle(
                    UserRequest(prompt="是的", session_id="s-save-confirm-phrase")
                )

        self.assertEqual(result.mode, "workflow")
        mocked_resume.assert_called_once()
        mocked_plan.assert_not_called()

    def test_ambiguous_thread_ownership_returns_clarification(self) -> None:
        from src.agent.planner import ActionPlan
        from src.agent.session_context import ContextualPlanCandidate
        from src.schemas.models import UserRequest

        candidate = ContextualPlanCandidate(
            plan=ActionPlan(
                action="tool",
                name="weather_lookup",
                input={
                    "region": "常德",
                    "start_date": "2026-03-23",
                    "end_date": "2026-03-29",
                    "year": 2026,
                },
            ),
            confidence=0.8,
            kind="tool",
            name="weather_lookup",
        )
        standalone_plan = ActionPlan(
            action="tool",
            name="sowing_suitability_lookup",
            input={"query": "这个呢"},
        )
        with patch.object(
            self.router, "_get_contextual_candidate", return_value=candidate
        ):
            with patch.object(
                self.router._intent_router, "plan", return_value=standalone_plan
            ):
                with patch("src.agent.router.execute_tool") as mocked_execute:
                    result = self.router.handle(
                        UserRequest(prompt="这个呢", session_id="s-clarify")
                    )

        self.assertEqual(result.mode, "none")
        self.assertIsNotNone(result.plan)
        self.assertIn("我不确定你是想继续当前的天气/农事适宜度查询", result.plan.message)
        self.assertIn("继续当前任务", result.plan.message)
        mocked_execute.assert_not_called()

    def test_clarification_pending_can_choose_new_task(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        self.router._pending_store.set(
            "s-clarification-resume",
            {
                "mode": "clarification",
                "pending_kind": "clarification",
                "options": ["继续当前任务", "开启新任务"],
                "pending_message": "请回复“继续当前任务”或“开启新任务”。",
                "contextual_plan": ActionPlan(
                    action="tool",
                    name="weather_lookup",
                    input={
                        "region": "常德",
                        "start_date": "2026-03-23",
                        "end_date": "2026-03-29",
                        "year": 2026,
                    },
                ).model_dump(mode="json"),
                "standalone_plan": ActionPlan(
                    action="tool",
                    name="sowing_suitability_lookup",
                    input={"query": "这个呢"},
                ).model_dump(mode="json"),
            },
        )
        tool_payload = ToolInvocation(
            name="sowing_suitability_lookup",
            message="ok",
            data={},
        )
        with patch("src.agent.router.execute_tool", return_value=tool_payload) as mocked_execute:
            result = self.router.handle(
                UserRequest(prompt="开启新任务", session_id="s-clarification-resume")
            )

        self.assertEqual(result.mode, "tool")
        self.assertEqual(result.tool.name, "sowing_suitability_lookup")
        self.assertEqual(mocked_execute.call_args[0][0], "sowing_suitability_lookup")
        self.assertIsNone(self.router._pending_store.get("s-clarification-resume"))

    def test_fallback_from_planner_updates_session_context_from_response(self) -> None:
        from src.schemas.models import HandleResponse, ToolInvocation, UserRequest

        response = HandleResponse(
            mode="tool",
            tool=ToolInvocation(
                name="sowing_suitability_lookup",
                message="success",
                data={
                    "resolved": {
                        "variety": "美香占2号",
                        "culti_type": "早稻",
                        "planting_method": "direct_seeding",
                        "region_id": "常德",
                        "crop": "水稻",
                    }
                },
            ),
        )
        with patch.object(self.router._intent_router, "plan", return_value=None):
            with patch.object(
                self.router._plan_executor,
                "fallback_from_planner",
                return_value=response,
            ):
                result = self.router.handle(
                    UserRequest(prompt="常德呢", session_id="s-fallback-context")
                )

        self.assertEqual(result.mode, "tool")
        session_context = self.router._session_context_store.get("s-fallback-context")
        self.assertEqual(
            session_context.get("last_context"),
            {"kind": "tool", "name": "sowing_suitability_lookup"},
        )

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

    def test_llm_only_mode_rewrites_travel_weather_query_to_none(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        self.router._intent_mode = "llm_only"
        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={"region": "湖南常德"},
            reason="llm:weather",
        )
        with patch.object(self.router._planner, "plan", return_value=plan):
            with patch("src.agent.router.execute_tool") as mocked_execute:
                result = self.router.handle(
                    UserRequest(prompt="湖南常德下周适合旅游吗", session_id="s-llm-travel")
                )

        self.assertEqual(result.mode, "none")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.message, "未识别到与农事相关的需求。")
        mocked_execute.assert_not_called()

    def test_weather_plan_execution_infers_requested_operations_from_prompt(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import ToolInvocation, UserRequest

        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={
                "region": "湖南常德",
                "start_date": "2026-03-30",
                "end_date": "2026-04-05",
                "year": 2026,
            },
        )
        tool_payload = ToolInvocation(name="weather_lookup", message="ok", data={})
        with patch.object(self.router._intent_router, "plan", return_value=plan):
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = self.router.handle(
                    UserRequest(prompt="湖南常德下周哪天最适合施肥", session_id="s-weather-op-infer")
                )

        self.assertEqual(result.mode, "tool")
        payload = mocked_execute.call_args[0][1]
        parsed = json.loads(payload)
        self.assertEqual(parsed.get("requested_operations"), ["施肥"])

    def test_crop_calendar_scheme_is_not_hijacked_by_weather_keywords(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest, WorkflowResponse

        self.router._intent_mode = "llm_only"
        plan = ActionPlan(
            action="tool",
            name="weather_lookup",
            input={"region": "湖南常德", "requested_operations": ["移栽"]},
            reason="llm:weather",
        )
        plan_payload = WorkflowResponse(message="done")
        with patch.object(self.router._planner, "plan", return_value=plan):
            with patch.object(
                self.router,
                "_run_named_workflow",
                return_value=plan_payload,
            ) as mocked_run:
                with patch("src.agent.router.execute_tool") as mocked_execute:
                    result = self.router.handle(
                        UserRequest(
                            prompt="我想建立一个在湖南常德种植的湘早籼24号的移栽方案",
                            session_id="s-crop-calendar-transplant-scheme",
                        )
                    )

        self.assertEqual(result.mode, "workflow")
        self.assertIsNotNone(result.plan)
        self.assertEqual(result.plan.message, "done")
        mocked_run.assert_called_once()
        mocked_execute.assert_not_called()

    def test_plan_list_context_can_resume_delete_tool(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        self.router._session_context_store.set(
            "s-plan-delete",
            {
                "tool_contexts": {
                    "plant_plan_list_active": {
                        "plans": [
                            {"plan_id": "11", "plan_name": "早稻计划"},
                            {"plan_id": "12", "plan_name": "晚稻计划"},
                        ]
                    }
                },
                "last_context": {"kind": "tool", "name": "plant_plan_list_active"},
            },
        )

        tool_payload = ToolInvocation(
            name="plant_plan_delete",
            message="已删除种植计划。",
            data={"plant_season_id": "12", "response": {"ok": True}},
        )
        with patch.object(self.router._intent_router, "plan", return_value=None) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = self.router.handle(
                    UserRequest(prompt="删除第2个计划", session_id="s-plan-delete")
                )

        self.assertEqual(result.mode, "tool")
        self.assertIsNotNone(result.tool)
        self.assertEqual(result.tool.name, "plant_plan_delete")
        mocked_plan.assert_not_called()
        self.assertEqual(mocked_execute.call_args[0][0], "plant_plan_delete")
        self.assertIn('"plant_season_id": "12"', mocked_execute.call_args[0][1])

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
