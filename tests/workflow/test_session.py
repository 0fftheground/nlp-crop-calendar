import importlib.util
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
class WorkflowSessionContextTests(unittest.TestCase):
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

    def _make_workflow_response(self, payload: dict):
        from src.schemas.models import WorkflowResponse

        return WorkflowResponse(
            message=str(payload.get("message") or ""),
            data=dict(payload.get("data") or {}),
        )

    def test_workflow_session_context_scenarios(self) -> None:
        from src.agent.planner import ActionPlan
        from src.schemas.models import UserRequest

        scenario_sets = load_yaml_scenarios("workflow/session.yaml")
        for group_name in ("growth_stage", "crop_calendar"):
            for scenario in scenario_sets[group_name]:
                with self.subTest(group=group_name, scenario=scenario["id"]):
                    router = build_test_router()
                    seed_plan = ActionPlan(
                        action="workflow",
                        name=scenario["seed_turn"]["workflow_name"],
                        input=dict(scenario["seed_turn"]["plan_input"]),
                    )
                    seed_payload = self._make_workflow_response(
                        scenario["seed_turn"]["workflow_result"]
                    )
                    with patch.object(router._intent_router, "plan", return_value=seed_plan):
                        with patch.object(
                            router, "_run_named_workflow", return_value=seed_payload
                        ):
                            router.handle(
                                UserRequest(
                                    prompt=scenario["seed_turn"]["prompt"],
                                    session_id=scenario["session_id"],
                                )
                            )

                    followup_payload = self._make_workflow_response(
                        scenario["followup_turn"]["workflow_result"]
                    )
                    with patch.object(
                        router._intent_router, "plan", return_value=None
                    ) as mocked_plan:
                        with patch.object(
                            router,
                            "_run_named_workflow",
                            return_value=followup_payload,
                        ) as mocked_run:
                            result = router.handle(
                                UserRequest(
                                    prompt=scenario["followup_turn"]["prompt"],
                                    session_id=scenario["session_id"],
                                )
                            )

                    self.assertEqual(result.mode, "workflow")
                    mocked_plan.assert_not_called()
                    self.assertEqual(
                        mocked_run.call_args[0][1],
                        scenario["expected"]["workflow_name"],
                    )
                    merged_prompt = mocked_run.call_args[0][0]
                    for snippet in scenario["expected"]["prompt_contains"]:
                        self.assertIn(snippet, merged_prompt)

    def test_crop_calendar_save_response_does_not_override_session_context(self) -> None:
        from src.agent.session_context import extract_session_context_from_workflow
        from src.schemas.models import WorkflowResponse

        router = build_test_router()
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
        extracted = extract_session_context_from_workflow(
            "crop_calendar_workflow", save_plan
        )
        self.assertIsNone(extracted)
        self.assertEqual(
            session_context.get("last_context"),
            {"kind": "workflow", "name": "crop_calendar_workflow"},
        )

    def test_plan_list_context_can_resume_growth_stage_workflow(self) -> None:
        from src.schemas.models import UserRequest, WorkflowResponse

        router = build_test_router()
        router._session_context_store.set(
            "w-plan-growth",
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

        workflow_payload = WorkflowResponse(message="ok", data={"workflow": {"plan_id": "11"}})
        with patch.object(router._intent_router, "plan", return_value=None) as mocked_plan:
            with patch.object(
                router, "_run_named_workflow", return_value=workflow_payload
            ) as mocked_run:
                result = router.handle(
                    UserRequest(prompt="查询第1个计划的生育期", session_id="w-plan-growth")
                )

        self.assertEqual(result.mode, "workflow")
        self.assertIsNotNone(result.plan)
        mocked_plan.assert_not_called()
        self.assertEqual(mocked_run.call_args[0][1], "growth_stage_query_workflow")
        self.assertIn("id=11", mocked_run.call_args[0][0])

    def test_crop_calendar_context_can_resume_delete_tool(self) -> None:
        from src.schemas.models import ToolInvocation, UserRequest

        router = build_test_router()
        router._session_context_store.set(
            "w-crop-delete",
            {
                "workflow_contexts": {
                    "crop_calendar_workflow": {
                        "planting": {
                            "region_id": "长沙",
                            "crop": "水稻",
                            "variety": "美香占2号",
                            "culti_type": "早稻",
                            "planting_method": "direct_seeding",
                        },
                        "plant_season_id": 21,
                    }
                },
                "last_context": {"kind": "workflow", "name": "crop_calendar_workflow"},
            },
        )

        tool_payload = ToolInvocation(
            name="plant_plan_delete",
            message="已删除种植计划。",
            data={"plant_season_id": "21", "response": {"ok": True}},
        )
        with patch.object(router._intent_router, "plan", return_value=None) as mocked_plan:
            with patch(
                "src.agent.router.execute_tool", return_value=tool_payload
            ) as mocked_execute:
                result = router.handle(
                    UserRequest(prompt="删除这个计划", session_id="w-crop-delete")
                )

        self.assertEqual(result.mode, "tool")
        self.assertIsNotNone(result.tool)
        self.assertEqual(result.tool.name, "plant_plan_delete")
        mocked_plan.assert_not_called()
        self.assertEqual(mocked_execute.call_args[0][0], "plant_plan_delete")
        self.assertIn('"plant_season_id": "21"', mocked_execute.call_args[0][1])


if __name__ == "__main__":
    unittest.main()
