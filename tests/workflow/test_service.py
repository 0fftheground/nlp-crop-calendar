import importlib.util
import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from tests.scenario_loader import load_yaml_scenarios
    from src.agent.workflows.crop_calendar_graph import _ask_node
    from src.agent.workflows.crop_calendar_graph import _extract_node
    from src.agent.tools.plant_plan import growth_stage_lookup
    from src.infra.config import get_config
    from src.infra.tool_cache import get_tool_result_cache
    from src.schemas import GrowthStageResult, PlantingDetailsDraft


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class WorkflowServiceScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = {
            "TOOL_CACHE_STORE": os.environ.get("TOOL_CACHE_STORE"),
            "PENDING_STORE": os.environ.get("PENDING_STORE"),
        }
        os.environ["TOOL_CACHE_STORE"] = "memory"
        os.environ["PENDING_STORE"] = "memory"
        get_config.cache_clear()
        get_tool_result_cache.cache_clear()

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        get_config.cache_clear()
        get_tool_result_cache.cache_clear()

    def _draft_from_yaml(self, payload):
        if payload is None:
            return None
        data = dict(payload)
        for field in ("sowing_date", "transplant_date"):
            if data.get(field):
                data[field] = date.fromisoformat(str(data[field]))
        return PlantingDetailsDraft(**data)

    def test_growth_stage_service_scenarios(self) -> None:
        scenarios = load_yaml_scenarios("workflow/service.yaml")

        for scenario in scenarios["growth_stage_feature_cases"]:
            with self.subTest(group="growth_stage", scenario=scenario["id"]):
                draft = self._draft_from_yaml(scenario["draft"])
                planting = draft.to_canonical()
                growth_payload = GrowthStageResult(
                    stages={
                        "stage_dates": json.dumps(
                            scenario["growth_result"]["stage_dates"],
                            ensure_ascii=False,
                        )
                    }
                )
                with patch(
                    "src.agent.tools.plant_plan.resolve_planting_from_plan_id",
                    return_value=planting,
                ), patch(
                    "src.agent.tools.plant_plan.query_growth_stage_from_plan_id",
                    return_value=growth_payload,
                ):
                    result = growth_stage_lookup.invoke(
                        json.dumps(
                            {
                                "query": scenario["user_prompt"],
                                "plan_id": str(
                                    scenario["search_result"]["records"][0]["id"]
                                ),
                            },
                            ensure_ascii=False,
                        )
                    )

                message = result.message
                for snippet in scenario["expected"]["message_contains"]:
                    self.assertIn(snippet, message)
                for snippet in scenario["expected"]["message_not_contains"]:
                    self.assertNotIn(snippet, message)

    def test_crop_calendar_service_scenarios(self) -> None:
        scenarios = load_yaml_scenarios("workflow/service.yaml")

        for scenario in scenarios["crop_calendar_graph_cases"]:
            with self.subTest(group="crop_calendar", scenario=scenario["id"]):
                with patch(
                    "src.agent.workflows.crop_calendar_graph.find_exact_variety_in_text",
                    return_value=scenario.get("exact_variety_in_text"),
                ), patch(
                    "src.agent.workflows.crop_calendar_graph.retrieve_variety_candidates",
                    return_value=scenario.get("retrieve_variety_candidates", []),
                ), patch(
                    "src.agent.workflows.crop_calendar_graph.load_variety_names",
                    return_value=["美香占2号"],
                ), patch(
                    "src.agent.workflows.crop_calendar_graph.resolve_sowing_method_code",
                    side_effect=lambda value: (
                        "ZB"
                        if str(getattr(value, "value", value))
                        in {"direct_seeding", "直播"}
                        else "YC"
                        if str(getattr(value, "value", value))
                        in {"transplanting", "移栽", "插秧"}
                        else None
                    ),
                ), patch(
                    "src.agent.workflows.crop_calendar_graph.resolve_culti_type_code",
                    side_effect=lambda value: (
                        "ZD"
                        if str(value) in {"中稻", "早稻", "晚稻", "双季晚稻"}
                        else None
                    ),
                ), patch(
                    "src.agent.workflows.crop_calendar_graph.list_code_names",
                    return_value=["直播", "插秧", "早稻", "中稻"],
                ):
                    if scenario["mode"] == "graph":
                        draft = self._draft_from_yaml(scenario["draft"])
                        with patch(
                            "src.agent.workflows.crop_calendar_graph.extract_planting_details",
                            return_value=draft,
                        ):
                            state = _ask_node(
                                _extract_node(
                                    {
                                        "user_prompt": scenario["user_prompt"],
                                        "draft": None,
                                        "missing_fields": [],
                                        "followup_count": 0,
                                        "pending_options": [],
                                        "trace": [],
                                    }
                                )
                            )
                        message = state.get("message", "")
                        for field in scenario["expected"].get(
                            "missing_fields_contains", []
                        ):
                            self.assertIn(field, state.get("missing_fields", []))
                        for snippet in scenario["expected"].get("message_contains", []):
                            self.assertIn(snippet, message)
                        for snippet in scenario["expected"].get(
                            "message_not_contains", []
                        ):
                            self.assertNotIn(snippet, message)
                        continue

                    prior = self._draft_from_yaml(scenario["prior_draft"])
                    followup = self._draft_from_yaml(scenario["followup_draft"])
                    with patch(
                        "src.agent.workflows.crop_calendar_graph.extract_planting_details",
                        return_value=followup,
                    ):
                        state = _extract_node(
                            {
                                "user_prompt": scenario["user_prompt"],
                                "draft": prior.model_dump(mode="json"),
                                "missing_fields": list(scenario["missing_fields"]),
                                "followup_count": scenario["followup_count"],
                                "trace": [],
                                "pending_options": [],
                            }
                        )

                    draft = PlantingDetailsDraft.model_validate(
                        state.get("draft") or state.get("planting_draft")
                    )
                    for field, value in scenario["expected"].get("draft", {}).items():
                        expected_value = value
                        if field in ("sowing_date", "transplant_date") and value:
                            expected_value = date.fromisoformat(str(value))
                        self.assertEqual(getattr(draft, field), expected_value)
                    for field in scenario["expected"].get(
                        "missing_fields_not_contains", []
                    ):
                        self.assertNotIn(field, state.get("missing_fields", []))
                    for prefix in scenario["expected"].get("assumptions_prefix", []):
                        self.assertTrue(
                            any(
                                str(item).startswith(prefix)
                                for item in (draft.assumptions or [])
                            )
                        )


if __name__ == "__main__":
    unittest.main()
