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
    from src.agent.tools.plant_plan import plant_task_create
    from src.observability.interaction_context import (
        reset_interaction_context,
        set_interaction_context,
    )


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class PlanTaskCreateToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._interaction_token = set_interaction_context({"user_id": "当前登录用户"})

    def tearDown(self) -> None:
        reset_interaction_context(self._interaction_token)

    def test_custom_task_name_falls_back_to_other_task_type(self) -> None:
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract",
            return_value={
                "plan_id": "11",
                "name": "田间巡查",
                "date": "2026-03-30",
                "is_completed": True,
                "work_desc": "完成巡查。",
            },
        ):
            with patch(
                "src.agent.tools.plant_plan.resolve_code_name", return_value=None
            ):
                with patch(
                    "src.agent.tools.plant_plan.create_or_record_plan_task",
                    return_value={"status": "success", "target": "record"},
                ) as mocked_create:
                    result = plant_task_create.invoke("记录 plan_id=11 的田间巡查，已完成。")

        self.assertEqual(result.name, "plant_task_create")
        self.assertIn("已记录农事", result.message)
        payload = mocked_create.call_args[0][1]
        self.assertEqual(payload.get("name"), "田间巡查")
        self.assertEqual(payload.get("task_type"), "其他")

    def test_farmwork_dict_match_uses_same_name_and_task_type(self) -> None:
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract", return_value={}
        ):
            with patch("src.agent.tools.plant_plan.resolve_code_name", return_value="追肥"):
                with patch(
                    "src.agent.tools.plant_plan.create_or_record_plan_task",
                    return_value={"status": "success", "target": "record"},
                ) as mocked_create:
                    result = plant_task_create.invoke(
                        "记录 plan_id=11 的追肥，2026-03-26，已完成，说明：亩施10kg。"
                    )

        self.assertEqual(result.name, "plant_task_create")
        self.assertIn("已记录农事", result.message)
        mocked_create.assert_called_once()
        self.assertEqual(mocked_create.call_args[0][0], "11")
        payload = mocked_create.call_args[0][1]
        self.assertEqual(payload.get("name"), "追肥")
        self.assertEqual(payload.get("task_type"), "追肥")
        self.assertEqual(payload.get("is_completed"), True)
        self.assertEqual(payload.get("date"), "2026-03-26")
        self.assertEqual((payload.get("detail") or {}).get("work_desc"), "亩施10kg。")

    def test_future_task_date_derives_is_completed_false(self) -> None:
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract", return_value={}
        ):
            with patch("src.agent.tools.plant_plan.resolve_code_name", return_value="追肥"):
                with patch(
                    "src.agent.tools.plant_plan.create_or_record_plan_task",
                    return_value={"status": "success", "target": "extra"},
                ) as mocked_create:
                    result = plant_task_create.invoke(
                        "记录 plan_id=11 的追肥，2026-04-10。"
                    )

        self.assertEqual(result.name, "plant_task_create")
        self.assertIn("已新增农事", result.message)
        payload = mocked_create.call_args[0][1]
        self.assertEqual(payload.get("date"), "2026-04-10")
        self.assertEqual(payload.get("is_completed"), False)

    def test_completed_task_detail_defaults_operator_to_logged_in_user(self) -> None:
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract",
            return_value={
                "plan_id": "11",
                "name": "追肥",
                "date": "2026-03-30",
                "work_desc": "亩施10kg。",
            },
        ):
            with patch("src.agent.tools.plant_plan.resolve_code_name", return_value="追肥"):
                with patch(
                    "src.agent.tools.plant_plan.create_or_record_plan_task",
                    return_value={"status": "success", "target": "record"},
                ) as mocked_create:
                    result = plant_task_create.invoke("记录 plan_id=11 的追肥，说明：亩施10kg。")

        self.assertEqual(result.name, "plant_task_create")
        payload = mocked_create.call_args[0][1]
        self.assertEqual(
            payload.get("detail"),
            {"operator": "当前登录用户", "work_desc": "亩施10kg。"},
        )

    def test_completed_task_can_be_recorded_without_detail(self) -> None:
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract", return_value={}
        ):
            with patch("src.agent.tools.plant_plan.resolve_code_name", return_value="追肥"):
                with patch(
                    "src.agent.tools.plant_plan.create_or_record_plan_task",
                    return_value={"status": "success", "target": "record"},
                ) as mocked_create:
                    result = plant_task_create.invoke(
                        "记录 plan_id=11 的追肥，2026-03-26，已完成。"
                    )

        self.assertEqual(result.name, "plant_task_create")
        self.assertIn("已记录农事", result.message)
        payload = mocked_create.call_args[0][1]
        self.assertEqual(payload.get("name"), "追肥")
        self.assertEqual(payload.get("is_completed"), True)
        self.assertEqual(
            payload.get("detail"),
            {"operator": "当前登录用户", "work_desc": None},
        )

    def test_vague_plan_task_prompt_skips_llm_extract(self) -> None:
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract"
        ) as mocked_extract:
            result = plant_task_create.invoke("我要录农事")

        self.assertEqual(result.name, "plant_task_create")
        self.assertIn("plant_season_id", result.message)
        mocked_extract.assert_not_called()

    def test_missing_name_followup_uses_raw_reply_as_task_name(self) -> None:
        followup_prompt = (
            '{"query":"给id=189的种植计划新增一个任务，时间是2026-04-15",'
            '"followup":{"prompt":"施分蘖肥","draft":{"plan_id":"189","date":"2026-04-15"},'
            '"missing_fields":["name"],"followup_count":1,"pending_kind":"field_fill"}}'
        )
        with patch(
            "src.agent.tools.plant_plan.llm_structured_extract"
        ) as mocked_extract:
            with patch(
                "src.agent.tools.plant_plan.resolve_code_name", return_value=None
            ):
                with patch(
                    "src.agent.tools.plant_plan.create_or_record_plan_task",
                    return_value={"status": "success", "target": "extra"},
                ) as mocked_create:
                    result = plant_task_create.invoke(followup_prompt)

        self.assertEqual(result.name, "plant_task_create")
        self.assertIn("已新增农事", result.message)
        mocked_extract.assert_not_called()
        payload = mocked_create.call_args[0][1]
        self.assertEqual(payload.get("name"), "施分蘖肥")
        self.assertEqual(payload.get("task_type"), "其他")


if __name__ == "__main__":
    unittest.main()
