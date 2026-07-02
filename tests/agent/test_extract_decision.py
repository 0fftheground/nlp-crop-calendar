import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from src.agent.extract_decision import should_extract_for_route


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class ExtractDecisionTests(unittest.TestCase):
    def test_plan_task_vague_prompt_does_not_extract(self) -> None:
        decision = should_extract_for_route(
            action="tool",
            name="plant_task_create",
            prompt="我要录农事",
        )
        self.assertFalse(decision.should_extract)

    def test_plan_task_structured_prompt_extracts(self) -> None:
        decision = should_extract_for_route(
            action="tool",
            name="plant_task_create",
            prompt="给181计划录入一个施基肥，时间是2026年2月19日，作业人张三。",
        )
        self.assertTrue(decision.should_extract)

    def test_crop_calendar_vague_prompt_does_not_extract(self) -> None:
        decision = should_extract_for_route(
            action="workflow",
            name="crop_calendar_workflow",
            prompt="生成种植计划",
        )
        self.assertFalse(decision.should_extract)

    def test_crop_calendar_rich_prompt_extracts(self) -> None:
        decision = should_extract_for_route(
            action="workflow",
            name="crop_calendar_workflow",
            prompt="我想建立一个在湖南常德种植的湘早籼24号的移栽方案",
        )
        self.assertTrue(decision.should_extract)

    def test_sowing_enum_only_prompt_extracts(self) -> None:
        decision = should_extract_for_route(
            action="tool",
            name="sowing_suitability_lookup",
            prompt="直播",
        )
        self.assertTrue(decision.should_extract)

    def test_plan_task_single_anchor_field_extracts(self) -> None:
        decision = should_extract_for_route(
            action="tool",
            name="plant_task_create",
            prompt="给id=189的种植计划新增一个任务",
        )
        self.assertTrue(decision.should_extract)

    def test_crop_calendar_single_field_extracts(self) -> None:
        decision = should_extract_for_route(
            action="workflow",
            name="crop_calendar_workflow",
            prompt="直播",
        )
        self.assertTrue(decision.should_extract)

    def test_sowing_rich_prompt_extracts(self) -> None:
        decision = should_extract_for_route(
            action="tool",
            name="sowing_suitability_lookup",
            prompt="美香占2号在长沙什么时候适合播种",
        )
        self.assertTrue(decision.should_extract)


if __name__ == "__main__":
    unittest.main()
