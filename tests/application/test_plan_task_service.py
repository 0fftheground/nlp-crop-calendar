import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_MISSING_PYDANTIC_SETTINGS = importlib.util.find_spec("pydantic_settings") is None

if not _MISSING_PYDANTIC_SETTINGS:
    from src.application.services.plan_task_service import _format_plan_task_http_error


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class PlanTaskServiceTests(unittest.TestCase):
    def test_format_plan_task_http_error_prefers_detail(self) -> None:
        message = _format_plan_task_http_error(
            status_code=404,
            response_text='{"detail":"Plan not found"}',
        )
        self.assertEqual(message, "农事录入失败：Plan not found")

    def test_format_plan_task_http_error_falls_back_to_status(self) -> None:
        message = _format_plan_task_http_error(
            status_code=404,
            response_text="",
        )
        self.assertEqual(message, "农事录入失败（status=404）。")


if __name__ == "__main__":
    unittest.main()
