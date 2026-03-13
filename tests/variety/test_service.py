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
    from src.application.services import variety_service
    from src.infra.config import get_config
    from src.infra.tool_cache import get_tool_result_cache


@unittest.skipUnless(
    not _MISSING_PYDANTIC_SETTINGS, "pydantic_settings is not installed"
)
class VarietyServiceScenarioTests(unittest.TestCase):
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

    def test_variety_service_scenarios(self) -> None:
        scenarios = load_yaml_scenarios("variety/service.yaml")["feature_cases"]
        for scenario in scenarios:
            with self.subTest(scenario=scenario["id"]):
                if scenario["id"] == "requires_exact_name":
                    prompt = json.dumps(scenario["prompt"], ensure_ascii=False)
                    lookup_records = scenario["lookup_records"]
                    with patch(
                        "src.application.services.variety_service._lookup_variety_records",
                        return_value=(
                            lookup_records["records"],
                            lookup_records["raw_records"],
                        ),
                    ), patch(
                        "src.application.services.variety_service.retrieve_variety_candidates",
                        return_value=lookup_records["candidates"],
                    ):
                        result = variety_service.lookup_variety(prompt)

                    self.assertEqual(result.name, "variety_lookup")
                    for snippet in scenario["expected"]["message_contains"]:
                        self.assertIn(snippet, result.message)
                    self.assertEqual(
                        result.data.get("missing_fields"),
                        scenario["expected"]["missing_fields"],
                    )
                    continue

                original_query = scenario["original_query"]
                candidate_followup_prompt = json.dumps(
                    scenario["candidate_followup_prompt"], ensure_ascii=False
                )
                fake_lookup_config = scenario["fake_lookup"]

                def fake_lookup(prompt, *, limit=5, confirmed_candidate=None):
                    prompt_text = str(prompt)
                    if confirmed_candidate == fake_lookup_config["confirmed_candidate"]:
                        return (
                            fake_lookup_config["records"],
                            fake_lookup_config["raw_records"],
                        )
                    if fake_lookup_config["bug_trigger_keyword"] in prompt_text:
                        return (
                            fake_lookup_config["bug_trigger_records"],
                            fake_lookup_config["bug_trigger_raw_records"],
                        )
                    return (
                        fake_lookup_config["fallback_records"],
                        fake_lookup_config["fallback_raw_records"],
                    )

                with patch(
                    "src.application.services.variety_service._lookup_variety_records",
                    side_effect=fake_lookup,
                ):
                    step1 = variety_service.lookup_variety(candidate_followup_prompt)
                    self.assertEqual(step1.name, "variety_lookup")
                    for snippet in scenario["expected"]["step1_message_contains"]:
                        self.assertIn(snippet, step1.message)
                    self.assertEqual(step1.data.get("query"), original_query)

                    region_followup = dict(scenario["region_followup_prompt"])
                    region_followup["query"] = step1.data.get("query")
                    region_followup["followup"]["draft"] = step1.data.get("draft")
                    region_followup["followup"]["missing_fields"] = step1.data.get(
                        "missing_fields"
                    )
                    step2 = variety_service.lookup_variety(
                        json.dumps(region_followup, ensure_ascii=False)
                    )

                self.assertEqual(step2.name, "variety_lookup")
                for snippet in scenario["expected"]["step2_message_contains"]:
                    self.assertIn(snippet, step2.message)
                self.assertEqual(
                    (step2.data or {}).get("variety"),
                    scenario["expected"]["step2_variety"],
                )


if __name__ == "__main__":
    unittest.main()
