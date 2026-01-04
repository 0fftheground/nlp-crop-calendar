import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.intent_rules import IntentRouter


def _load_cases() -> list[dict]:
    path = Path(__file__).resolve().parent / "intent_routing_cases.jsonl"
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


class IntentRoutingTests(unittest.TestCase):
    def test_intent_cases(self) -> None:
        router = IntentRouter()
        cases = _load_cases()
        for case in cases:
            prompt = case.get("prompt", "")
            session_id = case.get("session_id")
            expected = case.get("expect", {})
            expected_mode = expected.get("mode")
            expected_tool = expected.get("tool")

            actual_mode, actual_tool = router.route(prompt, session_id=session_id)

            self.assertEqual(
                actual_mode,
                expected_mode,
                msg=f"{case.get('id')}: mode mismatch",
            )
            if expected_mode == "tool":
                self.assertEqual(
                    actual_tool,
                    expected_tool,
                    msg=f"{case.get('id')}: tool mismatch",
                )
            else:
                self.assertIsNone(
                    actual_tool,
                    msg=f"{case.get('id')}: expected no tool",
                )


if __name__ == "__main__":
    unittest.main()
