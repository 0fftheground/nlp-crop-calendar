import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.agent.router import RequestRouter
from src.schemas.models import UserRequest


def _load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"cases file not found: {path}")
    if path.suffix == ".jsonl":
        cases = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
        return cases
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON cases must be a list")
        return payload
    raise ValueError("cases file must be .jsonl or .json")


def _evaluate_case(router: RequestRouter, case: dict) -> dict:
    prompt = case.get("prompt", "")
    session_id = case.get("session_id")
    expected = case.get("expect", {})
    started = time.time()
    response = router.handle(UserRequest(prompt=prompt, session_id=session_id))
    elapsed = time.time() - started

    actual_mode = response.mode
    actual_tool = response.tool.name if response.tool else None
    expected_mode = expected.get("mode")
    expected_tool = expected.get("tool")

    ok = actual_mode == expected_mode
    if expected_mode == "tool":
        ok = ok and actual_tool == expected_tool

    return {
        "id": case.get("id", ""),
        "prompt": prompt,
        "expected_mode": expected_mode,
        "expected_tool": expected_tool,
        "actual_mode": actual_mode,
        "actual_tool": actual_tool,
        "ok": ok,
        "latency_sec": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate intent routing and tool/workflow execution."
    )
    parser.add_argument(
        "--cases",
        default=str(ROOT / "tests" / "intent_routing_cases.jsonl"),
        help="Path to .jsonl/.json test cases.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any case fails.",
    )
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    if not cases:
        print("No cases found.")
        return 1

    router = RequestRouter()
    results = []
    for case in cases:
        try:
            result = _evaluate_case(router, case)
        except Exception as exc:
            result = {
                "id": case.get("id", ""),
                "prompt": case.get("prompt", ""),
                "expected_mode": case.get("expect", {}).get("mode"),
                "expected_tool": case.get("expect", {}).get("tool"),
                "actual_mode": "error",
                "actual_tool": None,
                "ok": False,
                "error": str(exc),
            }
        results.append(result)

    passed = sum(1 for r in results if r.get("ok"))
    failed = len(results) - passed

    for result in results:
        status = "PASS" if result.get("ok") else "FAIL"
        print(
            f"[{status}] {result.get('id')} "
            f"expected={result.get('expected_mode')}/{result.get('expected_tool')} "
            f"actual={result.get('actual_mode')}/{result.get('actual_tool')} "
            f"latency={result.get('latency_sec', '-')}"
        )
        if result.get("error"):
            print(f"  error={result.get('error')}")

    print(f"\nTotal: {len(results)}  Passed: {passed}  Failed: {failed}")
    if args.strict and failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
