from __future__ import annotations

import sys
from typing import Iterable, List

from . import audit as audit_module
from . import cli as run_module
from . import promotion_import as promote_module
from . import release_compare as compare_module


def _print_top_level_help() -> None:
    print("usage: python -m src.eval_platform [run|compare|audit|promote] ...")
    print()
    print("commands:")
    print("  run       run offline eval datasets or governance profiles")
    print("  compare   compare baseline and candidate models")
    print("  audit     run production-audit sampling and review utilities")
    print("  promote   import production-audit promotion candidates")
    print()
    print("examples:")
    print("  python -m src.eval_platform run --profile expert_blocking_gate")
    print("  python -m src.eval_platform compare --candidate-llm-model gpt-5-mini")
    print("  python -m src.eval_platform audit run-latest --limit 50 --days 30")
    print("  python -m src.eval_platform promote --promotion path/to/file.yaml")
    print()
    print("backward compatibility:")
    print("  python -m src.eval_platform --profile expert_blocking_gate")


def main(argv: Iterable[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_top_level_help()
        return 0
    command = args[0]
    if command in {"-h", "--help", "help"}:
        _print_top_level_help()
        return 0
    if command == "run":
        return int(run_module.main(args[1:]))
    if command == "compare":
        return int(compare_module.main(args[1:]))
    if command == "audit":
        return int(audit_module.main(args[1:]))
    if command == "promote":
        return int(promote_module.main(args[1:]))
    return int(run_module.main(args))
