import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
DOMAIN_ROOT = SRC_ROOT / "domain"


def _module_name_for_file(path: Path) -> str:
    rel = path.relative_to(SRC_ROOT)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(["src", *parts]) if parts else "src"


def _resolve_relative_import(cur_mod: str, level: int, module: str) -> str:
    base = cur_mod.split(".")[:-1]
    up = max(level - 1, 0)
    prefix = base[: len(base) - up] if up <= len(base) else []
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


class ArchitectureDependencyTests(unittest.TestCase):
    def test_domain_does_not_import_schemas(self) -> None:
        violations: list[str] = []
        for path in DOMAIN_ROOT.rglob("*.py"):
            cur_mod = _module_name_for_file(path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name
                        if name == "src.schemas" or name.startswith("src.schemas."):
                            violations.append(f"{path}:{node.lineno} -> import {name}")
                elif isinstance(node, ast.ImportFrom):
                    target = ""
                    if node.level:
                        target = _resolve_relative_import(
                            cur_mod, node.level, node.module or ""
                        )
                    elif node.module:
                        target = node.module
                    if target == "src.schemas" or target.startswith("src.schemas."):
                        violations.append(
                            f"{path}:{node.lineno} -> from {target} import ..."
                        )
        self.assertFalse(
            violations,
            "Domain layer should not depend on schemas layer:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
