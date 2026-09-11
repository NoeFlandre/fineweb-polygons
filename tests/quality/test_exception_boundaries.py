"""Guard exception boundaries against broad catches."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_CODE_ROOTS = (_REPOSITORY_ROOT / "src", _REPOSITORY_ROOT / "scripts")
pytestmark = pytest.mark.architecture


def _broad_handlers() -> list[str]:
    handlers: list[str] = []
    for root in _CODE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler):
                    continue
                if _catches_broad_exception(node.type):
                    handlers.append(f"{path}:{node.lineno}")
    return handlers


def _catches_broad_exception(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Name):
        return node.id in {"Exception", "BaseException"}
    return isinstance(node, ast.Tuple) and any(
        _catches_broad_exception(item) for item in node.elts
    )


def test_production_code_does_not_catch_broad_exceptions() -> None:
    assert _broad_handlers() == []
