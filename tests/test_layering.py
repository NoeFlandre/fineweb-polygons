"""Structural guards that keep the direction layering from eroding.

These tests are the reason a new direction can be added without a cleanup
afterwards: they fail the moment `core` learns about a direction, two
directions reach into each other, the CLI hard-codes a version, or a version
is declared anywhere but the registry.
"""

from __future__ import annotations

import ast
import re
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

import pytest

from fineweb_polygons import registry

_SOURCE_ROOT = Path(registry.__file__).resolve().parent
_PACKAGE = "fineweb_polygons"
pytestmark = pytest.mark.architecture


def _modules(relative: str) -> list[Path]:
    return sorted((_SOURCE_ROOT / relative).rglob("*.py"))


def _direction_modules() -> list[Path]:
    return sorted(
        module
        for direction in registry.DIRECTIONS
        for module in _modules(
            direction.package.removeprefix(f"{_PACKAGE}.").replace(".", "/")
        )
    )


def _module_name(path: Path) -> str:
    relative = path.relative_to(_SOURCE_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((_PACKAGE, *parts))


def _source_dependencies() -> dict[str, set[str]]:
    paths = sorted(_SOURCE_ROOT.rglob("*.py"))
    known = {_module_name(path) for path in paths}
    dependencies: dict[str, set[str]] = {}
    for path in paths:
        imported = _imported_packages(path)
        dependencies[_module_name(path)] = {
            candidate for name in imported for candidate in _parent_modules(name, known)
        }
    return dependencies


def _parent_modules(name: str, known: set[str]) -> tuple[str, ...]:
    parts = name.split(".")
    return tuple(
        ".".join(parts[:index])
        for index in range(len(parts), 1, -1)
        if ".".join(parts[:index]) in known
    )[:1]


def _imported_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return {name for name in imported if name.startswith(_PACKAGE)}


def test_source_import_graph_is_acyclic() -> None:
    try:
        tuple(TopologicalSorter(_source_dependencies()).static_order())
    except CycleError as error:
        pytest.fail(f"source import cycle detected: {error}")


@pytest.mark.parametrize("module", _modules("core"), ids=lambda p: p.name)
def test_core_never_depends_on_a_direction(module: Path) -> None:
    offending = {
        name
        for name in _imported_packages(module)
        if not name.startswith(f"{_PACKAGE}.core")
    }
    assert offending == set(), (
        f"{module.name} makes core depend on {sorted(offending)}; core must stay "
        "direction-agnostic"
    )


@pytest.mark.parametrize(
    "module",
    _direction_modules(),
    ids=lambda p: str(p.relative_to(_SOURCE_ROOT)),
)
def test_directions_never_import_each_other(module: Path) -> None:
    own = (
        f"{_PACKAGE}.directions."
        + module.relative_to(_SOURCE_ROOT / "directions").parts[0]
    )
    offending = {
        name
        for name in _imported_packages(module)
        if name.startswith(f"{_PACKAGE}.directions.") and not name.startswith(own)
    }
    assert offending == set(), (
        f"{module.name} imports another direction: {sorted(offending)}"
    )


@pytest.mark.parametrize(
    "module",
    _direction_modules(),
    ids=lambda p: str(p.relative_to(_SOURCE_ROOT)),
)
def test_directions_never_depend_on_the_registry_or_the_cli(module: Path) -> None:
    offending = {
        name
        for name in _imported_packages(module)
        if name in {f"{_PACKAGE}.registry", f"{_PACKAGE}.cli", f"{_PACKAGE}.catalog"}
    }
    assert offending == set(), (
        f"{module.name} depends on {sorted(offending)}; the registry composes "
        "directions, not the other way round"
    )


def test_the_cli_never_names_a_version() -> None:
    source = (_SOURCE_ROOT / "cli.py").read_text(encoding="utf-8")

    assert re.search(r"\bv\d+\b", source) is None, (
        "cli.py names a version; the command table belongs in registry.py so "
        "adding a version never touches the CLI"
    )
    assert "direction2" not in source
    assert "lexical" not in source


def test_public_paths_are_declared_once_and_read_by_the_registry() -> None:
    declaring = {
        path.relative_to(_SOURCE_ROOT)
        for path in _SOURCE_ROOT.rglob("*.py")
        if re.search(r'"data/direction-', path.read_text(encoding="utf-8"))
    }
    allowed = {Path("registry.py")} | {
        Path(f"directions/lexical/{name}/models.py") for name in ("v1", "v2", "v3")
    }

    assert declaring <= allowed, (
        f"{sorted(declaring - allowed)} hard-code a public dataset path; a path "
        "belongs to its version's contract and is read from there"
    )


def test_registry_prefixes_match_each_version_contract() -> None:
    from fineweb_polygons.directions.lexical.v1 import models as v1
    from fineweb_polygons.directions.lexical.v2 import models as v2
    from fineweb_polygons.directions.lexical.v3 import models as v3

    declared = {
        v1.DIRECTION_VERSION: (v1.DATA_PREFIX, v1.HF_CONFIG_NAME),
        v2.DIRECTION_V2_VERSION: (v2.DATA_PREFIX, v2.HF_CONFIG_NAME_V2),
        v3.DIRECTION_V3_VERSION: (v3.DATA_PREFIX, v3.HF_CONFIG_NAME_V3),
    }
    for version in registry.LEXICAL.versions:
        assert (version.data_prefix, version.hf_config) == declared[version.id]


def test_every_direction_is_a_real_importable_package() -> None:
    for direction in registry.DIRECTIONS:
        relative = direction.package.removeprefix(f"{_PACKAGE}.").replace(".", "/")

        assert (_SOURCE_ROOT / relative / "__init__.py").is_file(), (
            f"{direction.id} declares a package that does not exist"
        )
