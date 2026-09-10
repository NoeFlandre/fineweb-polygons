"""Tests for the single declaration of directions, versions, and commands."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fineweb_polygons import catalog, registry
from fineweb_polygons.registry import DIRECTIONS

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_every_direction_and_version_identifier_is_unique() -> None:
    direction_ids = [direction.id for direction in DIRECTIONS]
    version_ids = [v.id for d in DIRECTIONS for v in d.versions]
    configs = [v.hf_config for d in DIRECTIONS for v in d.versions]
    prefixes = [v.data_prefix for d in DIRECTIONS for v in d.versions]

    assert len(set(direction_ids)) == len(direction_ids)
    assert len(set(version_ids)) == len(version_ids)
    assert len(set(configs)) == len(configs)
    assert len(set(prefixes)) == len(prefixes)


def test_registry_import_does_not_initialize_pipeline_dependencies() -> None:
    source_root = _REPOSITORY_ROOT / "src"
    environment = {**os.environ, "PYTHONPATH": str(source_root)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import fineweb_polygons.registry; "
                "assert 'pyarrow' not in sys.modules; "
                "assert 'wtpsplit' not in sys.modules"
            ),
        ],
        check=False,
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("direction", DIRECTIONS, ids=lambda d: d.id)
def test_every_version_publishes_under_its_own_direction_prefix(direction) -> None:
    for version in direction.versions:
        assert version.data_prefix.startswith("data/")
        assert version.metadata_prefix.startswith("metadata/")
        for entry in version.data_files():
            assert entry["path"].startswith(f"{version.data_prefix}/")
        for path in version.metadata_files():
            assert path.startswith(f"{version.metadata_prefix}/")


@pytest.mark.parametrize("direction", DIRECTIONS, ids=lambda d: d.id)
def test_a_source_version_always_names_an_earlier_version(direction) -> None:
    seen: list[str] = []
    for version in direction.versions:
        if version.source_version is not None:
            assert version.source_version in seen
        seen.append(version.id)


def test_each_version_declares_one_data_file_per_country() -> None:
    for direction in DIRECTIONS:
        for version in direction.versions:
            assert len(version.files) == len(version.countries), version.id


def test_every_command_targets_a_declared_direction_and_version() -> None:
    for command in registry.commands():
        direction = registry.direction(command.direction)
        for version_id in command.produces:
            assert direction.version(version_id).id == version_id


def test_every_command_name_and_runner_keyword_is_unique() -> None:
    names = [command.name for command in registry.commands()]
    keywords = [command.runner_keyword for command in registry.commands()]

    assert len(set(names)) == len(names)
    assert len(set(keywords)) == len(keywords)


def test_unknown_directions_and_versions_raise() -> None:
    with pytest.raises(KeyError, match="unknown direction"):
        registry.direction("direction-99")
    with pytest.raises(KeyError, match="unknown version"):
        registry.RETRIEVAL.version("v99")


def test_planned_geographic_footprint_direction_has_no_public_version() -> None:
    direction = registry.GEOGRAPHIC_FOOTPRINT

    assert direction.status == "planned"
    assert direction.versions == ()
    assert direction.latest_version is None
    assert direction.package == (
        "fineweb_polygons.directions.text_geographic_footprint"
    )


def test_planned_direction_renders_metadata_without_a_dataset_config() -> None:
    record = catalog.build_direction_record(registry.GEOGRAPHIC_FOOTPRINT, "2026-09-08")

    assert record["latest_version"] is None
    assert record["versions"] == []
    assert record["outputs"] == {"hf_config": None, "data_files": []}


def test_generated_catalog_artifacts_are_committed_and_current() -> None:
    import scripts.build_catalog as build_catalog

    assert build_catalog.main(["--check", "--root", str(_REPOSITORY_ROOT)]) == 0, (
        "generated catalog files are stale; run `just catalog`"
    )


def test_catalog_covers_every_declared_direction_and_version() -> None:
    record = json.loads(
        (_REPOSITORY_ROOT / "metadata" / "catalog.json").read_text(encoding="utf-8")
    )

    assert [entry["id"] for entry in record["directions"]] == [
        direction.id for direction in DIRECTIONS
    ]
    for entry, direction in zip(record["directions"], DIRECTIONS, strict=True):
        assert [version["id"] for version in entry["versions"]] == [
            version.id for version in direction.versions
        ]
        expected_latest = (
            None if direction.latest_version is None else direction.latest_version.id
        )
        assert entry["latest_version"] == expected_latest


def test_huggingface_configs_match_the_registry() -> None:
    configs = catalog.build_hf_configs()

    assert [entry["config_name"] for entry in configs] == [
        version.hf_config for direction in DIRECTIONS for version in direction.versions
    ]


def test_lexical_facade_loads_legacy_exports_lazily() -> None:
    from fineweb_polygons.directions import lexical
    from fineweb_polygons.directions.lexical.v1.models import (
        DIRECTION_VERSION,
        Direction2RunConfig,
    )

    assert lexical.DIRECTION_VERSION == DIRECTION_VERSION
    assert lexical.Direction2RunConfig is Direction2RunConfig
    assert callable(lexical.run_direction2)


def test_lexical_facade_rejects_unknown_exports() -> None:
    from fineweb_polygons.directions import lexical

    unknown_name = "not_an_export"
    with pytest.raises(AttributeError, match="no attribute"):
        getattr(lexical, unknown_name)


def test_retrieval_facade_rejects_unknown_exports() -> None:
    from fineweb_polygons.directions import retrieval

    unknown_name = "not_an_export"
    with pytest.raises(
        AttributeError,
        match=r"module 'fineweb_polygons\.directions\.retrieval' has no attribute",
    ):
        getattr(retrieval, unknown_name)


def test_lazy_runner_imports_and_forwards_config(monkeypatch) -> None:
    import_calls: list[str | None] = []
    received: list[object] = []

    class FakeModule:
        @staticmethod
        def execute(config: object) -> object:
            received.append(config)
            return config

    def fake_import(module_name: str | None) -> FakeModule:
        import_calls.append(module_name)
        return FakeModule()

    monkeypatch.setattr(registry, "import_module", fake_import)
    config = object()
    runner = registry._lazy_runner("example.module", "execute")

    assert runner(config) is config
    assert import_calls == ["example.module"]
    assert received == [config]


def test_each_direction_facade_resolves_and_caches_its_declared_exports() -> None:
    from fineweb_polygons.directions import lexical, retrieval

    for facade in (lexical, retrieval):
        for name in tuple(facade._LAZY_EXPORTS):
            facade.__dict__.pop(name, None)
            value = getattr(facade, name)
            assert value is not None
            assert getattr(facade, name) is value
