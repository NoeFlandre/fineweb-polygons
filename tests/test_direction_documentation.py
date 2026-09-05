"""Tests for the public research-direction archive."""

from __future__ import annotations

import json
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DIRECTION_DOCUMENT = (
    _REPOSITORY_ROOT / "docs" / "directions" / "fineweb-retrieval" / "README.md"
)
_DIRECTION_RECORD = (
    _REPOSITORY_ROOT / "metadata" / "directions" / "direction-1-fineweb-retrieval.json"
)
_DIRECTION_2_DOCUMENT = (
    _REPOSITORY_ROOT / "docs" / "directions" / "lexical-candidates" / "README.md"
)
_DIRECTION_2_V2_DOCUMENT = (
    _REPOSITORY_ROOT
    / "docs"
    / "directions"
    / "lexical-candidates"
    / "lexical-v2"
    / "README.md"
)
_DIRECTION_2_RECORD = (
    _REPOSITORY_ROOT / "metadata" / "directions" / "direction-2-lexical-candidates.json"
)
_CATALOG = _REPOSITORY_ROOT / "metadata" / "catalog.json"
_REPOSITORY_CATALOG = _REPOSITORY_ROOT / "docs" / "dataset-catalog.json"


def test_direction_record_freezes_the_complete_v1_to_v10_line() -> None:
    record = json.loads(_DIRECTION_RECORD.read_text(encoding="utf-8"))

    assert record["direction_id"] == "direction-1-fineweb-retrieval"
    assert record["status"] == "frozen"
    assert record["latest_version"] == "v10"
    assert record["versions"] == [f"v{number}" for number in range(1, 11)]
    assert record["github"] == "https://github.com/NoeFlandre/fineweb-polygons"
    assert record["huggingface"] == (
        "https://huggingface.co/datasets/NoeFlandre/fineweb-polygons"
    )


def test_catalog_points_to_the_frozen_direction() -> None:
    catalog = json.loads(_CATALOG.read_text(encoding="utf-8"))

    assert catalog["direction"]["id"] == "direction-1-fineweb-retrieval"
    assert catalog["direction"]["status"] == "frozen"
    assert catalog["direction"]["latest_version"] == "v10"
    assert [version["id"] for version in catalog["versions"]] == [
        f"v{number}" for number in range(1, 11)
    ]


def test_direction2_has_a_separate_contract_and_public_namespace() -> None:
    record = json.loads(_DIRECTION_2_RECORD.read_text(encoding="utf-8"))
    document = _DIRECTION_2_DOCUMENT.read_text(encoding="utf-8")

    assert record["direction_id"] == "direction-2-lexical-candidates"
    assert record["latest_version"] == "direction-2-lexical-v2"
    assert record["versions"] == [
        "direction-2-lexical-v1",
        "direction-2-lexical-v2",
    ]
    assert record["outputs"]["hf_config"] == "direction_2_lexical_v2"
    assert record["outputs"]["data_files"][0]["path"].startswith(
        "data/direction-2/lexical-v2/"
    )
    assert record["historical_outputs"]["direction-2-lexical-v1"]["hf_config"] == (
        "direction_2_lexical_v1"
    )
    assert "Aho" in document
    assert "no LLM" in document
    assert "direction-2-lexical-v2" in document
    assert "frequency" in _DIRECTION_2_V2_DOCUMENT.read_text(encoding="utf-8")

    catalog = json.loads(_CATALOG.read_text(encoding="utf-8"))
    direction2 = catalog["additional_directions"][0]
    assert direction2["id"] == record["direction_id"]
    assert direction2["latest_version"] == "direction-2-lexical-v2"
    assert direction2["huggingface_config"] == "direction_2_lexical_v2"
    assert direction2["historical_huggingface_configs"] == ["direction_2_lexical_v1"]
    assert any(
        data_file["path"] == "data/direction-2/lexical-v2/monaco.parquet"
        for version in direction2["versions"]
        for data_file in version["data_files"]
    )


def test_repository_and_public_catalogs_share_the_same_direction_record() -> None:
    public_catalog = json.loads(_CATALOG.read_text(encoding="utf-8"))
    repository_catalog = json.loads(_REPOSITORY_CATALOG.read_text(encoding="utf-8"))

    assert repository_catalog["catalog_date"] == public_catalog["catalog_date"]
    assert repository_catalog["direction"] == public_catalog["direction"]
    assert repository_catalog["versions"] == public_catalog["versions"]
    assert (
        repository_catalog["additional_directions"]
        == public_catalog["additional_directions"]
    )


def test_direction_document_covers_each_version_and_handoff() -> None:
    document = _DIRECTION_DOCUMENT.read_text(encoding="utf-8")

    assert "Direction 1" in document
    assert "Frozen" in document
    assert "https://github.com/NoeFlandre/fineweb-polygons" in document
    assert "https://huggingface.co/datasets/NoeFlandre/fineweb-polygons" in document
    for number in range(1, 11):
        assert f"### V{number}" in document
    assert "Direction 2" in document
