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
_CATALOG = _REPOSITORY_ROOT / "metadata" / "catalog.json"
_REPOSITORY_CATALOG = _REPOSITORY_ROOT / "docs" / "dataset-catalog.json"
_MKDOCS_CONFIG = _REPOSITORY_ROOT / "mkdocs.yml"


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


def test_repository_and_public_catalogs_share_the_same_direction_record() -> None:
    public_catalog = json.loads(_CATALOG.read_text(encoding="utf-8"))
    repository_catalog = json.loads(_REPOSITORY_CATALOG.read_text(encoding="utf-8"))

    assert repository_catalog["catalog_date"] == public_catalog["catalog_date"]
    assert repository_catalog["direction"] == public_catalog["direction"]
    assert repository_catalog["versions"] == public_catalog["versions"]


def test_direction_document_covers_each_version_and_handoff() -> None:
    document = _DIRECTION_DOCUMENT.read_text(encoding="utf-8")

    assert "Direction 1" in document
    assert "Frozen" in document
    assert "https://github.com/NoeFlandre/fineweb-polygons" in document
    assert "https://huggingface.co/datasets/NoeFlandre/fineweb-polygons" in document
    for number in range(1, 11):
        assert f"### V{number}" in document
    assert "Direction 2" in document


def test_mkdocs_quotes_the_direction_label_containing_a_colon() -> None:
    config = _MKDOCS_CONFIG.read_text(encoding="utf-8")

    assert (
        '      - "Direction 1: FineWeb retrieval": '
        "directions/fineweb-retrieval/README.md"
    ) in config


def test_mkdocs_direction_links_use_explicit_documentation_targets() -> None:
    docs_with_direction_links = (
        _REPOSITORY_ROOT / "docs" / "index.md",
        _REPOSITORY_ROOT / "docs" / "dataset-catalog.md",
        _REPOSITORY_ROOT / "docs" / "development.md",
        _REPOSITORY_ROOT / "docs" / "versions.md",
    )

    for path in docs_with_direction_links:
        document = path.read_text(encoding="utf-8")
        assert "](directions/fineweb-retrieval/)" not in document

    direction_document = _DIRECTION_DOCUMENT.read_text(encoding="utf-8")
    assert "](../../../metadata/catalog.json)" not in direction_document
    assert (
        "https://github.com/NoeFlandre/fineweb-polygons/blob/main/metadata/catalog.json"
        in direction_document
    )
