from dataclasses import replace
from pathlib import Path

import pytest

from fineweb_polygons.directions.lexical.models import PolygonRecord
from fineweb_polygons.directions.lexical.v2.specificity import (
    build_name_inventory,
)
from fineweb_polygons.directions.lexical.v3.evidence import score_candidate
from fineweb_polygons.directions.lexical.v3.models import (
    DATA_PREFIX,
    DIRECTION_V3_VERSION,
    HF_CONFIG_NAME_V3,
    Direction2V3RunConfig,
)


def _config(tmp_path: Path) -> Direction2V3RunConfig:
    monaco = tmp_path / "monaco.osm.pbf"
    liechtenstein = tmp_path / "liechtenstein.osm.pbf"
    shard = tmp_path / "shard.parquet"
    return Direction2V3RunConfig(
        monaco_pbf=monaco,
        liechtenstein_pbf=liechtenstein,
        shard_path=shard,
        output_dir=tmp_path / "artifacts",
        manifest_path=tmp_path / "runs" / "manifest.json",
        dataset_card_path=tmp_path / "card.md",
        log_path=tmp_path / "logs" / "run.jsonl",
        name_inventory_path=tmp_path / "runs" / "name-inventory.json",
    )


def _profile(name: str, *, polygon_count: int = 1, frequency: int = 1):
    polygons = tuple(
        PolygonRecord(
            polygon_id=f"monaco/way/{index}",
            source_key="monaco",
            name=name,
            aliases=(),
            tags=(),
            centroid=None,
        )
        for index in range(polygon_count)
    )
    profiles = build_name_inventory(
        polygons,
        document_frequencies={name.casefold(): frequency},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )
    return next(profile for profile in profiles if profile.normalized_name)


def test_v3_contract_constants_are_stable() -> None:
    assert DIRECTION_V3_VERSION == "direction-2-lexical-v3"
    assert HF_CONFIG_NAME_V3 == "direction_2_lexical_v3"
    assert DATA_PREFIX == "data/direction-2-lexical/v3"


def test_v3_config_rejects_non_positive_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        replace(_config(tmp_path), batch_size=0)


def test_v3_config_rejects_duplicate_input_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="input paths must be different"):
        replace(config, liechtenstein_pbf=config.monaco_pbf)


def test_generic_name_with_url_only_is_possible() -> None:
    evidence = score_candidate(
        _profile("Central", polygon_count=2, frequency=100),
        alias="Central",
        sentence="Central has a long history.",
        context="Central has a long history.",
        url="https://example.test/central",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )

    assert evidence.name_in_url is True
    assert evidence.score == 3
    assert evidence.tier == "possible"
    assert evidence.reasons == ("name_in_url",)


def test_generic_name_with_country_sentence_and_url_is_high_confidence() -> None:
    evidence = score_candidate(
        _profile("Central", polygon_count=2, frequency=100),
        alias="Central",
        sentence="Central in Monaco is a place.",
        context="Central in Monaco is a place.",
        url="https://example.test/central",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )

    assert evidence.score == 6
    assert evidence.tier == "high_confidence"
    assert evidence.reasons == ("name_in_url", "country_in_sentence")


def test_distinctive_name_without_context_is_possible() -> None:
    evidence = score_candidate(
        _profile("Palais du Prince"),
        alias="Palais du Prince",
        sentence="Palais du Prince is mentioned.",
        context="Palais du Prince is mentioned.",
        url="https://example.test/article",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )

    assert evidence.score == 2
    assert evidence.tier == "possible"
    assert evidence.reasons == ("distinctive_name",)


def test_context_evidence_is_scored_without_double_counting_country() -> None:
    evidence = score_candidate(
        _profile("Central", polygon_count=2, frequency=100),
        alias="Central",
        sentence="Central has a long history.",
        context="Central has a long history. Monaco is nearby.",
        url="https://example.test/article",
        country_name="Monaco",
        same_polygon_alias_nearby=True,
        other_polygon_name_nearby=True,
    )

    assert evidence.country_in_sentence is False
    assert evidence.country_in_context is True
    assert evidence.score == 4
    assert evidence.tier == "high_confidence"
    assert evidence.reasons == (
        "country_in_context",
        "same_polygon_alias_nearby",
        "other_polygon_name_nearby",
    )


def test_generic_name_without_evidence_is_rejected() -> None:
    evidence = score_candidate(
        _profile("Central", polygon_count=2, frequency=100),
        alias="Central",
        sentence="Central has a long history.",
        context="Central has a long history.",
        url="https://example.test/article",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )

    assert evidence.score == 0
    assert evidence.tier == "rejected"
    assert evidence.reasons == ()
