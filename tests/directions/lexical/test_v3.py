import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.directions.lexical.models import PolygonRecord
from fineweb_polygons.directions.lexical.v2.specificity import (
    build_name_inventory,
)
from fineweb_polygons.directions.lexical.v3 import pipeline
from fineweb_polygons.directions.lexical.v3.card import render_dataset_card
from fineweb_polygons.directions.lexical.v3.evidence import (
    CandidateEvidence,
    score_candidate,
)
from fineweb_polygons.directions.lexical.v3.matching import V3NameMatch, V3NameMatcher
from fineweb_polygons.directions.lexical.v3.models import (
    DATA_PREFIX,
    DIRECTION_V3_VERSION,
    HF_CONFIG_NAME_V3,
    OUTPUT_COLUMNS_V3,
    Direction2V3RunConfig,
)
from fineweb_polygons.directions.lexical.v3.pipeline import run_direction2_v3


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


def test_v3_config_rejects_non_positive_output_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="output_batch_size must be positive"):
        replace(_config(tmp_path), output_batch_size=0)


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


def test_score_candidate_rejects_empty_country_name() -> None:
    with pytest.raises(ValueError, match=r"\Acountry_name must not be empty\Z"):
        score_candidate(
            _profile("Central"),
            alias="Central",
            sentence="Central is here.",
            context="Central is here.",
            url="https://example.test/article",
            country_name=" ",
            same_polygon_alias_nearby=False,
            other_polygon_name_nearby=False,
        )


def _write_osm(path: Path, names: tuple[tuple[str, str], ...]) -> None:
    ways = []
    for index, (name, alias) in enumerate(names, start=10):
        tags = [f'<tag k="name" v="{name}" />']
        if alias:
            tags.append(f'<tag k="name:en" v="{alias}" />')
        ways.append(
            "\n".join(
                [
                    f'  <way id="{index}">',
                    '    <nd ref="1" />',
                    '    <nd ref="2" />',
                    '    <nd ref="3" />',
                    '    <nd ref="4" />',
                    '    <nd ref="1" />',
                    *[f"    {tag}" for tag in tags],
                    "  </way>",
                ]
            )
        )
    path.write_text(
        "\n".join(
            [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<osm version="0.6" generator="test">',
                '  <node id="1" lat="43.70" lon="7.40" />',
                '  <node id="2" lat="43.70" lon="7.41" />',
                '  <node id="3" lat="43.71" lon="7.41" />',
                '  <node id="4" lat="43.71" lon="7.40" />',
                *ways,
                "</osm>",
            ]
        ),
        encoding="utf-8",
    )


def _make_v3_fixture(tmp_path: Path) -> Direction2V3RunConfig:
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    _write_osm(monaco, (("Palais du Prince", "Prince's Palace"), ("Central", "")))
    _write_osm(liechtenstein, (("Alps View", ""), ("Central", "")))
    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(
                    [
                        "Central appears on a page.",
                        "Central in Monaco is a place.",
                        "Palais du Prince is visible.",
                        "Alps View in Liechtenstein is visible.",
                        "Central has no country context.",
                    ]
                ),
                pa.array(
                    [
                        "https://example.test/central",
                        "https://example.test/article",
                        "https://example.test/palais-du-prince",
                        "https://example.test/article-2",
                        "https://example.test/article-3",
                    ]
                ),
            ],
            names=["text", "url"],
        ),
        shard,
    )
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


def test_v3_run_writes_all_tiers_and_reproducibility_artifacts(tmp_path: Path) -> None:
    config = _make_v3_fixture(tmp_path)

    summary = run_direction2_v3(config)

    assert summary.direction == DIRECTION_V3_VERSION
    assert summary.polygons_read == 4
    assert summary.fineweb_docs_frequency_pass == 5
    assert summary.fineweb_docs_match_pass == 5
    assert summary.matches_found == 8
    assert summary.high_confidence_matches == 2
    assert summary.possible_matches == 3
    assert summary.rejected_matches == 3
    assert summary.unique_polygons_matched == 4
    assert summary.names_considered == 4
    assert summary.names_indexed == 4
    assert summary.names_discarded == 0
    assert summary.generic_names == 1
    assert [item.source_key for item in summary.country_summaries] == [
        "monaco",
        "liechtenstein",
    ]
    assert summary.output_paths == (
        config.output_dir / "monaco.parquet",
        config.output_dir / "liechtenstein.parquet",
    )
    assert summary.manifest_path == config.manifest_path
    assert summary.dataset_card_path == config.dataset_card_path
    assert summary.log_path == config.log_path
    assert summary.name_inventory_path == config.name_inventory_path

    monaco = pq.read_table(config.output_dir / "monaco.parquet")
    liechtenstein = pq.read_table(config.output_dir / "liechtenstein.parquet")
    assert monaco.column_names == list(OUTPUT_COLUMNS_V3)
    assert liechtenstein.column_names == list(OUTPUT_COLUMNS_V3)
    assert set(monaco["decision_tier"].to_pylist()) == {
        "high_confidence",
        "possible",
        "rejected",
    }
    assert set(liechtenstein["decision_tier"].to_pylist()) == {
        "high_confidence",
        "possible",
        "rejected",
    }
    assert "Palais du Prince is visible." in monaco["sentence"].to_pylist()
    assert (
        "Alps View in Liechtenstein is visible."
        in liechtenstein["sentence"].to_pylist()
    )
    assert config.manifest_path.is_file()
    assert config.dataset_card_path.is_file()
    assert config.log_path.is_file()
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["direction"] == DIRECTION_V3_VERSION
    assert manifest["status"] == "complete"
    assert manifest["schema"] == list(OUTPUT_COLUMNS_V3)
    assert manifest["sources"]["fineweb_shard"]["path"] == str(config.shard_path)
    assert len(manifest["sources"]["fineweb_shard"]["sha256"]) == 64
    assert [item["source_key"] for item in manifest["sources"]["osm_pbf"]] == [
        "monaco",
        "liechtenstein",
    ]
    assert [item["path"] for item in manifest["sources"]["osm_pbf"]] == [
        str(config.monaco_pbf),
        str(config.liechtenstein_pbf),
    ]
    assert all(len(item["sha256"]) == 64 for item in manifest["sources"]["osm_pbf"])
    result_files = manifest["results"]["files"]
    assert [item["path"] for item in result_files] == [
        f"{DATA_PREFIX}/monaco.parquet",
        f"{DATA_PREFIX}/liechtenstein.parquet",
    ]
    assert all(len(item["sha256"]) == 64 for item in result_files)

    card = config.dataset_card_path.read_text(encoding="utf-8")
    assert "# Direction 2 — lexical polygon candidates V3" in card
    assert "- 4 polygon objects read" in card
    assert "- 4 normalized names considered" in card
    assert "- 4 names indexed" in card
    assert "- 1 names classified as generic" in card
    assert "- 0 names discarded" in card
    assert "- 5 FineWeb documents in the frequency pass" in card
    assert "- 5 FineWeb documents in the matching pass" in card
    assert "- 8 candidates written" in card
    assert "- 2 high-confidence candidates" in card
    assert "- 3 possible candidates" in card
    assert "- 3 rejected candidates" in card
    assert "- 4 unique polygons matched" in card
    assert "- FineWeb frequency ratio: 0.001" in card
    assert "- Minimum alphabetic characters: 3" in card
    assert "- Frequency inventory reused: False" in card
    assert "| liechtenstein | 4 | 1 | 1 | 2 |" in card
    assert "| monaco | 4 | 1 | 2 | 1 |" in card
    assert (
        "The base score is 2 for a V2 distinctive name and 0 for a generic "
        "name. Add 3 when the name appears in the URL"
    ) in card
    assert "There is no LLM, NER model, embedding, thematic filter" in card
    assert "https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/" in card
    descriptions = {
        "polygon_id": "stable source/object identifier",
        "polygon_name": "OSM main name value",
        "matched_alias": "name or alias value that matched",
        "osm_tags": "all OSM tags as sorted JSON",
        "centroid": "centroid as JSON with latitude and longitude",
        "fineweb_url": "FineWeb document URL",
        "sentence": "the sentence containing the match",
        "context": "the sentence plus one neighboring sentence on each side",
        "name_match_class": "distinctive_name or generic_name",
        "osm_polygon_count": "number of OSM polygons using the normalized name",
        "fineweb_document_frequency": (
            "FineWeb documents containing the normalized name"
        ),
        "decision_tier": "high_confidence, possible, or rejected",
        "evidence_score": "bounded deterministic score before tiering",
        "evidence_reasons": "pipe-separated reasons contributing to the score",
        "name_in_url": "matched alias appears in the normalized URL",
        "country_in_sentence": "source country appears in the matching sentence",
        "country_in_context": "source country appears only in nearby context",
        "same_polygon_alias_nearby": "another alias of the polygon is nearby",
        "other_polygon_name_nearby": "another same-source polygon name is nearby",
    }
    for column, description in descriptions.items():
        assert f"| {column} | {description} |" in card

    expected_lines = [
        "---",
        "config_name: direction_2_lexical_v3",
        "---",
        "# Direction 2 — lexical polygon candidates V3",
        "",
        "V3 keeps the lexical recall of V2 and adds a small, deterministic "
        "evidence score for generic-name disambiguation.",
        "",
        "## Measured run",
        "",
        "- 4 polygon objects read",
        "- 4 normalized names considered",
        "- 4 names indexed",
        "- 1 names classified as generic",
        "- 0 names discarded",
        "- 5 FineWeb documents in the frequency pass",
        "- 5 FineWeb documents in the matching pass",
        "- 8 candidates written",
        "- 2 high-confidence candidates",
        "- 3 possible candidates",
        "- 3 rejected candidates",
        "- 4 unique polygons matched",
        "",
        "## Decision rule",
        "",
        "The base score is 2 for a V2 distinctive name and 0 for a generic "
        "name. Add 3 when the name appears in the URL, 3 when the country "
        "appears in the matching sentence, 1 when it appears only in the "
        "context window, 2 when another alias of the same polygon is nearby, "
        "and 1 when another same-source polygon name is nearby.",
        "",
        "A score of 4 or more is `high_confidence`, 2-3 is `possible`, and "
        "below 2 is `rejected`. All tiers are retained so the decision is "
        "auditable; filter `decision_tier` to `high_confidence` for the "
        "strictest candidate view.",
        "",
        "There is no LLM, NER model, embedding, thematic filter, URL-only "
        "keep rule, deduplication, or geographic resolver in V3.",
        "",
        "## Configuration",
        "",
        "- FineWeb frequency ratio: 0.001",
        "- Minimum alphabetic characters: 3",
        "- Frequency inventory reused: False",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
    ]
    expected_lines.extend(
        f"| {column} | {description} |" for column, description in descriptions.items()
    )
    expected_lines.extend(
        [
            "",
            "## Source splits",
            "",
            "| Source | Candidates | High confidence | Possible | Rejected |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| liechtenstein | 4 | 1 | 1 | 2 |",
            "| monaco | 4 | 1 | 2 | 1 |",
            "",
            "This card is generated deterministically from the run manifest. "
            "The full contract is in the GitHub V3 README at "
            "https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/"
            "directions/lexical-candidates/v3/README.md. The V1 and V2 pages "
            "remain available for comparison. The deterministic V2/V3 report "
            "is published at "
            "metadata/direction-2-lexical/v3/comparison-v2-v3.md.",
        ]
    )
    assert card == "\n".join(expected_lines) + "\n"

    assert manifest["configuration"] == {
        "evidence_weights": {
            "distinctive_name": 2,
            "name_in_url": 3,
            "country_in_sentence": 3,
            "country_in_context": 1,
            "same_polygon_alias_nearby": 2,
            "other_polygon_name_nearby": 1,
        },
        "fineweb_document_frequency_ratio": 0.001,
        "generic_osm_polygon_count_threshold": 1,
        "minimum_name_letters": 3,
        "normalization_version": manifest["configuration"]["normalization_version"],
        "tier_thresholds": {"high_confidence": 4, "possible": 2},
        "batch_size": 8192,
        "frequency_pass_reused": False,
        "matcher": "Aho-Corasick",
        "output_batch_size": 4096,
        "sentence_context": "matching sentence plus one sentence on each side",
    }
    assert manifest["polygon_inventory"] == {
        "generic_names": 1,
        "names_considered": 4,
        "names_discarded": 0,
        "names_indexed": 4,
        "polygons_read": 4,
    }
    assert manifest["results"]["fineweb_docs_frequency_pass"] == 5
    assert manifest["results"]["fineweb_docs_match_pass"] == 5
    assert manifest["results"]["matches_found"] == 8
    assert manifest["results"]["high_confidence_matches"] == 2
    assert manifest["results"]["possible_matches"] == 3
    assert manifest["results"]["rejected_matches"] == 3
    assert manifest["results"]["unique_polygons_matched"] == 4
    assert [item["source_key"] for item in manifest["results"]["files"]] == [
        "monaco",
        "liechtenstein",
    ]
    assert manifest["name_inventory"]["path"] == str(config.name_inventory_path)
    assert len(manifest["name_inventory"]["sha256"]) == 64

    inventory = json.loads(config.name_inventory_path.read_text(encoding="utf-8"))
    frequencies = {
        item["normalized_name"]: item["decision"]["document_frequency"]
        for item in inventory["names"]
    }
    assert frequencies == {
        "alps view": 1,
        "central": 3,
        "palais du prince": 1,
        "prince s palace": 0,
    }
    assert inventory["summary"] == {
        "generic_names": 1,
        "names_considered": 4,
        "names_discarded": 0,
        "names_indexed": 4,
    }

    events = [
        json.loads(line)
        for line in config.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event"] for event in events] == [
        "run_started",
        "frequency_progress",
        "names_loaded",
        "match_progress",
        "run_completed",
    ]
    assert events[0]["version"] == DIRECTION_V3_VERSION
    assert events[1]["docs_scanned"] == 5
    assert events[2]["names_considered"] == 4
    assert events[2]["names_discarded"] == 0
    assert events[2]["names_indexed"] == 4
    assert events[2]["frequency_pass_reused"] is False
    assert events[3]["docs_scanned"] == 5
    assert events[4]["matches_found"] == 8

    run_direction2_v3(config)

    reused_manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert reused_manifest["configuration"]["frequency_pass_reused"] is True


def test_v3_rejects_missing_input(tmp_path: Path) -> None:
    config = _make_v3_fixture(tmp_path)
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError) as error:
        run_direction2_v3(replace(config, shard_path=missing))

    assert error.value.args == (missing,)


def test_v3_run_uses_zero_document_count_for_initial_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_v3_fixture(tmp_path)
    document_counts: list[int] = []
    original = pipeline.build_name_inventory

    def spy_build_name_inventory(*args: Any, **kwargs: Any) -> Any:
        document_counts.append(kwargs["document_count"])
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_name_inventory", spy_build_name_inventory)

    run_direction2_v3(config)

    assert document_counts == [0, 5]


def test_v3_run_creates_a_deeply_nested_log_parent(tmp_path: Path) -> None:
    config = _make_v3_fixture(tmp_path)
    config = replace(config, log_path=tmp_path / "nested" / "deeper" / "run.jsonl")

    run_direction2_v3(config)

    assert config.log_path.is_file()


def test_v3_run_opens_the_log_with_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_v3_fixture(tmp_path)
    encodings: list[object] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == config.log_path:
            encodings.append(kwargs.get("encoding"))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    run_direction2_v3(config)

    assert encodings == ["utf-8"]


def test_v3_run_reports_discarded_names_in_summary_and_log(tmp_path: Path) -> None:
    config = _make_v3_fixture(tmp_path)
    _write_osm(config.monaco_pbf, (("A", ""),))

    summary = run_direction2_v3(config)

    assert summary.names_considered == 3
    assert summary.names_indexed == 2
    assert summary.names_discarded == 1
    events = [
        json.loads(line)
        for line in config.log_path.read_text(encoding="utf-8").splitlines()
    ]
    loaded = next(event for event in events if event["event"] == "names_loaded")
    assert loaded["names_considered"] == 3
    assert loaded["names_indexed"] == 2
    assert loaded["names_discarded"] == 1
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["polygon_inventory"] == {
        "generic_names": 1,
        "names_considered": 3,
        "names_discarded": 1,
        "names_indexed": 2,
        "polygons_read": 3,
    }


def test_v3_write_inventory_counts_each_decision(tmp_path: Path) -> None:
    profiles = (
        _profile("A"),
        _profile("Central", polygon_count=2, frequency=100),
        _profile("Palais du Prince"),
    )
    path = tmp_path / "runs" / "inventory.json"

    pipeline._write_inventory(
        path,
        profiles,
        fingerprints={"source": "fixture"},
        documents_scanned=7,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["summary"] == {
        "generic_names": 1,
        "names_considered": 3,
        "names_discarded": 1,
        "names_indexed": 2,
    }


def test_v3_rejects_malformed_frequency_cache_records(tmp_path: Path) -> None:
    path = tmp_path / "name-inventory.json"
    valid = {
        "direction": DIRECTION_V3_VERSION,
        "fineweb_docs_scanned": 3,
        "inputs": {},
        "names": [
            {
                "normalized_name": "central",
                "decision": {"document_frequency": 2},
            }
        ],
        "policy": pipeline._policy_record(),
        "status": "complete",
    }
    malformed_records = (
        {**valid, "status": "incomplete"},
        {**valid, "fineweb_docs_scanned": "3"},
        {**valid, "names": "invalid"},
        {**valid, "names": [1]},
        {
            **valid,
            "names": [{"normalized_name": 1, "decision": {}}],
        },
        {
            **valid,
            "names": [{"normalized_name": "central", "decision": "invalid"}],
        },
        {
            **valid,
            "names": [
                {
                    "normalized_name": "central",
                    "decision": {"document_frequency": "2"},
                }
            ],
        },
    )

    for record in malformed_records:
        path.write_text(json.dumps(record), encoding="utf-8")
        assert pipeline._read_cached_frequencies(path, fingerprints={}) is None

    path.write_text(json.dumps(valid), encoding="utf-8")
    assert pipeline._read_cached_frequencies(path, fingerprints={}) == (
        {"central": 2},
        3,
    )


def test_v3_matcher_exposes_unique_patterns_and_empty_matches() -> None:
    profile = _profile("Palais du Prince")
    matcher = V3NameMatcher.build((profile,))

    assert matcher.names_indexed == 1
    assert matcher.find_unique_patterns("No indexed name here.") == ()
    assert matcher.find_unique_patterns("Palais du Prince is here.") == (
        "palais du prince",
    )
    matches = matcher.find("Palais du Prince is here.")
    assert len(matches) == 1
    assert matches[0].start == 0
    assert matches[0].end == len("Palais du Prince")


def test_v3_matcher_does_not_index_discarded_profiles() -> None:
    discarded = _profile("A")
    distinctive = _profile("Palais du Prince")
    matcher = V3NameMatcher.build((discarded, distinctive))

    assert discarded.decision.decision == "discard"
    assert matcher.names_indexed == 1
    matches = matcher.find("A and Palais du Prince are here.")
    assert len(matches) == 1
    assert matches[0].profile.normalized_name == "palais du prince"


def test_v3_scan_document_ignores_documents_without_matches() -> None:
    class Capture:
        def __init__(self) -> None:
            self.rows: list[object] = []

        def add(self, *args: object) -> None:
            self.rows.append(args)

    result = pipeline._ScanResult()
    outputs = Capture()
    pipeline._scan_document(
        "There is no indexed polygon name.",
        "https://example.test/article",
        matcher=V3NameMatcher.build((_profile("Palais du Prince"),)),
        outputs=cast(pipeline._ParquetOutputs, outputs),
        result=result,
    )

    assert outputs.rows == []
    assert result.matches_found == 0


def test_v3_requires_text_and_url_columns(tmp_path: Path) -> None:
    path = tmp_path / "text-only.parquet"
    pq.write_table(pa.table({"text": ["Palais du Prince"]}), path)

    with pytest.raises(
        ValueError,
        match=r"\AFineWeb shard must contain text and url columns; missing url\Z",
    ):
        pipeline._count_document_frequencies(
            path,
            patterns=("palais du prince",),
            batch_size=1,
            log=io.StringIO(),
        )


def test_v3_require_columns_reports_both_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"other": ["value"]}), path)

    with pytest.raises(ValueError) as error:
        pipeline._require_columns(pq.ParquetFile(path))

    assert str(error.value) == (
        "FineWeb shard must contain text and url columns; missing text, url"
    )


def test_v3_frequency_pass_requests_only_text_with_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeParquetFile:
        schema_arrow = SimpleNamespace(names=["text", "url", "extra"])

        def __init__(self, path: Path) -> None:
            self.path = path

        def iter_batches(self, **kwargs: object) -> tuple[object, ...]:
            calls.append(kwargs)
            return ()

    monkeypatch.setattr(pipeline.pq, "ParquetFile", FakeParquetFile)

    assert pipeline._count_document_frequencies(
        Path("unused.parquet"),
        patterns=("palais",),
        batch_size=17,
        log=io.StringIO(),
    ) == ({"palais": 0}, 0)
    assert calls == [{"batch_size": 17, "columns": ["text"], "use_threads": True}]


def test_v3_match_pass_requests_text_and_url_with_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, object]] = []

    class FakeParquetFile:
        schema_arrow = SimpleNamespace(names=["text", "url", "extra"])

        def __init__(self, path: Path) -> None:
            self.path = path

        def iter_batches(self, **kwargs: object) -> tuple[object, ...]:
            calls.append(kwargs)
            return ()

    monkeypatch.setattr(pipeline.pq, "ParquetFile", FakeParquetFile)
    sources = (
        pipeline._V3Source("monaco", tmp_path / "monaco.osm"),
        pipeline._V3Source("liechtenstein", tmp_path / "liechtenstein.osm"),
    )
    paths = tuple(tmp_path / f"{source.key}.parquet" for source in sources)

    result = pipeline._scan_matches(
        Path("unused.parquet"),
        profiles=(),
        sources=sources,
        output_paths=paths,
        batch_size=19,
        output_batch_size=3,
        log=io.StringIO(),
    )

    assert result.documents_scanned == 0
    assert calls == [
        {"batch_size": 19, "columns": ["text", "url"], "use_threads": True}
    ]


def test_v3_scan_document_records_nearby_alias_and_polygon_evidence() -> None:
    polygons = (
        PolygonRecord(
            polygon_id="monaco/way/1",
            source_key="monaco",
            name="Palais du Prince",
            aliases=("Prince's Palace",),
            tags=(),
            centroid=None,
        ),
        PolygonRecord(
            polygon_id="monaco/way/2",
            source_key="monaco",
            name="Central",
            aliases=(),
            tags=(),
            centroid=None,
        ),
    )
    profiles = build_name_inventory(
        polygons,
        document_frequencies={
            "palais du prince": 1,
            "prince s palace": 1,
            "central": 1,
        },
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )

    class Capture:
        def __init__(self) -> None:
            self.rows: list[tuple[str, dict[str, object], str, CandidateEvidence]] = []

        def add(
            self,
            source_key: str,
            row: dict[str, object],
            polygon_id: str,
            evidence: CandidateEvidence,
        ) -> None:
            self.rows.append((source_key, row, polygon_id, evidence))

    outputs = Capture()
    pipeline._scan_document(
        "Palais du Prince is also called Prince's Palace. Central is in Monaco.",
        "https://example.test/article",
        matcher=V3NameMatcher.build(profiles),
        outputs=cast(pipeline._ParquetOutputs, outputs),
        result=pipeline._ScanResult(),
    )

    evidence = [item[3] for item in outputs.rows]
    assert len(evidence) == 3
    assert {item[0] for item in outputs.rows} == {"monaco"}
    assert {item[2] for item in outputs.rows} == {
        "monaco/way/1",
        "monaco/way/2",
    }
    assert all(
        item[1]["fineweb_url"] == "https://example.test/article"
        for item in outputs.rows
    )
    assert any(
        item.same_polygon_alias_nearby and item.other_polygon_name_nearby
        for item in evidence
    )
    palais_evidence = next(
        item[3]
        for item in outputs.rows
        if item[2] == "monaco/way/1" and item[1]["matched_alias"] == "Palais du Prince"
    )
    assert palais_evidence.country_in_context is True


def test_v3_match_row_contains_the_complete_output_contract() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/1",
        source_key="monaco",
        name="Palais du Prince",
        aliases=("Prince's Palace",),
        tags=(("building", "yes"),),
        centroid=(43.738, 7.424),
    )
    profile = _profile("Palais du Prince")
    match = V3NameMatcher.build((profile,)).find("Palais du Prince is visible.")[0]
    match = replace(
        match,
        candidate=replace(match.candidate, polygon=polygon),
    )
    evidence = score_candidate(
        profile,
        alias=match.candidate.alias,
        sentence="Palais du Prince is visible.",
        context="Palais du Prince is visible.",
        url="https://example.test/palais",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )

    row = pipeline._match_row(
        "Palais du Prince is visible.",
        "https://example.test/palais",
        pipeline.split_sentences("Palais du Prince is visible."),
        match,
        evidence,
    )

    assert row == {
        "polygon_id": "monaco/way/1",
        "polygon_name": "Palais du Prince",
        "matched_alias": "Palais du Prince",
        "osm_tags": '{"building":"yes"}',
        "centroid": '{"lat":7.424,"lon":43.738}',
        "fineweb_url": "https://example.test/palais",
        "sentence": "Palais du Prince is visible.",
        "context": "Palais du Prince is visible.",
        "name_match_class": "distinctive_name",
        "osm_polygon_count": 1,
        "fineweb_document_frequency": 1,
        **evidence.to_record(),
    }


def test_v3_match_row_labels_generic_profiles_as_generic() -> None:
    profile = _profile("Central", polygon_count=2, frequency=100)
    match = V3NameMatcher.build((profile,)).find("Central is here.")[0]
    evidence = score_candidate(
        profile,
        alias=match.candidate.alias,
        sentence="Central is here.",
        context="Central is here.",
        url="https://example.test/article",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )

    row = pipeline._match_row(
        "Central is here.",
        "https://example.test/article",
        pipeline.split_sentences("Central is here."),
        match,
        evidence,
    )

    assert row["name_match_class"] == "generic_name"


def test_v3_record_match_counts_a_possible_candidate() -> None:
    profile = _profile("Central", polygon_count=2, frequency=100)
    match = V3NameMatcher.build((profile,)).find("Central")[0]
    evidence = score_candidate(
        profile,
        alias=match.candidate.alias,
        sentence="Central is here.",
        context="Central is here.",
        url="https://example.test/central",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )
    result = pipeline._ScanResult()

    pipeline._record_match(result, match, evidence)

    assert result.matches_found == 1
    assert result.possible_matches == 1
    assert result.rejected_matches == 0


def test_v3_polygon_evidence_distinguishes_aliases_and_other_polygons() -> None:
    polygons = (
        PolygonRecord(
            "monaco/way/1",
            "monaco",
            "Palais du Prince",
            ("Prince's Palace",),
            (),
            None,
        ),
        PolygonRecord("monaco/way/2", "monaco", "Central", (), (), None),
        PolygonRecord(
            "liechtenstein/way/3", "liechtenstein", "Alps View", (), (), None
        ),
    )
    profiles = build_name_inventory(
        polygons,
        document_frequencies={
            "palais du prince": 1,
            "prince s palace": 1,
            "central": 1,
            "alps view": 1,
        },
        document_count=1000,
        country_names=pipeline.COUNTRY_NAMES,
    )
    matches = V3NameMatcher.build(profiles).find(
        "Palais du Prince. Prince's Palace. Central. Alps View."
    )
    current = next(
        match for match in matches if match.candidate.alias == "Palais du Prince"
    )
    alias = next(
        match for match in matches if match.candidate.alias == "Prince's Palace"
    )
    other = next(match for match in matches if match.candidate.alias == "Central")
    foreign = next(match for match in matches if match.candidate.alias == "Alps View")

    assert pipeline._is_same_polygon_alias(current, alias) is True
    assert pipeline._is_same_polygon_alias(current, current) is False
    assert pipeline._is_same_polygon_alias(current, other) is False
    assert pipeline._is_other_polygon_name(current, other) is True
    assert pipeline._is_other_polygon_name(current, current) is False
    assert pipeline._is_other_polygon_name(current, foreign) is False


def test_v3_nearby_predicate_observes_index_and_half_open_context_bounds() -> None:
    match = V3NameMatcher.build((_profile("Palais du Prince"),)).find(
        "Palais du Prince"
    )[0]

    assert pipeline._is_nearby(0, 0, match, 0, 10) is False
    assert pipeline._is_nearby(1, 0, match, 0, 10) is True
    assert pipeline._is_nearby(1, 0, match, 1, 10) is False
    at_end = replace(match, start=10, end=11)
    assert pipeline._is_nearby(1, 0, at_end, 0, 10) is False


def test_v3_nearby_evidence_passes_each_match_position_to_the_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matches = V3NameMatcher.build((_profile("Palais du Prince"),)).find(
        "Palais du Prince and Palais du Prince"
    )
    spans = pipeline.split_sentences("Palais du Prince and Palais du Prince")
    calls: list[tuple[object, object]] = []
    original = pipeline._is_nearby

    def spy_is_nearby(
        index: int,
        current_index: int,
        candidate: V3NameMatch,
        context_start: int,
        context_end: int,
    ) -> bool:
        calls.append((index, current_index))
        return original(index, current_index, candidate, context_start, context_end)

    monkeypatch.setattr(pipeline, "_is_nearby", spy_is_nearby)

    pipeline._nearby_evidence(spans, matches, 0)

    assert calls == [(0, 0), (1, 0)]


def test_v3_text_conversion_and_context_bounds_cover_edges() -> None:
    assert pipeline._as_text(None) == ""
    assert pipeline._as_text(42) == "42"
    text = "First. Palais du Prince. Last."
    spans = pipeline.split_sentences(text)
    assert pipeline._context_bounds(spans, spans[0].start) == (
        spans[0].start,
        spans[1].end,
    )
    assert pipeline._context_bounds(spans, spans[2].start) == (
        spans[1].start,
        spans[2].end,
    )


def test_v3_profiles_for_source_filters_mixed_profiles() -> None:
    polygons = (
        PolygonRecord("monaco/way/1", "monaco", "Monaco Hall", (), (), None),
        PolygonRecord(
            "liechtenstein/way/2", "liechtenstein", "Vaduz Hall", (), (), None
        ),
        PolygonRecord(
            "liechtenstein/way/3", "liechtenstein", "Alps View", (), (), None
        ),
    )
    profiles = build_name_inventory(
        polygons,
        document_frequencies={
            "monaco hall": 1,
            "vaduz hall": 1,
            "alps view": 1,
        },
        document_count=1000,
        country_names=pipeline.COUNTRY_NAMES,
    )

    selected = pipeline._profiles_for_source(profiles, "monaco")

    assert [profile.normalized_name for profile in selected] == ["monaco hall"]


def test_v3_country_summary_preserves_counts_and_digest(tmp_path: Path) -> None:
    output = tmp_path / "result.parquet"
    output.write_bytes(b"result")
    source = pipeline._V3Source("monaco", tmp_path / "monaco.osm")
    polygons = (
        PolygonRecord("monaco/way/1", "monaco", "One", (), (), None),
        PolygonRecord("monaco/way/2", "monaco", "Two", (), (), None),
        PolygonRecord("liechtenstein/way/3", "liechtenstein", "Three", (), (), None),
    )
    profiles = (_profile("Palais du Prince"), _profile("A"), _profile("B"))
    stats = pipeline._CountryStats(polygon_ids={"monaco/way/1"})

    summary = pipeline._country_summary(
        source,
        output,
        polygons=polygons,
        profiles=profiles,
        stats=stats,
    )

    assert summary.output_path == output
    assert summary.polygons_read == 2
    assert summary.names_indexed == 1
    assert summary.unique_polygons_matched == 1
    assert summary.result_sha256 == pipeline.sha256_file(output)


def test_v3_country_summaries_reject_mismatched_sequences(tmp_path: Path) -> None:
    output = tmp_path / "result.parquet"
    output.write_bytes(b"result")
    sources = (
        pipeline._V3Source("monaco", tmp_path / "monaco.osm"),
        pipeline._V3Source("liechtenstein", tmp_path / "liechtenstein.osm"),
    )

    with pytest.raises(ValueError):
        pipeline._country_summaries(
            sources,
            (output,),
            polygons=(),
            profiles=(),
            country_stats={"monaco": pipeline._CountryStats()},
        )


def test_v3_summary_preserves_artifact_paths_and_discard_count(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = pipeline._summary(
        config=config,
        output_paths=(tmp_path / "monaco.parquet",),
        profiles=(_profile("A"), _profile("Palais du Prince")),
        polygons_read=2,
        frequency=pipeline._FrequencyResult({}, 3, False),
        scan=pipeline._ScanResult(),
        country_summaries=(),
    )

    assert summary.manifest_path == config.manifest_path
    assert summary.dataset_card_path == config.dataset_card_path
    assert summary.log_path == config.log_path
    assert summary.name_inventory_path == config.name_inventory_path
    assert summary.polygons_read == 2
    assert summary.names_discarded == 1


def test_v3_manifest_rejects_mismatched_country_summaries(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.name_inventory_path.parent.mkdir(parents=True)
    config.name_inventory_path.write_text("{}", encoding="utf-8")
    source = pipeline._V3Source("monaco", tmp_path / "monaco.osm")

    with pytest.raises(ValueError):
        pipeline._manifest(
            config=config,
            sources=(source,),
            fingerprints={},
            profiles=(),
            polygons_read=0,
            frequency=pipeline._FrequencyResult({}, 0, False),
            scan=pipeline._ScanResult(),
            country_summaries=(),
        )


@pytest.mark.parametrize(
    ("section", "manifest"),
    [
        ("polygon_inventory", {"polygon_inventory": []}),
        ("results", {"polygon_inventory": {}, "results": []}),
        (
            "countries",
            {"polygon_inventory": {}, "results": {}, "countries": []},
        ),
        (
            "configuration",
            {
                "polygon_inventory": {},
                "results": {},
                "countries": {},
                "configuration": [],
            },
        ),
    ],
)
def test_v3_card_rejects_non_object_manifest_sections(
    section: str, manifest: dict[str, object]
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"\Amanifest field '{section}' must be an object\Z",
    ):
        render_dataset_card(manifest)


def test_v3_card_rejects_non_object_country_summary() -> None:
    with pytest.raises(
        ValueError,
        match=r"\Amanifest field 'country' must be an object\Z",
    ):
        render_dataset_card(
            {
                "polygon_inventory": {},
                "results": {},
                "countries": {"monaco": []},
                "configuration": {},
            }
        )


def _output_row() -> dict[str, object]:
    return {
        field.name: (
            False
            if pa.types.is_boolean(field.type)
            else 0
            if pa.types.is_integer(field.type)
            else ""
        )
        for field in pipeline._OUTPUT_SCHEMA
    }


def test_v3_parquet_state_flushes_and_aborts_atomically(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "deeper" / "result.parquet"
    state = pipeline._ParquetState(path, batch_size=1)
    assert state.writer is None
    state.open()
    state.add(_output_row(), "monaco/way/1", "rejected")
    assert state.rows == []
    state.add(_output_row(), "monaco/way/2", "possible")
    state.add(_output_row(), "monaco/way/3", "high_confidence")
    state.add(_output_row(), "monaco/way/4", "high_confidence")
    state.flush()
    state.publish()

    assert pq.read_table(path).num_rows == 4
    assert state.stats.matches_found == 4
    assert state.stats.high_confidence_matches == 2
    assert state.stats.possible_matches == 1
    assert state.stats.rejected_matches == 1
    assert state.stats.polygon_ids == {
        "monaco/way/1",
        "monaco/way/2",
        "monaco/way/3",
        "monaco/way/4",
    }

    aborted = pipeline._ParquetState(tmp_path / "aborted.parquet", batch_size=2)
    aborted.open()
    temporary = aborted.temporary
    aborted.abort()
    assert not temporary.exists()


def test_v3_parquet_outputs_abort_all_states_on_error(tmp_path: Path) -> None:
    paths = (tmp_path / "monaco.parquet", tmp_path / "liechtenstein.parquet")

    with (
        pytest.raises(RuntimeError, match="stop"),
        pipeline._ParquetOutputs(("monaco", "liechtenstein"), paths, 1),
    ):
        raise RuntimeError("stop")

    assert all(not path.exists() for path in paths)


def test_v3_parquet_outputs_reject_mismatched_source_and_output_counts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        pipeline._ParquetOutputs(
            ("monaco",),
            (tmp_path / "one.parquet", tmp_path / "two.parquet"),
            batch_size=2,
        )


def test_v3_parquet_outputs_add_preserves_polygon_id(tmp_path: Path) -> None:
    profile = _profile("Palais du Prince")
    evidence = score_candidate(
        profile,
        alias="Palais du Prince",
        sentence="Palais du Prince is here.",
        context="Palais du Prince is here.",
        url="https://example.test/article",
        country_name="Monaco",
        same_polygon_alias_nearby=False,
        other_polygon_name_nearby=False,
    )
    path = tmp_path / "monaco.parquet"

    with pipeline._ParquetOutputs(("monaco",), (path,), 1) as outputs:
        outputs.add("monaco", _output_row(), "monaco/way/1", evidence)
        assert outputs.stats("monaco").polygon_ids == {"monaco/way/1"}


def test_v3_parquet_state_requests_zstd_compression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, object] = {}

    class FakeWriter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["compression"] = kwargs["compression"]

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline.pq, "ParquetWriter", FakeWriter)
    state = pipeline._ParquetState(tmp_path / "result.parquet", batch_size=2)

    state.open()

    assert calls["compression"] == "zstd"


def test_v3_parquet_state_abort_closes_writer_and_allows_missing_temp(
    tmp_path: Path,
) -> None:
    class FakeWriter:
        closed = False

        def close(self) -> None:
            self.closed = True

    state = pipeline._ParquetState(tmp_path / "result.parquet", batch_size=2)
    writer = FakeWriter()
    state.writer = writer
    state.temporary.parent.mkdir(parents=True, exist_ok=True)
    state.temporary.unlink(missing_ok=True)

    state.abort()

    assert writer.closed is True


def test_v3_write_card_uses_the_deterministic_temporary_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_v3_fixture(tmp_path)
    run_direction2_v3(config)
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    factories: list[object] = []
    original_atomic_text_output = pipeline.atomic_text_output

    def spy_atomic_text_output(path: Path, **kwargs: Any) -> Any:
        factories.append(kwargs.get("temporary_factory"))
        return original_atomic_text_output(path, **kwargs)

    monkeypatch.setattr(pipeline, "atomic_text_output", spy_atomic_text_output)
    pipeline._write_card(tmp_path / "copy.md", manifest)

    assert factories == [pipeline.deterministic_temporary_path]
