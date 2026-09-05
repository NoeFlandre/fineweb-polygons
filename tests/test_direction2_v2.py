import io
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fineweb_polygons.direction2.v2_pipeline as v2_pipeline
import fineweb_polygons.direction2.v2_specificity as v2_specificity
from fineweb_polygons.artifact_io import sha256_file
from fineweb_polygons.direction2.matching import (
    AhoCorasickPatternMatcher,
    PatternMatch,
)
from fineweb_polygons.direction2.models import PolygonRecord
from fineweb_polygons.direction2.v2_card import render_dataset_card
from fineweb_polygons.direction2.v2_matching import (
    V2NameMatcher,
    has_independent_country_match,
)
from fineweb_polygons.direction2.v2_models import (
    DIRECTION_V2_VERSION,
    OUTPUT_COLUMNS_V2,
    Direction2V2RunConfig,
)
from fineweb_polygons.direction2.v2_pipeline import run_direction2_v2
from fineweb_polygons.direction2.v2_specificity import (
    build_name_inventory,
    classify_name,
    fineweb_frequency_cutoff,
    searchable_name_patterns,
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


def _make_v2_fixture(tmp_path: Path) -> tuple[Direction2V2RunConfig, Path]:
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    _write_osm(
        monaco,
        (
            ("Palais du Prince", "Prince's Palace"),
            ("Central", ""),
            ("Monaco", ""),
        ),
    )
    _write_osm(
        liechtenstein,
        (
            ("Alps View", ""),
            ("Central", ""),
            ("Liechtenstein", ""),
        ),
    )
    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [
                    "The Palais du Prince is visible. "
                    "Central appears without a country.",
                    "Central in Monaco is a place.",
                    "Central in Liechtenstein is a place.",
                    "Alps View is visible. Central is here.",
                ],
                "url": ["one", "two", "three", "four"],
            }
        ),
        shard,
    )
    config = Direction2V2RunConfig(
        monaco_pbf=monaco,
        liechtenstein_pbf=liechtenstein,
        shard_path=shard,
        output_dir=tmp_path / "artifacts",
        manifest_path=tmp_path / "runs" / "manifest.json",
        dataset_card_path=tmp_path / "card.md",
        log_path=tmp_path / "logs" / "run.jsonl",
        name_inventory_path=tmp_path / "runs" / "name-inventory.json",
    )
    return config, shard


def test_name_policy_rejects_short_and_numeric_names() -> None:
    short = classify_name(
        "A",
        polygon_count=1,
        document_frequency=0,
        document_count=1000,
    )
    numeric = classify_name(
        "2",
        polygon_count=1,
        document_frequency=0,
        document_count=1000,
    )

    assert short.decision == "discard"
    assert short.reason == "too_few_letters"
    assert numeric.decision == "discard"
    assert numeric.reason == "no_letters"


def test_name_policy_keeps_three_letters_but_marks_eight_generic() -> None:
    three_letters = classify_name(
        "Cat",
        polygon_count=1,
        document_frequency=0,
        document_count=1000,
    )
    eight_letters = classify_name(
        "Mountain",
        polygon_count=1,
        document_frequency=0,
        document_count=1000,
    )

    assert three_letters.decision == "generic"
    assert three_letters.reason == "short_single_token"
    assert eight_letters.decision == "generic"
    assert eight_letters.reason == "short_single_token"


def test_name_policy_marks_short_single_token_names_generic() -> None:
    result = classify_name(
        "Central",
        polygon_count=1,
        document_frequency=1,
        document_count=1000,
    )

    assert result.decision == "generic"
    assert result.reason == "short_single_token"


def test_name_policy_marks_reused_names_generic() -> None:
    result = classify_name(
        "Old Mill",
        polygon_count=2,
        document_frequency=1,
        document_count=1000,
    )

    assert result.decision == "generic"
    assert result.reason == "osm_reuse"


def test_name_policy_marks_frequent_names_generic() -> None:
    result = classify_name(
        "Distinctive Hall",
        polygon_count=1,
        document_frequency=11,
        document_count=1000,
    )

    assert result.decision == "generic"
    assert result.reason == "fineweb_frequency"


def test_name_policy_keeps_rare_specific_names() -> None:
    result = classify_name(
        "Palais du Prince",
        polygon_count=1,
        document_frequency=1,
        document_count=1000,
    )

    assert result.decision == "distinctive"


def test_name_policy_marks_source_country_name_non_indexable() -> None:
    result = classify_name(
        "Monaco",
        polygon_count=1,
        document_frequency=1,
        document_count=1000,
        country_name="Monaco",
    )

    assert result.decision == "discard"
    assert result.reason == "country_name"


def test_name_policy_does_not_discard_a_name_for_a_different_country() -> None:
    result = classify_name(
        "Monaco",
        polygon_count=1,
        document_frequency=0,
        document_count=1000,
        country_name="France",
    )

    assert result.decision == "generic"
    assert result.reason == "short_single_token"


def test_name_policy_does_not_decode_percent_escapes_in_names_or_countries() -> None:
    result = classify_name(
        "Foo%20Bar",
        polygon_count=1,
        document_frequency=0,
        document_count=1000,
        country_name="Foo Bar",
    )

    assert result.decision == "distinctive"
    assert result.reason == "specific"


def test_name_grouping_explicitly_disables_url_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = v2_specificity.normalize_for_search

    def spy(value: object, *, decode_url: bool = True) -> str:
        calls.append(decode_url)
        return original(value, decode_url=decode_url)

    monkeypatch.setattr(v2_specificity, "normalize_for_search", spy)
    polygon = PolygonRecord("monaco/way/1", "monaco", "A name", (), (), None)

    v2_specificity._group_candidates((polygon,))

    assert calls == [False]


def test_country_normalization_explicitly_disables_url_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = v2_specificity.normalize_for_search

    def spy(value: object, *, decode_url: bool = True) -> str:
        calls.append(decode_url)
        return original(value, decode_url=decode_url)

    monkeypatch.setattr(v2_specificity, "normalize_for_search", spy)

    v2_specificity._normalized_countries({"monaco": "Monaco"})

    assert calls == [False]


def test_name_classification_explicitly_disables_url_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = v2_specificity.normalize_for_search

    def spy(value: object, *, decode_url: bool = True) -> str:
        calls.append(decode_url)
        return original(value, decode_url=decode_url)

    monkeypatch.setattr(v2_specificity, "normalize_for_search", spy)

    v2_specificity.classify_name(
        "A named place",
        polygon_count=1,
        document_frequency=0,
        document_count=100,
    )

    assert calls == [False]


def test_discard_check_explicitly_disables_url_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []
    original = v2_specificity.normalize_for_search

    def spy(value: object, *, decode_url: bool = True) -> str:
        calls.append(decode_url)
        return original(value, decode_url=decode_url)

    monkeypatch.setattr(v2_specificity, "normalize_for_search", spy)

    v2_specificity._discard_reason(
        "a named place",
        letter_count=12,
        country_name="Monaco",
    )

    assert calls == [False]


def test_fineweb_frequency_cutoff_has_stable_boundaries() -> None:
    assert fineweb_frequency_cutoff(0) == 0
    assert fineweb_frequency_cutoff(1000) == 1
    assert fineweb_frequency_cutoff(2000) == 2
    with pytest.raises(ValueError) as error:
        fineweb_frequency_cutoff(-1)
    assert str(error.value) == "document_count must be non-negative"


def test_name_policy_validates_each_count_with_a_stable_error() -> None:
    for field, values in (
        ("polygon_count", (-1, 0, 1000)),
        ("document_frequency", (1, -1, 1000)),
        ("document_count", (1, 0, -1)),
    ):
        with pytest.raises(ValueError) as error:
            classify_name(
                "A name",
                polygon_count=values[0],
                document_frequency=values[1],
                document_count=values[2],
            )
        assert str(error.value) == f"{field} must be non-negative"


def test_searchable_name_patterns_excludes_unsearchable_decisions() -> None:
    polygons = tuple(
        PolygonRecord(
            polygon_id=f"monaco/way/{index}",
            source_key="monaco",
            name=name,
            aliases=(),
            tags=(),
            centroid=None,
        )
        for index, name in enumerate(("A", "2", "Alps View"), start=1)
    )

    profiles = build_name_inventory(
        polygons,
        document_frequencies={},
        document_count=0,
        country_names={},
    )

    assert searchable_name_patterns(profiles) == ("alps view",)


def test_name_inventory_uses_zero_for_unseen_frequencies() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Grand Palace",
        aliases=(),
        tags=(),
        centroid=None,
    )

    profile = build_name_inventory(
        (polygon,),
        document_frequencies={},
        document_count=0,
        country_names={},
    )[0]

    assert profile.decision.document_frequency == 0
    assert profile.decision.decision == "distinctive"


def test_name_inventory_does_not_decode_percent_escapes() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="X%20Y",
        aliases=(),
        tags=(),
        centroid=None,
    )

    profile = build_name_inventory(
        (polygon,),
        document_frequencies={},
        document_count=0,
        country_names={},
    )[0]

    assert profile.normalized_name == "x 20y"


def test_name_inventory_uses_country_names_without_url_decoding() -> None:
    polygon = PolygonRecord(
        polygon_id="test/way/10",
        source_key="test",
        name="Foo 20bar",
        aliases=(),
        tags=(),
        centroid=None,
    )

    profile = build_name_inventory(
        (polygon,),
        document_frequencies={},
        document_count=0,
        country_names={"test": "Foo%20Bar"},
    )[0]

    assert profile.decision.decision == "discard"


def test_name_inventory_groups_aliases_and_counts_distinct_polygons() -> None:
    polygons = (
        PolygonRecord(
            polygon_id="monaco/way/2",
            source_key="monaco",
            name="Central",
            aliases=("Centre", "Central"),
            tags=(),
            centroid=None,
        ),
        PolygonRecord(
            polygon_id="monaco/way/1",
            source_key="monaco",
            name="Old Mill",
            aliases=("Vieux Moulin",),
            tags=(),
            centroid=None,
        ),
        PolygonRecord(
            polygon_id="liechtenstein/way/3",
            source_key="liechtenstein",
            name="Central",
            aliases=(),
            tags=(),
            centroid=None,
        ),
    )

    inventory = build_name_inventory(
        polygons,
        document_frequencies={
            "central": 1,
            "centre": 1,
            "old mill": 1,
            "vieux moulin": 1,
        },
        document_count=1000,
        country_names={"monaco": "Monaco", "liechtenstein": "Liechtenstein"},
    )

    assert [profile.normalized_name for profile in inventory] == [
        "central",
        "centre",
        "old mill",
        "vieux moulin",
    ]
    central = inventory[0]
    assert central.osm_polygon_count == 2
    assert [(item.polygon_id, item.alias) for item in central.candidates] == [
        ("liechtenstein/way/3", "Central"),
        ("monaco/way/2", "Central"),
    ]
    assert central.decision.decision == "generic"
    assert central.decision.reason == "osm_reuse"


def test_name_inventory_is_deterministic_and_excludes_source_country_names() -> None:
    polygons = (
        PolygonRecord(
            polygon_id="monaco/relation/9",
            source_key="monaco",
            name="Monaco",
            aliases=("Monaco",),
            tags=(),
            centroid=None,
        ),
        PolygonRecord(
            polygon_id="monaco/way/10",
            source_key="monaco",
            name="Palais du Prince",
            aliases=("Prince's Palace",),
            tags=(),
            centroid=None,
        ),
    )

    inventory = build_name_inventory(
        tuple(reversed(polygons)),
        document_frequencies={
            "monaco": 50,
            "palais du prince": 1,
            "prince's palace": 1,
        },
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )

    assert [profile.normalized_name for profile in inventory] == [
        "monaco",
        "palais du prince",
        "prince s palace",
    ]
    assert inventory[0].decision.reason == "country_name"
    assert inventory[0].decision.decision == "discard"


def test_pattern_matcher_returns_each_normalized_pattern_once_per_document() -> None:
    matcher = AhoCorasickPatternMatcher.build(("Old Mill", "Palais"))

    assert matcher.find_unique_patterns(
        "Old mill is here. Old Mill again. Palaisage is not one. Palais."
    ) == ("old mill", "palais")


def test_pattern_matcher_preserves_offsets_after_casefold_expansion() -> None:
    matcher = AhoCorasickPatternMatcher.build(("ss",))

    matches = matcher.find("ß is a match.")

    assert [(match.pattern, match.start, match.end) for match in matches] == [
        ("ss", 0, 1)
    ]


def test_pattern_matcher_handles_empty_inputs() -> None:
    matcher = AhoCorasickPatternMatcher.build(("", "   "))

    assert matcher.patterns_indexed == 0
    assert matcher.find("") == ()
    assert matcher.find_unique_patterns("anything") == ()


def test_pattern_matcher_retries_when_combining_marks_change_normalized_text() -> None:
    matcher = AhoCorasickPatternMatcher.build(("e",))

    matches = matcher.find("e\u0301")

    assert [(match.pattern, match.start, match.end) for match in matches] == [
        ("e", 0, 1)
    ]


def test_v2_matcher_returns_candidates_for_accepted_name_profiles() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais du Prince",
        aliases=(),
        tags=(),
        centroid=None,
    )
    profiles = build_name_inventory(
        (polygon,),
        document_frequencies={"palais du prince": 1},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )

    matches = V2NameMatcher.build(profiles).find("Palais du Prince is visible.")

    assert len(matches) == 1
    assert matches[0].candidate.alias == "Palais du Prince"
    assert matches[0].profile.decision.decision == "distinctive"
    assert (matches[0].start, matches[0].end) == (0, 16)


def test_v2_matcher_excludes_discarded_country_profiles() -> None:
    polygons = (
        PolygonRecord(
            polygon_id="monaco/way/10",
            source_key="monaco",
            name="Palais du Prince",
            aliases=(),
            tags=(),
            centroid=None,
        ),
        PolygonRecord(
            polygon_id="monaco/way/11",
            source_key="monaco",
            name="Monaco",
            aliases=(),
            tags=(),
            centroid=None,
        ),
    )
    profiles = build_name_inventory(
        polygons,
        document_frequencies={"palais du prince": 1, "monaco": 1},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )

    matcher = V2NameMatcher.build(profiles)

    assert matcher.names_indexed == 1
    assert matcher.find("Monaco and Palais du Prince")
    assert matcher.find("Monaco and Palais du Prince")[0].candidate.alias == (
        "Palais du Prince"
    )


def test_v2_matcher_exposes_document_level_unique_patterns() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais du Prince",
        aliases=(),
        tags=(),
        centroid=None,
    )
    profiles = build_name_inventory(
        (polygon,),
        document_frequencies={"palais du prince": 1},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )

    assert V2NameMatcher.build(profiles).find_unique_patterns(
        "Palais du Prince appears twice: Palais du Prince."
    ) == ("palais du prince",)


def test_country_evidence_must_not_overlap_the_generic_name() -> None:
    country_matcher = AhoCorasickPatternMatcher.build(("Monaco",))

    assert not has_independent_country_match(
        country_matcher.find("Monaco"),
        name_start=0,
        name_end=6,
    )
    assert has_independent_country_match(
        country_matcher.find("Central in Monaco"),
        name_start=0,
        name_end=7,
    )


def test_country_evidence_treats_touching_spans_as_independent() -> None:
    country_match = PatternMatch(pattern="monaco", start=0, end=6)

    assert has_independent_country_match(
        (country_match,),
        name_start=6,
        name_end=13,
    )
    assert has_independent_country_match(
        (PatternMatch(pattern="monaco", start=7, end=13),),
        name_start=0,
        name_end=7,
    )


def test_run_direction2_v2_counts_and_gates_specificity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    config = replace(config, log_path=tmp_path / "logs" / "résultats.jsonl")
    original_open = Path.open

    def open_with_ascii_locale(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == config.log_path:
            kwargs["encoding"] = kwargs.get("encoding") or "ascii"
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_with_ascii_locale)
    summary = run_direction2_v2(config)

    assert summary.output_paths == (
        config.output_dir / "monaco.parquet",
        config.output_dir / "liechtenstein.parquet",
    )
    assert summary.manifest_path == config.manifest_path
    assert summary.dataset_card_path == config.dataset_card_path
    assert summary.log_path == config.log_path
    assert summary.name_inventory_path == config.name_inventory_path
    assert summary.polygons_read == 6
    assert summary.names_considered == 6
    assert summary.names_indexed == 4
    assert summary.direction == DIRECTION_V2_VERSION
    assert summary.fineweb_docs_frequency_pass == 4
    assert summary.fineweb_docs_match_pass == 4
    assert summary.matches_found == 4
    assert summary.unique_polygons_matched == 4
    assert summary.generic_matches == 2
    assert summary.distinctive_matches == 2
    assert summary.names_discarded == 2
    assert summary.generic_names == 1

    monaco_rows = pq.read_table(config.output_dir / "monaco.parquet")
    liechtenstein_rows = pq.read_table(config.output_dir / "liechtenstein.parquet")
    assert monaco_rows.column_names == list(OUTPUT_COLUMNS_V2)
    assert liechtenstein_rows.column_names == list(OUTPUT_COLUMNS_V2)
    assert monaco_rows["matched_alias"].to_pylist() == [
        "Palais du Prince",
        "Central",
    ]
    assert liechtenstein_rows["matched_alias"].to_pylist() == [
        "Central",
        "Alps View",
    ]
    assert monaco_rows["name_match_class"].to_pylist() == [
        "distinctive_name",
        "generic_name_with_country",
    ]
    assert liechtenstein_rows["name_match_class"].to_pylist() == [
        "generic_name_with_country",
        "distinctive_name",
    ]
    assert monaco_rows["fineweb_document_frequency"].to_pylist() == [1, 4]
    assert liechtenstein_rows["fineweb_document_frequency"].to_pylist() == [4, 1]
    assert monaco_rows.to_pylist() == [
        {
            "polygon_id": "monaco/way/10",
            "polygon_name": "Palais du Prince",
            "matched_alias": "Palais du Prince",
            "osm_tags": '{"name":"Palais du Prince","name:en":"Prince\'s Palace"}',
            "centroid": '{"lat":43.704999995373186,"lon":7.404999999215664}',
            "fineweb_url": "one",
            "sentence": "The Palais du Prince is visible.",
            "context": (
                "The Palais du Prince is visible. Central appears without a country."
            ),
            "name_match_class": "distinctive_name",
            "osm_polygon_count": 1,
            "fineweb_document_frequency": 1,
        },
        {
            "polygon_id": "monaco/way/11",
            "polygon_name": "Central",
            "matched_alias": "Central",
            "osm_tags": '{"name":"Central"}',
            "centroid": '{"lat":43.704999995373186,"lon":7.404999999215664}',
            "fineweb_url": "two",
            "sentence": "Central in Monaco is a place.",
            "context": "Central in Monaco is a place.",
            "name_match_class": "generic_name_with_country",
            "osm_polygon_count": 2,
            "fineweb_document_frequency": 4,
        },
    ]
    assert liechtenstein_rows.to_pylist() == [
        {
            "polygon_id": "liechtenstein/way/11",
            "polygon_name": "Central",
            "matched_alias": "Central",
            "osm_tags": '{"name":"Central"}',
            "centroid": '{"lat":43.704999995373186,"lon":7.404999999215664}',
            "fineweb_url": "three",
            "sentence": "Central in Liechtenstein is a place.",
            "context": "Central in Liechtenstein is a place.",
            "name_match_class": "generic_name_with_country",
            "osm_polygon_count": 2,
            "fineweb_document_frequency": 4,
        },
        {
            "polygon_id": "liechtenstein/way/10",
            "polygon_name": "Alps View",
            "matched_alias": "Alps View",
            "osm_tags": '{"name":"Alps View"}',
            "centroid": '{"lat":43.704999995373186,"lon":7.404999999215664}',
            "fineweb_url": "four",
            "sentence": "Alps View is visible.",
            "context": "Alps View is visible. Central is here.",
            "name_match_class": "distinctive_name",
            "osm_polygon_count": 1,
            "fineweb_document_frequency": 1,
        },
    ]

    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["direction"] == DIRECTION_V2_VERSION
    assert manifest["results"]["matches_found"] == 4
    assert (
        json.loads(config.name_inventory_path.read_text(encoding="utf-8"))["status"]
        == "complete"
    )
    card = config.dataset_card_path.read_text(encoding="utf-8")
    assert "generic-name noise" in card
    assert "exact source country name" in card
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
    assert events[0] == {"event": "run_started", "version": DIRECTION_V2_VERSION}
    assert events[1]["docs_scanned"] == 4
    assert events[2]["names_considered"] == 6
    assert events[2]["names_discarded"] == 2
    assert events[2]["names_indexed"] == 4
    assert events[2]["frequency_pass_reused"] is False
    assert events[3]["docs_scanned"] == 4
    assert events[4]["matches_found"] == 4
    assert manifest == {
        "configuration": {
            **v2_pipeline._policy_record(),
            "batch_size": 8192,
            "frequency_pass_reused": False,
            "matcher": "Aho-Corasick",
            "output_batch_size": 4096,
            "sentence_context": "matching sentence plus one sentence on each side",
            "url_is_condition": False,
        },
        "countries": {
            "liechtenstein": {
                "distinctive_matches": 1,
                "generic_matches": 1,
                "matches_found": 2,
                "names_indexed": 2,
                "output_path": str(config.output_dir / "liechtenstein.parquet"),
                "polygons_read": 3,
                "result_sha256": sha256_file(
                    config.output_dir / "liechtenstein.parquet"
                ),
                "unique_polygons_matched": 2,
            },
            "monaco": {
                "distinctive_matches": 1,
                "generic_matches": 1,
                "matches_found": 2,
                "names_indexed": 3,
                "output_path": str(config.output_dir / "monaco.parquet"),
                "polygons_read": 3,
                "result_sha256": sha256_file(config.output_dir / "monaco.parquet"),
                "unique_polygons_matched": 2,
            },
        },
        "direction": DIRECTION_V2_VERSION,
        "name_inventory": {
            "path": str(config.name_inventory_path),
            "sha256": sha256_file(config.name_inventory_path),
        },
        "polygon_inventory": {
            "generic_names": 1,
            "names_considered": 6,
            "names_discarded": 2,
            "names_indexed": 4,
            "polygons_read": 6,
        },
        "results": {
            "files": [
                {
                    "path": "data/direction-2/lexical-v2/monaco.parquet",
                    "sha256": sha256_file(config.output_dir / "monaco.parquet"),
                    "source_key": "monaco",
                },
                {
                    "path": "data/direction-2/lexical-v2/liechtenstein.parquet",
                    "sha256": sha256_file(config.output_dir / "liechtenstein.parquet"),
                    "source_key": "liechtenstein",
                },
            ],
            "fineweb_docs_frequency_pass": 4,
            "fineweb_docs_match_pass": 4,
            "generic_matches": 2,
            "matches_found": 4,
            "distinctive_matches": 2,
            "unique_polygons_matched": 4,
        },
        "schema": list(OUTPUT_COLUMNS_V2),
        "sources": {
            "fineweb_shard": {
                "path": str(config.shard_path),
                "sha256": sha256_file(config.shard_path),
            },
            "osm_pbf": [
                {
                    "path": str(config.monaco_pbf),
                    "sha256": sha256_file(config.monaco_pbf),
                    "source_key": "monaco",
                },
                {
                    "path": str(config.liechtenstein_pbf),
                    "sha256": sha256_file(config.liechtenstein_pbf),
                    "source_key": "liechtenstein",
                },
            ],
        },
        "status": "complete",
    }
    assert json.loads(config.name_inventory_path.read_text(encoding="utf-8")) == {
        "direction": DIRECTION_V2_VERSION,
        "fineweb_docs_scanned": 4,
        "inputs": manifest["sources"],
        "names": [
            {
                "candidates": [
                    {
                        "alias": "Alps View",
                        "normalized_name": "alps view",
                        "polygon_id": "liechtenstein/way/10",
                        "source_key": "liechtenstein",
                    }
                ],
                "decision": {
                    "decision": "distinctive",
                    "document_frequency": 1,
                    "frequency_cutoff": 1,
                    "letter_count": 8,
                    "normalized_name": "alps view",
                    "polygon_count": 1,
                    "reason": "specific",
                    "token_count": 2,
                },
                "normalized_name": "alps view",
                "osm_polygon_count": 1,
            },
            {
                "candidates": [
                    {
                        "alias": "Central",
                        "normalized_name": "central",
                        "polygon_id": "liechtenstein/way/11",
                        "source_key": "liechtenstein",
                    },
                    {
                        "alias": "Central",
                        "normalized_name": "central",
                        "polygon_id": "monaco/way/11",
                        "source_key": "monaco",
                    },
                ],
                "decision": {
                    "decision": "generic",
                    "document_frequency": 4,
                    "frequency_cutoff": 1,
                    "letter_count": 7,
                    "normalized_name": "central",
                    "polygon_count": 2,
                    "reason": "osm_reuse",
                    "token_count": 1,
                },
                "normalized_name": "central",
                "osm_polygon_count": 2,
            },
            {
                "candidates": [
                    {
                        "alias": "Liechtenstein",
                        "normalized_name": "liechtenstein",
                        "polygon_id": "liechtenstein/way/12",
                        "source_key": "liechtenstein",
                    }
                ],
                "decision": {
                    "decision": "discard",
                    "document_frequency": 1,
                    "frequency_cutoff": 1,
                    "letter_count": 13,
                    "normalized_name": "liechtenstein",
                    "polygon_count": 1,
                    "reason": "country_name",
                    "token_count": 1,
                },
                "normalized_name": "liechtenstein",
                "osm_polygon_count": 1,
            },
            {
                "candidates": [
                    {
                        "alias": "Monaco",
                        "normalized_name": "monaco",
                        "polygon_id": "monaco/way/12",
                        "source_key": "monaco",
                    }
                ],
                "decision": {
                    "decision": "discard",
                    "document_frequency": 1,
                    "frequency_cutoff": 1,
                    "letter_count": 6,
                    "normalized_name": "monaco",
                    "polygon_count": 1,
                    "reason": "country_name",
                    "token_count": 1,
                },
                "normalized_name": "monaco",
                "osm_polygon_count": 1,
            },
            {
                "candidates": [
                    {
                        "alias": "Palais du Prince",
                        "normalized_name": "palais du prince",
                        "polygon_id": "monaco/way/10",
                        "source_key": "monaco",
                    }
                ],
                "decision": {
                    "decision": "distinctive",
                    "document_frequency": 1,
                    "frequency_cutoff": 1,
                    "letter_count": 14,
                    "normalized_name": "palais du prince",
                    "polygon_count": 1,
                    "reason": "specific",
                    "token_count": 3,
                },
                "normalized_name": "palais du prince",
                "osm_polygon_count": 1,
            },
            {
                "candidates": [
                    {
                        "alias": "Prince's Palace",
                        "normalized_name": "prince s palace",
                        "polygon_id": "monaco/way/10",
                        "source_key": "monaco",
                    }
                ],
                "decision": {
                    "decision": "distinctive",
                    "document_frequency": 0,
                    "frequency_cutoff": 1,
                    "letter_count": 13,
                    "normalized_name": "prince s palace",
                    "polygon_count": 1,
                    "reason": "specific",
                    "token_count": 3,
                },
                "normalized_name": "prince s palace",
                "osm_polygon_count": 1,
            },
        ],
        "policy": v2_pipeline._policy_record(),
        "status": "complete",
        "summary": {
            "generic_names": 1,
            "names_considered": 6,
            "names_discarded": 2,
            "names_indexed": 4,
        },
    }


def test_run_direction2_v2_starts_name_inventory_before_frequency_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    document_counts: list[int] = []
    original = v2_pipeline.build_name_inventory

    def spy_build_name_inventory(*args: Any, **kwargs: Any) -> Any:
        document_counts.append(kwargs["document_count"])
        return original(*args, **kwargs)

    monkeypatch.setattr(v2_pipeline, "build_name_inventory", spy_build_name_inventory)

    run_direction2_v2(config)

    assert document_counts == [0, 4]


def test_run_direction2_v2_reports_missing_input_path(tmp_path: Path) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    missing = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError) as error:
        run_direction2_v2(replace(config, shard_path=missing))

    assert error.value.args == (missing,)


def test_policy_record_is_a_stable_public_run_contract() -> None:
    assert v2_pipeline._policy_record() == {
        "fineweb_document_frequency_ratio": 0.001,
        "generic_osm_polygon_count_threshold": 1,
        "generic_requires_country_in_same_sentence": True,
        "minimum_name_letters": 3,
        "normalization_version": "v1-nfkc-casefold-separators",
        "short_single_token_max_letters": 8,
    }


def test_frequency_pass_requests_only_text_with_bounded_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeParquetFile:
        schema_arrow = SimpleNamespace(names=["text", "url"])

        def __init__(self, path: Path) -> None:
            self.path = path

        def iter_batches(self, **kwargs: object) -> tuple[object, ...]:
            calls.append(kwargs)
            return ()

    monkeypatch.setattr(v2_pipeline.pq, "ParquetFile", FakeParquetFile)

    assert v2_pipeline._count_document_frequencies(
        Path("unused.parquet"),
        patterns=("palais",),
        batch_size=17,
        log=io.StringIO(),
    ) == ({"palais": 0}, 0)
    assert calls == [{"batch_size": 17, "columns": ["text"], "use_threads": True}]


def test_match_pass_requests_text_and_url_with_bounded_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FakeParquetFile:
        schema_arrow = SimpleNamespace(names=["text", "url"])

        def __init__(self, path: Path) -> None:
            self.path = path

        def iter_batches(self, **kwargs: object) -> tuple[object, ...]:
            calls.append(kwargs)
            return ()

    monkeypatch.setattr(v2_pipeline.pq, "ParquetFile", FakeParquetFile)
    sources = (
        v2_pipeline._V2Source("monaco", tmp_path / "monaco.osm", "Monaco"),
        v2_pipeline._V2Source(
            "liechtenstein", tmp_path / "liechtenstein.osm", "Liechtenstein"
        ),
    )
    paths = tuple(tmp_path / f"{source.key}.parquet" for source in sources)

    result = v2_pipeline._scan_matches(
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
        {
            "batch_size": 19,
            "columns": ["text", "url"],
            "use_threads": True,
        }
    ]


def test_require_columns_reports_each_missing_column(tmp_path: Path) -> None:
    text_only = tmp_path / "text-only.parquet"
    pq.write_table(pa.table({"text": ["text"]}), text_only)
    url_only = tmp_path / "url-only.parquet"
    pq.write_table(pa.table({"url": ["url"]}), url_only)
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"other": ["other"]}), empty)

    with pytest.raises(ValueError) as text_error:
        v2_pipeline._require_columns(pq.ParquetFile(text_only))
    with pytest.raises(ValueError) as url_error:
        v2_pipeline._require_columns(pq.ParquetFile(url_only))
    with pytest.raises(ValueError) as both_error:
        v2_pipeline._require_columns(pq.ParquetFile(empty))

    assert str(text_error.value) == (
        "FineWeb shard must contain text and url columns; missing url"
    )
    assert str(url_error.value) == (
        "FineWeb shard must contain text and url columns; missing text"
    )
    assert str(both_error.value) == (
        "FineWeb shard must contain text and url columns; missing text, url"
    )


def test_record_match_separates_generic_and_distinctive_counts() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais du Prince",
        aliases=(),
        tags=(),
        centroid=None,
    )
    profiles = build_name_inventory(
        (polygon,),
        document_frequencies={"palais du prince": 1},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )
    match = V2NameMatcher.build(profiles).find("Palais du Prince")[0]
    result = v2_pipeline._ScanResult()

    v2_pipeline._record_match(result, match)

    assert result.matches_found == 1
    assert result.distinctive_matches == 1
    assert result.generic_matches == 0
    assert result.polygon_ids == {"monaco/way/10"}


def test_containing_span_rejects_the_end_boundary_with_a_stable_error() -> None:
    with pytest.raises(ValueError) as error:
        v2_pipeline._containing_span((v2_pipeline.SentenceSpan(0, 5),), 5)

    assert str(error.value) == "match_start is outside the document sentences"


def test_generic_match_uses_offsets_local_to_its_sentence() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Central",
        aliases=(),
        tags=(),
        centroid=None,
    )
    profiles = build_name_inventory(
        (polygon,),
        document_frequencies={"central": 1},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )
    text = "Lead sentence. Central in Monaco is a place."
    match = V2NameMatcher.build(profiles).find(text)[0]
    span = v2_pipeline.split_sentences(text)[1]

    assert v2_pipeline._keep_match(
        text,
        span,
        match,
        country_matcher=AhoCorasickPatternMatcher.build(("Monaco",)),
    )


def test_generic_match_rejects_country_text_overlapping_the_name() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Monaco Central",
        aliases=(),
        tags=(),
        centroid=None,
    )
    profiles = build_name_inventory(
        (polygon,),
        document_frequencies={"monaco central": 2},
        document_count=1000,
        country_names={"monaco": "Monaco"},
    )
    text = "Lead sentence. Monaco Central is a place."
    match = V2NameMatcher.build(profiles).find(text)[0]
    span = v2_pipeline.split_sentences(text)[1]

    assert not v2_pipeline._keep_match(
        text,
        span,
        match,
        country_matcher=AhoCorasickPatternMatcher.build(("Monaco",)),
    )


def test_as_text_maps_null_to_empty_text() -> None:
    assert v2_pipeline._as_text(None) == ""
    assert v2_pipeline._as_text(42) == "42"


def test_country_summaries_require_matching_source_and_output_counts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.parquet"
    output.write_bytes(b"result")
    source = v2_pipeline._V2Source("monaco", tmp_path / "monaco.osm", "Monaco")
    polygons = (
        PolygonRecord("monaco/way/1", "monaco", "One", (), (), None),
        PolygonRecord("monaco/way/2", "monaco", "Two", (), (), None),
        PolygonRecord("liechtenstein/way/3", "liechtenstein", "Three", (), (), None),
    )

    summary = v2_pipeline._country_summary(
        source,
        output,
        polygons=polygons,
        profiles=(),
        stats=v2_pipeline._CountryStats(),
    )

    assert summary.polygons_read == 2


def test_country_summaries_reject_mismatched_sequences(tmp_path: Path) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    sources = v2_pipeline._sources(config)

    with pytest.raises(ValueError):
        v2_pipeline._country_summaries(
            sources,
            (),
            polygons=(),
            profiles=(),
            country_stats={},
        )


def test_manifest_rejects_mismatched_country_summaries(tmp_path: Path) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    source = v2_pipeline._sources(config)[0]
    config.name_inventory_path.parent.mkdir(parents=True)
    config.name_inventory_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        v2_pipeline._manifest(
            config=config,
            sources=(source,),
            fingerprints={},
            profiles=(),
            polygons_read=0,
            frequency=v2_pipeline._FrequencyResult({}, 0, False),
            scan=v2_pipeline._ScanResult(),
            country_summaries=(),
        )


def test_run_direction2_v2_closes_nested_log_after_polygon_read_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    path = tmp_path / "nested" / "deeper" / "run.jsonl"
    config = replace(config, log_path=path)
    streams: list[Any] = []
    original_open = Path.open

    def recording_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        stream = original_open(self, *args, **kwargs)
        if self == path:
            streams.append(stream)
        return stream

    def fail_read(sources: object) -> None:
        raise OSError("polygon read failed")

    monkeypatch.setattr(Path, "open", recording_open)
    monkeypatch.setattr(v2_pipeline, "read_polygon_records", fail_read)

    for _ in range(2):
        with pytest.raises(OSError, match="polygon read failed"):
            run_direction2_v2(config)

    assert len(streams) == 2
    assert all(stream.closed for stream in streams)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "event": "run_started",
        "version": DIRECTION_V2_VERSION,
    }


def _output_row() -> dict[str, object]:
    row: dict[str, object] = {column: "" for column in OUTPUT_COLUMNS_V2}
    row["osm_polygon_count"] = 1
    row["fineweb_document_frequency"] = 0
    return row


def test_parquet_state_flushes_counts_and_publishes_atomically(tmp_path: Path) -> None:
    state = v2_pipeline._ParquetState(
        tmp_path / "nested" / "deeper" / "result.parquet",
        batch_size=2,
    )
    v2_pipeline._ParquetState.open(state)
    v2_pipeline._ParquetState.add(state, _output_row(), "polygon-1", "generic")
    assert state.stats.matches_found == 1
    assert state.rows
    v2_pipeline._ParquetState.add(state, _output_row(), "polygon-2", "generic")
    assert state.rows == []
    v2_pipeline._ParquetState.add(state, _output_row(), "polygon-3", "generic")
    assert state.stats.generic_matches == 3
    assert state.stats.distinctive_matches == 0
    v2_pipeline._ParquetState.add(state, _output_row(), "polygon-4", "distinctive")
    v2_pipeline._ParquetState.add(state, _output_row(), "polygon-5", "distinctive")
    v2_pipeline._ParquetState.publish(state)

    assert state.stats.matches_found == 5
    assert state.stats.generic_matches == 3
    assert state.stats.distinctive_matches == 2
    assert state.stats.polygon_ids == {
        "polygon-1",
        "polygon-2",
        "polygon-3",
        "polygon-4",
        "polygon-5",
    }
    assert state.path.is_file()
    assert not state.temporary.exists()
    result = pq.read_table(state.path)
    assert result.column_names == list(OUTPUT_COLUMNS_V2)
    assert result.num_rows == 5
    assert pq.ParquetFile(state.path).metadata.row_group(0).column(0).compression == (
        "ZSTD"
    )


def test_parquet_state_requests_the_versioned_compression_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeWriter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            calls["compression"] = kwargs["compression"]

        def close(self) -> None:
            pass

    monkeypatch.setattr(v2_pipeline.pq, "ParquetWriter", FakeWriter)
    state = v2_pipeline._ParquetState(tmp_path / "result.parquet", batch_size=2)

    state.open()

    assert calls["compression"] == "zstd"


def test_parquet_state_flush_passes_the_versioned_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class FakeWriter:
        def __init__(self) -> None:
            self.tables: list[object] = []

        def write_table(self, table: object) -> None:
            self.tables.append(table)

    writer = FakeWriter()

    class FakeTable:
        @staticmethod
        def from_pylist(rows: list[dict[str, object]], **kwargs: object) -> str:
            calls["rows"] = rows
            calls["schema"] = kwargs.get("schema")
            return "table"

    monkeypatch.setattr(v2_pipeline, "pa", SimpleNamespace(Table=FakeTable))
    state = v2_pipeline._ParquetState(tmp_path / "result.parquet", batch_size=2)
    state.writer = writer
    state.rows = [_output_row()]

    state.flush()

    assert calls["schema"] is v2_pipeline._OUTPUT_SCHEMA
    assert writer.tables == ["table"]
    assert state.rows == []


def test_parquet_state_abort_is_safe_before_open(tmp_path: Path) -> None:
    state = v2_pipeline._ParquetState(tmp_path / "result.parquet", batch_size=2)

    v2_pipeline._ParquetState.abort(state)

    assert not state.temporary.exists()


def test_parquet_outputs_reject_mismatched_source_and_output_counts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        v2_pipeline._ParquetOutputs(
            ("monaco",),
            (tmp_path / "one.parquet", tmp_path / "two.parquet"),
            batch_size=2,
        )


def test_write_card_uses_the_deterministic_temporary_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    run_direction2_v2(config)
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    factories: list[object] = []
    original_atomic_text_output = v2_pipeline.atomic_text_output

    def spy_atomic_text_output(path: Path, **kwargs: Any) -> Any:
        factories.append(kwargs.get("temporary_factory"))
        return original_atomic_text_output(path, **kwargs)

    monkeypatch.setattr(v2_pipeline, "atomic_text_output", spy_atomic_text_output)
    v2_pipeline._write_card(tmp_path / "copy.md", manifest)

    assert factories == [v2_pipeline.deterministic_temporary_path]


def test_v2_dataset_card_has_a_stable_complete_contract() -> None:
    manifest = {
        "configuration": {
            "fineweb_document_frequency_ratio": 0.001,
            "frequency_pass_reused": False,
            "minimum_name_letters": 3,
            "short_single_token_max_letters": 8,
        },
        "countries": {
            "monaco": {
                "distinctive_matches": 1,
                "generic_matches": 1,
                "matches_found": 2,
            },
            "liechtenstein": {
                "distinctive_matches": 1,
                "generic_matches": 1,
                "matches_found": 2,
            },
        },
        "polygon_inventory": {
            "generic_names": 1,
            "names_considered": 6,
            "names_discarded": 2,
            "names_indexed": 4,
            "polygons_read": 6,
        },
        "results": {
            "distinctive_matches": 2,
            "fineweb_docs_frequency_pass": 4,
            "fineweb_docs_match_pass": 4,
            "generic_matches": 2,
            "matches_found": 4,
            "unique_polygons_matched": 4,
        },
    }

    expected = (
        "\n".join(
            [
                "---",
                "config_name: direction_2_lexical_v2",
                "---",
                "# Direction 2 — lexical polygon candidates V2",
                "",
                "This version reduces generic-name noise while keeping the retrieval "
                "lexical and deterministic.",
                "",
                "## Measured run",
                "",
                "- 6 polygon objects read",
                "- 6 normalized names considered",
                "- 4 names indexed",
                "- 1 names classified as generic",
                "- 2 names discarded",
                "- 4 FineWeb documents in the frequency pass",
                "- 4 FineWeb documents in the matching pass",
                "- 4 matches written",
                "- 2 distinctive-name matches",
                "- 2 generic-name matches with country",
                "- 4 unique polygons matched",
                "",
                "## Rule",
                "",
                "A name is discarded when it has no letters, fewer than three "
                "alphabetic characters, or is the exact source country name. A name "
                "is generic when it is reused by more than one OSM polygon, appears "
                "in more than 0.1% of "
                "FineWeb documents, or is one token with at most eight letters.",
                "",
                "Distinctive names are matched directly. Generic names are kept only "
                "when the source country appears independently in the same sentence. "
                "The URL is provenance only. There is no LLM, embedding, thematic "
                "filter, tag "
                "filter, deduplication, or geographic disambiguation.",
                "",
                "## Configuration",
                "",
                "- FineWeb frequency ratio: 0.001",
                "- Minimum alphabetic characters: 3",
                "- Short single-token limit: 8",
                "- Frequency inventory reused: False",
                "",
                "## Columns",
                "",
                "| Column | Meaning |",
                "| --- | --- |",
                "| polygon_id | stable source/object identifier |",
                "| polygon_name | OSM main name value |",
                "| matched_alias | name or alias value that matched |",
                "| osm_tags | all OSM tags as sorted JSON |",
                "| centroid | centroid as JSON with latitude and longitude |",
                "| fineweb_url | FineWeb document URL |",
                "| sentence | the sentence containing the match |",
                "| context | the sentence plus one neighboring sentence on each side |",
                "| name_match_class | distinctive_name or generic_name_with_country |",
                "| osm_polygon_count | number of OSM polygons "
                "using the normalized name |",
                "| fineweb_document_frequency | FineWeb documents "
                "containing the normalized name |",
                "",
                "## Source splits",
                "",
                "| Source | Matches | Distinctive | Generic with country |",
                "| --- | ---: | ---: | ---: |",
                "| liechtenstein | 2 | 1 | 1 |",
                "| monaco | 2 | 1 | 1 |",
                "",
                "This card is generated deterministically from the run manifest. "
                "The full "
                "contract is in the GitHub V2 README at "
                "https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/"
                "lexical-candidates/lexical-v2/README.md. The original Direction 2 V1 "
                "README remains available.",
            ]
        )
        + "\n"
    )

    assert render_dataset_card(manifest) == expected


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("polygon_inventory", "manifest field 'polygon_inventory' must be an object"),
        ("results", "manifest field 'results' must be an object"),
        ("countries", "manifest field 'countries' must be an object"),
        ("configuration", "manifest field 'configuration' must be an object"),
    ),
)
def test_dataset_card_rejects_invalid_top_level_sections(
    field: str,
    message: str,
) -> None:
    manifest = {
        "polygon_inventory": {},
        "results": {},
        "countries": {},
        "configuration": {},
    }
    manifest[field] = None

    with pytest.raises(ValueError) as error:
        render_dataset_card(manifest)

    assert str(error.value) == message


def test_dataset_card_rejects_an_invalid_country_section() -> None:
    manifest = {
        "polygon_inventory": {},
        "results": {},
        "countries": {"monaco": None},
        "configuration": {},
    }

    with pytest.raises(ValueError) as error:
        render_dataset_card(manifest)

    assert str(error.value) == "manifest field 'country' must be an object"


def test_run_direction2_v2_reuses_a_matching_frequency_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = _make_v2_fixture(tmp_path)
    run_direction2_v2(config)

    def fail_frequency_pass(*args: object, **kwargs: object) -> None:
        raise AssertionError("frequency pass should have been reused")

    monkeypatch.setattr(
        v2_pipeline,
        "_count_document_frequencies",
        fail_frequency_pass,
    )

    summary = run_direction2_v2(config)

    assert summary.fineweb_docs_frequency_pass == 4
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["frequency_pass_reused"] is True


def test_run_direction2_v2_rebuilds_frequency_inventory_after_shard_change(
    tmp_path: Path, monkeypatch
) -> None:
    config, shard = _make_v2_fixture(tmp_path)
    run_direction2_v2(config)
    pq.write_table(
        pa.table({"text": ["A changed shard."], "url": ["changed"]}),
        shard,
    )
    calls: list[bool] = []
    original = v2_pipeline._count_document_frequencies

    def count_frequency_pass(
        shard_path: Path,
        *,
        patterns: tuple[str, ...],
        batch_size: int,
        log: object,
    ) -> tuple[dict[str, int], int]:
        calls.append(True)
        return original(
            shard_path,
            patterns=patterns,
            batch_size=batch_size,
            log=log,
        )

    monkeypatch.setattr(
        v2_pipeline,
        "_count_document_frequencies",
        count_frequency_pass,
    )

    summary = run_direction2_v2(config)

    assert calls == [True]
    assert summary.fineweb_docs_frequency_pass == 1
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["frequency_pass_reused"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fineweb_docs_scanned", "not-an-int"),
        ("names", [1]),
    ],
)
def test_cached_frequency_inventory_rejects_invalid_top_level_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    record = {
        "direction": DIRECTION_V2_VERSION,
        "fineweb_docs_scanned": 1,
        "inputs": {},
        "names": [],
        "policy": v2_pipeline._policy_record(),
        "status": "complete",
    }
    record[field] = value
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(record), encoding="utf-8")

    assert v2_pipeline._read_cached_frequencies(path, fingerprints={}) is None


@pytest.mark.parametrize(
    "decision",
    [
        1,
        {"document_frequency": "not-an-int"},
    ],
)
def test_cached_frequency_inventory_rejects_invalid_name_decisions(
    tmp_path: Path,
    decision: object,
) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "direction": DIRECTION_V2_VERSION,
                "fineweb_docs_scanned": 1,
                "inputs": {},
                "names": [
                    {
                        "normalized_name": "name",
                        "decision": decision,
                    }
                ],
                "policy": v2_pipeline._policy_record(),
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    assert v2_pipeline._read_cached_frequencies(path, fingerprints={}) is None


def test_cached_frequency_inventory_rejects_invalid_name_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "inventory.json"
    path.write_text(
        json.dumps(
            {
                "direction": DIRECTION_V2_VERSION,
                "fineweb_docs_scanned": 1,
                "inputs": {},
                "names": [{"normalized_name": None, "decision": {}}],
                "policy": v2_pipeline._policy_record(),
                "status": "complete",
            }
        ),
        encoding="utf-8",
    )

    assert v2_pipeline._read_cached_frequencies(path, fingerprints={}) is None


@pytest.mark.parametrize(
    ("polygon_count", "document_frequency", "document_count", "message"),
    [
        (-1, 0, 1, "polygon_count"),
        (1, -1, 1, "document_frequency"),
    ],
)
def test_name_policy_rejects_negative_counts(
    polygon_count: int,
    document_frequency: int,
    document_count: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        classify_name(
            "A valid name",
            polygon_count=polygon_count,
            document_frequency=document_frequency,
            document_count=document_count,
        )


def test_parquet_state_abort_closes_and_removes_temporary_output(
    tmp_path: Path,
) -> None:
    state = v2_pipeline._ParquetState(tmp_path / "result.parquet", batch_size=2)
    state.open()
    temporary = state.temporary
    assert temporary.is_file()

    state.abort()

    assert not temporary.exists()
