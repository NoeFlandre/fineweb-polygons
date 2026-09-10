import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.directions.lexical.matching import (
    AhoCorasickPolygonMatcher,
)
from fineweb_polygons.directions.lexical.models import (
    PolygonRecord,
    PolygonSource,
)
from fineweb_polygons.directions.lexical.osm import (
    centroid_from_geojson,
    read_polygon_records,
)
from fineweb_polygons.directions.lexical.sentences import (
    context_for_match,
    split_sentences,
)
from fineweb_polygons.directions.lexical.v1.card import render_dataset_card
from fineweb_polygons.directions.lexical.v1.models import (
    Direction2RunConfig,
)
from fineweb_polygons.directions.lexical.v1.pipeline import (
    _ParquetState,
    run_direction2,
)

MINI_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="43.70" lon="7.40" />
  <node id="2" lat="43.70" lon="7.41" />
  <node id="3" lat="43.71" lon="7.41" />
  <node id="4" lat="43.71" lon="7.40" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="Palais du Prince" />
    <tag k="name:en" v="Prince's Palace" />
    <tag k="building" v="castle" />
  </way>
  <way id="11">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
  </way>
</osm>
"""


def test_read_polygon_records_extracts_names_aliases_tags_and_centroid(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "mini.osm"
    pbf.write_text(MINI_OSM_XML, encoding="utf-8")

    records = read_polygon_records((PolygonSource("monaco", pbf),))

    assert len(records) == 1
    record = records[0]
    assert record.polygon_id == "monaco/way/10"
    assert record.name == "Palais du Prince"
    assert record.aliases == ("Prince's Palace",)
    assert dict(record.tags)["building"] == "castle"
    assert record.centroid == pytest.approx((7.405, 43.705))


def test_centroid_supports_holes_and_returns_none_for_invalid_geometry() -> None:
    geometry = {
        "coordinates": [
            [
                [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                [[1, 1], [3, 1], [3, 3], [1, 3], [1, 1]],
            ]
        ]
    }

    assert centroid_from_geojson(geometry) == pytest.approx((2.0, 2.0))
    assert centroid_from_geojson({}) is None
    assert centroid_from_geojson({"coordinates": "invalid"}) is None


def test_centroid_falls_back_to_outer_points_for_a_degenerate_ring() -> None:
    geometry = {
        "coordinates": [[[[0, 0], [1, 0], [2, 0], [0, 0]]]],
    }

    assert centroid_from_geojson(geometry) == pytest.approx((1.0, 0.0))


def test_centroid_mean_fallback_uses_both_coordinates() -> None:
    geometry = {
        "coordinates": [[[[0, 1], [1, 3], [2, 5], [0, 1]]]],
    }

    assert centroid_from_geojson(geometry) == pytest.approx((1.0, 3.0))


def test_centroid_ignores_malformed_coordinates() -> None:
    geometry = {
        "coordinates": [
            [
                [
                    ["invalid", 1],
                    [1],
                    [0, 0],
                    [1, 0],
                    [0, 0],
                ]
            ]
        ]
    }

    assert centroid_from_geojson(geometry) == pytest.approx((0.5, 0.0))


def test_direction2_config_rejects_invalid_settings_and_source_keys(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / name for name in ("monaco", "liechtenstein", "shard")]
    with pytest.raises(ValueError, match="batch_size"):
        Direction2RunConfig(
            paths[0],
            paths[1],
            paths[2],
            tmp_path / "output",
            tmp_path / "manifest.json",
            tmp_path / "card.md",
            tmp_path / "run.jsonl",
            batch_size=0,
        )
    with pytest.raises(ValueError, match="input paths"):
        Direction2RunConfig(
            paths[0],
            paths[1],
            paths[0],
            tmp_path / "output",
            tmp_path / "manifest.json",
            tmp_path / "card.md",
            tmp_path / "run.jsonl",
        )
    with pytest.raises(ValueError, match="source key"):
        PolygonSource(" ", paths[0])


def test_matcher_indexes_names_and_aliases_without_matching_inside_words() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais",
        aliases=("The Palace",),
        tags=(),
        centroid=None,
    )
    matcher = AhoCorasickPolygonMatcher.build((polygon,))

    matches = matcher.find(
        "A PALais is visible. The Palace is nearby. Palaisage is not a match."
    )

    assert matcher.names_indexed == 2
    assert [match.matched_alias for match in matches] == ["Palais", "The Palace"]
    assert [match.start for match in matches] == [2, 21]


def test_matcher_returns_no_matches_for_empty_text() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais",
        aliases=(),
        tags=(),
        centroid=None,
    )

    assert AhoCorasickPolygonMatcher.build((polygon,)).find("") == ()


def test_matcher_does_not_decode_percent_escapes_in_document_text() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="A B",
        aliases=(),
        tags=(),
        centroid=None,
    )

    assert AhoCorasickPolygonMatcher.build((polygon,)).find("A%20B") == ()


def test_matcher_skips_unsearchable_names_and_continues_to_aliases() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="!!!",
        aliases=("Valid",),
        tags=(),
        centroid=None,
    )

    matcher = AhoCorasickPolygonMatcher.build((polygon,))

    assert matcher.names_indexed == 1
    assert [match.matched_alias for match in matcher.find("Valid")] == ["Valid"]


def test_matcher_handles_casefold_expansions_and_preserves_match_offsets() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="ss",
        aliases=(),
        tags=(),
        centroid=None,
    )

    matches = AhoCorasickPolygonMatcher.build((polygon,)).find("ß")

    assert [(match.start, match.end) for match in matches] == [(0, 1)]


def test_matcher_requires_word_boundaries_at_both_document_edges() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais",
        aliases=(),
        tags=(),
        centroid=None,
    )
    matcher = AhoCorasickPolygonMatcher.build((polygon,))

    assert [(match.start, match.end) for match in matcher.find("Palais")] == [(0, 6)]
    assert [(match.start, match.end) for match in matcher.find("xPalais")] == []
    assert [(match.start, match.end) for match in matcher.find("Palaisx")] == []
    assert [(match.start, match.end) for match in matcher.find(".Palais")] == [(1, 7)]
    assert [(match.start, match.end) for match in matcher.find("Palais.")] == [(0, 6)]


def test_matcher_maps_offsets_after_a_separator() -> None:
    polygon = PolygonRecord(
        polygon_id="monaco/way/10",
        source_key="monaco",
        name="Palais",
        aliases=(),
        tags=(),
        centroid=None,
    )

    matches = AhoCorasickPolygonMatcher.build((polygon,)).find("Prefix: Palais")

    assert [(match.start, match.end) for match in matches] == [(8, 14)]


def test_sentence_context_includes_one_neighbor_each_side() -> None:
    text = "Before. Palais is visible. After. Last."
    spans = split_sentences(text)
    window = context_for_match(text, spans, match_start=8)

    assert window.sentence == "Palais is visible."
    assert window.context == "Before. Palais is visible. After."


def test_sentence_split_keeps_unpunctuated_text_and_rejects_unknown_offsets() -> None:
    spans = split_sentences("One sentence")

    assert len(spans) == 1
    with pytest.raises(ValueError, match="outside"):
        context_for_match("One sentence", spans, match_start=99)


def test_run_direction2_streams_fineweb_and_writes_the_declared_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    monaco.write_text(MINI_OSM_XML, encoding="utf-8")
    liechtenstein.write_text(
        MINI_OSM_XML.replace("Palais du Prince", "Vaduz"), encoding="utf-8"
    )
    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [
                    "No polygon here.",
                    "Intro. The Prince's Palace is visible. End.",
                    "Vaduz has a castle. Other text.",
                ],
                "url": ["https://zero", "https://one", "https://two"],
            }
        ),
        shard,
    )
    output_dir = tmp_path / "output"
    manifest = tmp_path / "runs" / "manifest.json"
    card = tmp_path / "output" / "dataset-card.md"
    log = tmp_path / "nested" / "logs" / "run.jsonl"
    original_open = Path.open
    log_encodings: list[object] = []

    def open_spy(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == log and args and args[0] == "w":
            log_encodings.append(kwargs.get("encoding"))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_spy)

    summary = run_direction2(
        Direction2RunConfig(
            monaco_pbf=monaco,
            liechtenstein_pbf=liechtenstein,
            shard_path=shard,
            output_dir=output_dir,
            manifest_path=manifest,
            dataset_card_path=card,
            log_path=log,
            batch_size=2,
            output_batch_size=1,
        )
    )

    assert summary.polygons_read == 2
    assert summary.names_indexed == 3
    assert summary.fineweb_docs_scanned == 3
    assert summary.matches_found == 3
    assert summary.unique_polygons_matched == 2
    assert summary.manifest_path == manifest
    assert summary.dataset_card_path == card
    assert summary.log_path == log
    assert log_encodings == ["utf-8"]
    assert [
        (
            item.source_key,
            item.output_path,
            item.polygons_read,
            item.names_indexed,
            item.matches_found,
            item.unique_polygons_matched,
        )
        for item in summary.country_summaries
    ] == [
        ("monaco", output_dir / "monaco.parquet", 1, 2, 1, 1),
        ("liechtenstein", output_dir / "liechtenstein.parquet", 1, 2, 2, 1),
    ]
    assert set(summary.output_paths) == {
        output_dir / "monaco.parquet",
        output_dir / "liechtenstein.parquet",
    }
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete"
    log_records = [
        json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["event"] for record in log_records] == [
        "run_started",
        "polygons_loaded",
        "progress",
        "progress",
        "run_completed",
    ]
    assert log_records[0] == {
        "event": "run_started",
        "version": "direction-2-lexical-v1",
    }
    assert log_records[1]["names_indexed"] == 3
    assert log_records[1]["polygons_read"] == 2
    assert log_records[-1] == {"event": "run_completed", **summary.to_record()}
    assert [
        record["docs_scanned"]
        for record in log_records
        if record["event"] == "progress"
    ] == [2, 3]
    assert card.is_file()
    manifest_record = json.loads(manifest.read_text(encoding="utf-8"))
    from fineweb_polygons.core.artifact_io import sha256_file
    from fineweb_polygons.core.normalization import NORMALIZATION_VERSION
    from fineweb_polygons.directions.lexical.models import OUTPUT_COLUMNS
    from fineweb_polygons.directions.lexical.v1.models import DATA_PREFIX

    assert set(manifest_record) == {
        "configuration",
        "countries",
        "direction",
        "polygon_inventory",
        "results",
        "schema",
        "sources",
        "status",
    }
    assert manifest_record["configuration"] == {
        "batch_size": 2,
        "geographic_disambiguation": False,
        "matcher": "Aho-Corasick",
        "normalization_version": NORMALIZATION_VERSION,
        "output_batch_size": 1,
        "sentence_context": "matching sentence plus one sentence on each side",
        "thematic_filtering": False,
    }
    assert manifest_record["countries"] == {
        item.source_key: item.to_record() for item in summary.country_summaries
    }
    assert summary.country_summaries[0].result_sha256 == sha256_file(
        output_dir / "monaco.parquet"
    )
    assert summary.country_summaries[1].result_sha256 == sha256_file(
        output_dir / "liechtenstein.parquet"
    )
    assert manifest_record["direction"] == "direction-2-lexical-v1"
    assert manifest_record["polygon_inventory"] == {
        "named_polygons": 2,
        "names_indexed": 3,
        "polygons_read": 2,
    }
    assert manifest_record["results"] == {
        "files": [
            {
                "path": f"{DATA_PREFIX}/monaco.parquet",
                "sha256": summary.country_summaries[0].result_sha256,
                "source_key": "monaco",
            },
            {
                "path": f"{DATA_PREFIX}/liechtenstein.parquet",
                "sha256": summary.country_summaries[1].result_sha256,
                "source_key": "liechtenstein",
            },
        ],
        "fineweb_docs_scanned": 3,
        "matches_found": 3,
        "unique_polygons_matched": 2,
    }
    assert manifest_record["schema"] == list(OUTPUT_COLUMNS)
    assert manifest_record["sources"] == {
        "fineweb_shard": {
            "path": str(shard),
            "sha256": sha256_file(shard),
        },
        "osm_pbf": [
            {
                "path": str(monaco),
                "sha256": sha256_file(monaco),
                "source_key": "monaco",
            },
            {
                "path": str(liechtenstein),
                "sha256": sha256_file(liechtenstein),
                "source_key": "liechtenstein",
            },
        ],
    }
    assert manifest_record["status"] == "complete"

    monaco_rows = pq.read_table(output_dir / "monaco.parquet")
    assert monaco_rows.column_names == [
        "polygon_id",
        "polygon_name",
        "matched_alias",
        "osm_tags",
        "centroid",
        "fineweb_url",
        "sentence",
        "context",
    ]
    assert monaco_rows["matched_alias"].to_pylist() == ["Prince's Palace"]
    assert monaco_rows["sentence"].to_pylist() == ["The Prince's Palace is visible."]
    assert monaco_rows.to_pylist() == [
        {
            "polygon_id": "monaco/way/10",
            "polygon_name": "Palais du Prince",
            "matched_alias": "Prince's Palace",
            "osm_tags": (
                '{"building":"castle","name":"Palais du Prince",'
                '"name:en":"Prince\u0027s Palace"}'
            ),
            "centroid": '{"lat":43.704999995373186,"lon":7.404999999215664}',
            "fineweb_url": "https://one",
            "sentence": "The Prince's Palace is visible.",
            "context": "Intro. The Prince's Palace is visible. End.",
        }
    ]


def test_run_direction2_publishes_empty_country_files_when_there_are_no_matches(
    tmp_path: Path,
) -> None:
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    monaco.write_text(MINI_OSM_XML, encoding="utf-8")
    liechtenstein.write_text(MINI_OSM_XML, encoding="utf-8")
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"text": ["Nothing named here."], "url": ["url"]}), shard)
    output_dir = tmp_path / "output"

    run_direction2(
        Direction2RunConfig(
            monaco_pbf=monaco,
            liechtenstein_pbf=liechtenstein,
            shard_path=shard,
            output_dir=output_dir,
            manifest_path=tmp_path / "manifest.json",
            dataset_card_path=tmp_path / "card.md",
            log_path=tmp_path / "run.jsonl",
        )
    )

    result = pq.read_table(output_dir / "monaco.parquet")
    assert result.num_rows == 0
    assert result.column_names == [
        "polygon_id",
        "polygon_name",
        "matched_alias",
        "osm_tags",
        "centroid",
        "fineweb_url",
        "sentence",
        "context",
    ]


def test_run_direction2_removes_temporary_outputs_after_scan_failure(
    tmp_path: Path, monkeypatch
) -> None:
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    monaco.write_text(MINI_OSM_XML, encoding="utf-8")
    liechtenstein.write_text(MINI_OSM_XML, encoding="utf-8")
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"text": ["Palais."], "url": ["url"]}), shard)
    output_dir = tmp_path / "output"

    def fail_find(self, text: str):
        raise RuntimeError("matching failed")

    monkeypatch.setattr(AhoCorasickPolygonMatcher, "find", fail_find)
    with pytest.raises(RuntimeError, match="matching failed"):
        run_direction2(
            Direction2RunConfig(
                monaco_pbf=monaco,
                liechtenstein_pbf=liechtenstein,
                shard_path=shard,
                output_dir=output_dir,
                manifest_path=tmp_path / "manifest.json",
                dataset_card_path=tmp_path / "card.md",
                log_path=tmp_path / "run.jsonl",
            )
        )

    assert not (output_dir / "monaco.parquet").exists()
    assert not (output_dir / ".monaco.parquet.tmp").exists()


def test_parquet_state_abort_closes_writer_and_removes_temporary_output(
    tmp_path: Path,
) -> None:
    class TrackingWriter:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    state = _ParquetState(tmp_path / "result.parquet", batch_size=1)
    state.temporary.parent.mkdir(parents=True, exist_ok=True)
    state.temporary.touch()
    writer = TrackingWriter()
    state.writer = writer

    state.abort()

    assert writer.close_calls == 1
    assert not state.temporary.exists()


def test_run_direction2_rejects_a_shard_without_the_url_column(tmp_path: Path) -> None:
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    monaco.write_text(MINI_OSM_XML, encoding="utf-8")
    liechtenstein.write_text(MINI_OSM_XML, encoding="utf-8")
    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"text": ["Palais."]}), shard)

    with pytest.raises(ValueError, match="url"):
        run_direction2(
            Direction2RunConfig(
                monaco_pbf=monaco,
                liechtenstein_pbf=liechtenstein,
                shard_path=shard,
                output_dir=tmp_path / "output",
                manifest_path=tmp_path / "manifest.json",
                dataset_card_path=tmp_path / "card.md",
                log_path=tmp_path / "run.jsonl",
            )
        )


def test_dataset_card_is_deterministic_and_uses_manifest_counts() -> None:
    manifest = {
        "polygon_inventory": {
            "polygons_read": 2,
            "names_indexed": 3,
        },
        "results": {
            "fineweb_docs_scanned": 3,
            "matches_found": 2,
            "unique_polygons_matched": 2,
        },
        "countries": {
            "monaco": {"matches_found": 1},
            "liechtenstein": {"matches_found": 1},
        },
    }

    first = render_dataset_card(manifest)
    second = render_dataset_card(manifest)

    assert first == second
    assert "2 polygon objects" in first
    assert "3 unique normalized names" in first
    assert "3 FineWeb documents" in first
    assert "monaco" in first


def test_dataset_card_matches_the_complete_stable_contract() -> None:
    manifest = {
        "polygon_inventory": {
            "polygons_read": 2,
            "names_indexed": 3,
        },
        "results": {
            "fineweb_docs_scanned": 3,
            "matches_found": 2,
            "unique_polygons_matched": 2,
        },
        "countries": {
            "monaco": {"matches_found": 1},
            "liechtenstein": {"matches_found": 1},
        },
    }

    assert render_dataset_card(manifest) == (
        "---\n"
        "config_name: direction_2_lexical_v1\n"
        "---\n"
        "# Direction 2 — lexical polygon candidates\n"
        "\n"
        "This artifact is a lexical candidate-generation POC. It scans the FineWeb "
        "shard for OSM polygon names and `name:*` aliases.\n"
        "\n"
        "## Measured run\n"
        "\n"
        "- 2 polygon objects read\n"
        "- 3 unique normalized names indexed\n"
        "- 3 FineWeb documents scanned\n"
        "- 2 name mentions written\n"
        "- 2 unique polygons matched\n"
        "\n"
        "## Rule\n"
        "\n"
        "A row is written for every boundary-aware name or alias mention in the "
        "FineWeb document text. The row contains the matching sentence and the "
        "sentence immediately before and after it when available.\n"
        "\n"
        "No URL matching, LLM, embedding, thematic filter, or geographic "
        "disambiguation is used.\n"
        "\n"
        "## Columns\n"
        "\n"
        "| Column | Meaning |\n"
        "| --- | --- |\n"
        "| `polygon_id` | stable source/object identifier |\n"
        "| `polygon_name` | OSM main `name` value |\n"
        "| `matched_alias` | name or alias value that matched |\n"
        "| `osm_tags` | all OSM tags as sorted JSON |\n"
        "| `centroid` | centroid as JSON with latitude and longitude |\n"
        "| `fineweb_url` | FineWeb document URL |\n"
        "| `sentence` | the sentence containing the match |\n"
        "| `context` | the sentence plus one neighboring sentence on each side |\n"
        "\n"
        "## Source splits\n"
        "\n"
        "| Source | Matches |\n"
        "| --- | ---: |\n"
        "| `liechtenstein` | 1 |\n"
        "| `monaco` | 1 |\n"
        "\n"
        "This card is generated deterministically from the run manifest. The full "
        "Direction 2 contract is in the [GitHub direction README](https://github.com/"
        "NoeFlandre/fineweb-polygons/blob/main/docs/directions/lexical-candidates/"
        "README.md); the frozen Direction 1 archive remains in the same repository.\n"
    )


@pytest.mark.parametrize("field", ("polygon_inventory", "results", "countries"))
def test_dataset_card_rejects_non_object_manifest_fields(field: str) -> None:
    manifest = {
        "polygon_inventory": {"polygons_read": 1, "names_indexed": 1},
        "results": {
            "fineweb_docs_scanned": 1,
            "matches_found": 1,
            "unique_polygons_matched": 1,
        },
        "countries": {"monaco": {"matches_found": 1}},
    }
    manifest[field] = []

    with pytest.raises(ValueError, match=f"manifest field {field!r}"):
        render_dataset_card(manifest)


def test_dataset_card_rejects_a_non_object_country_summary() -> None:
    manifest = {
        "polygon_inventory": {"polygons_read": 1, "names_indexed": 1},
        "results": {
            "fineweb_docs_scanned": 1,
            "matches_found": 1,
            "unique_polygons_matched": 1,
        },
        "countries": {"monaco": []},
    }

    with pytest.raises(ValueError, match="manifest field 'country'"):
        render_dataset_card(manifest)


def test_matchers_disable_url_decoding_at_every_normalization_boundary(
    monkeypatch,
) -> None:
    from fineweb_polygons.directions.lexical import matching as matching_module

    original = matching_module.normalize_for_search
    calls: list[bool] = []

    def spy(value: object, *, decode_url: bool = True) -> str:
        calls.append(decode_url)
        return original(value, decode_url=decode_url)

    monkeypatch.setattr(matching_module, "normalize_for_search", spy)
    polygon = PolygonRecord("sample/way/1", "sample", "Palais", (), (), None)

    matching_module.AhoCorasickPolygonMatcher.build((polygon,)).find("Palais")
    matching_module.AhoCorasickPatternMatcher.build(("Palais",)).find("Palais")

    assert calls == [False, False, False, False]


def test_polygon_matcher_discards_matches_changed_by_source_normalization() -> None:
    from fineweb_polygons.directions.lexical.matching import AhoCorasickPolygonMatcher

    polygon = PolygonRecord("sample/way/1", "sample", "é", (), (), None)
    assert AhoCorasickPolygonMatcher.build((polygon,)).find("e" + chr(0x301)) == ()


def test_pattern_matcher_preserves_offsets_for_expanded_separator_characters() -> None:
    from fineweb_polygons.directions.lexical.matching import AhoCorasickPatternMatcher

    matches = AhoCorasickPatternMatcher.build(("1 4",)).find(chr(0xBC))

    assert [(match.pattern, match.start, match.end) for match in matches] == [
        ("1 4", 0, 1)
    ]


def test_closing_point_removal_keeps_singletons_and_handles_two_points() -> None:
    from fineweb_polygons.directions.lexical import osm as osm_module

    assert osm_module._without_closing_point([(1.0, 1.0)]) == [(1.0, 1.0)]
    assert osm_module._without_closing_point([(1.0, 1.0), (1.0, 1.0)]) == [(1.0, 1.0)]


def test_ring_moment_accepts_triangles_and_rejects_short_rings() -> None:
    from fineweb_polygons.directions.lexical import osm as osm_module

    triangle = [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)]

    assert osm_module._ring_moment(triangle, is_outer=True) == pytest.approx(
        (2.0, 4 / 3, 4 / 3)
    )
    assert osm_module._ring_moment(triangle[:2], is_outer=True) == (0.0, 0.0, 0.0)
    assert osm_module._polygon_moment(None) == (0.0, 0.0, 0.0, [])


def test_centroid_accumulates_multipolygons_and_subtracts_asymmetric_holes() -> None:
    first = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    second = [[10, 2], [12, 2], [12, 4], [10, 4], [10, 2]]
    geometry = {"coordinates": [[first], [second]]}
    holed_geometry = {
        "coordinates": [
            [
                [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
                [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
            ]
        ]
    }

    assert centroid_from_geojson(geometry) == pytest.approx((8.9, 2.5))
    assert centroid_from_geojson(holed_geometry) == pytest.approx((2.1, 2.1))


def test_centroid_uses_mean_at_the_area_epsilon_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fineweb_polygons.directions.lexical import osm as osm_module

    monkeypatch.setattr(osm_module, "_GEOMETRY_EPSILON", 4.0)
    triangle = [(3.0, 4.0), (5.0, 7.0), (3.0, 8.0)]

    assert osm_module._ring_centroid(triangle) == (4.0, 3.0, 4.0)
    assert centroid_from_geojson({"coordinates": [[triangle]]}) == pytest.approx(
        (11 / 3, 19 / 3)
    )


def test_aliases_skip_empty_values_and_deduplicate_values() -> None:
    from fineweb_polygons.directions.lexical import osm as osm_module

    tags = (("name:fr", ""), ("name:en", "Alpha"), ("name:de", "Alpha"))

    assert osm_module._aliases(tags) == ("Alpha",)


def test_area_record_defaults_missing_name_and_identifies_relations() -> None:
    from typing import Any, cast

    from fineweb_polygons.directions.lexical import osm as osm_module

    class FakeArea:
        def __init__(self) -> None:
            self.tags = {"building": "yes"}

        def from_way(self) -> bool:
            return False

        def orig_id(self) -> int:
            return 9

    class FakeFactory:
        def create_multipolygon(self, area: object) -> str:
            return '{"coordinates":[[[[0,0],[1,0],[0,1],[0,0]]]]}'

    record = osm_module._area_record(
        PolygonSource("sample", Path("source")),
        FakeArea(),
        cast(Any, FakeFactory()),
    )

    assert record.name == ""
    assert record.polygon_id == "sample/relation/9"


def test_validate_inputs_reports_the_exact_missing_path(tmp_path: Path) -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    missing = tmp_path / "missing.pbf"
    config = Direction2RunConfig(
        missing,
        tmp_path / "liechtenstein.pbf",
        tmp_path / "shard.parquet",
        tmp_path / "output",
        tmp_path / "manifest.json",
        tmp_path / "card.md",
        tmp_path / "run.jsonl",
    )

    with pytest.raises(FileNotFoundError) as error:
        pipeline_module._validate_inputs(config)

    assert error.value.args == (missing,)


def test_log_event_writes_the_event_and_values_as_one_json_record() -> None:
    from io import StringIO

    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    stream = StringIO()
    pipeline_module._log_event(stream, "progress", docs_scanned=3)

    assert json.loads(stream.getvalue()) == {"event": "progress", "docs_scanned": 3}


def test_scan_and_country_states_start_empty() -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    scan = pipeline_module._ScanResult()
    country = pipeline_module._CountryStats()

    assert scan.docs_scanned == 0
    assert scan.matches_found == 0
    assert scan.polygon_ids == set()
    assert scan.country_stats == {}
    assert country.matches_found == 0
    assert country.polygon_ids == set()


def test_as_text_maps_null_values_to_empty_text() -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    assert pipeline_module._as_text(None) == ""
    assert pipeline_module._as_text(7) == "7"


def test_country_summaries_require_equal_source_and_output_lengths(
    tmp_path: Path,
) -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    source_path = tmp_path / "source.parquet"
    extra_path = tmp_path / "extra.parquet"
    source_path.write_bytes(b"source")
    extra_path.write_bytes(b"extra")
    stats = pipeline_module._CountryStats()

    with pytest.raises(ValueError):
        pipeline_module._country_summaries(
            (PolygonSource("monaco", source_path),),
            (source_path, extra_path),
            (),
            {"monaco": stats},
        )


def test_parquet_state_enforces_schema_and_flush_threshold(tmp_path: Path) -> None:
    from typing import Any

    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    class TrackingWriter:
        def __init__(self) -> None:
            self.tables: list[Any] = []

        def write_table(self, table: Any) -> None:
            self.tables.append(table)

        def close(self) -> None:
            pass

    state = pipeline_module._ParquetState(tmp_path / "result.parquet", batch_size=1)
    writer = TrackingWriter()
    state.writer = writer

    state.add({"polygon_id": "sample/way/1"}, "sample/way/1")

    assert len(writer.tables) == 1
    assert writer.tables[0].schema.names == list(pipeline_module.OUTPUT_COLUMNS)
    assert state.rows == []
    assert state.stats.matches_found == 1
    assert state.stats.polygon_ids == {"sample/way/1"}


def test_parquet_state_abort_is_safe_when_no_temporary_file_exists(
    tmp_path: Path,
) -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    state = pipeline_module._ParquetState(tmp_path / "result.parquet", batch_size=1)
    state.abort()


def test_scan_requests_only_required_columns_with_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import StringIO
    from typing import Any

    from fineweb_polygons.directions.lexical import matching as matching_module
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    calls: list[dict[str, Any]] = []

    class FakeFile:
        schema_arrow = type("Schema", (), {"names": ("text", "url", "title")})()

        def iter_batches(self, **kwargs: Any):
            calls.append(kwargs)
            return iter(())

    monkeypatch.setattr(pipeline_module.pq, "ParquetFile", lambda path: FakeFile())
    pipeline_module._scan_fineweb(
        Path("shard.parquet"),
        matcher=matching_module.AhoCorasickPolygonMatcher.build(()),
        output_paths=(),
        batch_size=17,
        output_batch_size=2,
        log=StringIO(),
    )

    assert calls == [
        {"batch_size": 17, "columns": ["text", "url"], "use_threads": True}
    ]


def test_require_columns_reports_sorted_missing_names(tmp_path: Path) -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    shard = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"other": ["value"]}), shard)

    with pytest.raises(ValueError) as error:
        pipeline_module._require_columns(pq.ParquetFile(shard))

    assert str(error.value) == (
        "FineWeb shard must contain text and url columns; missing text, url"
    )


def test_write_card_selects_the_deterministic_temporary_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from contextlib import nullcontext
    from io import StringIO

    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    stream = StringIO()
    captured: dict[str, object] = {}

    def fake_atomic(path: Path, *, temporary_factory):
        captured["path"] = path
        captured["temporary_factory"] = temporary_factory
        return nullcontext(stream)

    monkeypatch.setattr(pipeline_module, "atomic_text_output", fake_atomic)
    monkeypatch.setattr(pipeline_module, "render_dataset_card", lambda manifest: "card")
    path = tmp_path / "card.md"

    pipeline_module._write_card(path, {})

    assert captured == {
        "path": path,
        "temporary_factory": pipeline_module.deterministic_temporary_path,
    }
    assert stream.getvalue() == "card"


def test_sentence_context_keeps_only_one_neighbor_on_each_side() -> None:
    spans = split_sentences("Zero. One. Palais. Three. Four.")

    window = context_for_match("Zero. One. Palais. Three. Four.", spans, match_start=10)

    assert window.context == "One. Palais. Three."


def test_sentence_index_treats_sentence_end_as_the_next_boundary() -> None:
    from fineweb_polygons.directions.lexical import sentences as sentence_module

    spans = split_sentences("One. Two.")

    assert sentence_module._sentence_index(spans, spans[0].end) == 1
    with pytest.raises(
        ValueError, match=r"\Amatch_start is outside the document sentences\Z"
    ):
        sentence_module._sentence_index(spans, 99)


def test_read_json_object_uses_utf8_encoding() -> None:
    from typing import cast

    from fineweb_polygons.core import artifact_io

    captured: dict[str, str] = {}

    class FakePath:
        def read_text(self, *, encoding: str) -> str:
            captured["encoding"] = encoding
            return '{"ok": true}'

    assert artifact_io.read_json_object(cast(Path, FakePath())) == {"ok": True}
    assert captured == {"encoding": "utf-8"}


def test_as_float_reports_non_numeric_values() -> None:
    from fineweb_polygons.directions.lexical import osm as osm_module

    with pytest.raises(TypeError, match="coordinate must be numeric") as error:
        osm_module._as_float(object())

    assert str(error.value) == "coordinate must be numeric"


def test_parquet_state_open_creates_parent_and_uses_zstd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    captured: dict[str, object] = {}

    class TrackingWriter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["args"] = args
            captured["kwargs"] = kwargs

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline_module.pq, "ParquetWriter", TrackingWriter)
    state = pipeline_module._ParquetState(
        tmp_path / "nested" / "deeper" / "result.parquet",
        batch_size=1,
    )

    state.open()

    assert state.path.parent.is_dir()
    assert captured == {
        "args": (state.temporary, pipeline_module._OUTPUT_SCHEMA),
        "kwargs": {"compression": "zstd"},
    }
    state.abort()


def test_run_direction2_tracks_multiple_polygon_ids_per_country(
    tmp_path: Path,
) -> None:
    extra_way = """
  <way id="12">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="Casino" />
    <tag k="building" v="yes" />
  </way>
"""
    monaco = tmp_path / "monaco.osm"
    liechtenstein = tmp_path / "liechtenstein.osm"
    monaco.write_text(
        MINI_OSM_XML.replace("</osm>", extra_way + "</osm>"),
        encoding="utf-8",
    )
    liechtenstein.write_text(
        MINI_OSM_XML.replace("Palais du Prince", "Vaduz"),
        encoding="utf-8",
    )
    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["Palais du Prince. Casino.", "Vaduz."],
                "url": ["https://one", "https://two"],
            }
        ),
        shard,
    )

    summary = run_direction2(
        Direction2RunConfig(
            monaco_pbf=monaco,
            liechtenstein_pbf=liechtenstein,
            shard_path=shard,
            output_dir=tmp_path / "output",
            manifest_path=tmp_path / "manifest.json",
            dataset_card_path=tmp_path / "card.md",
            log_path=tmp_path / "run.jsonl",
        )
    )

    assert summary.matches_found == 3
    assert summary.unique_polygons_matched == 3
    assert [
        (
            item.source_key,
            item.polygons_read,
            item.names_indexed,
            item.matches_found,
            item.unique_polygons_matched,
        )
        for item in summary.country_summaries
    ] == [
        ("monaco", 2, 3, 2, 2),
        ("liechtenstein", 1, 2, 1, 1),
    ]
    from fineweb_polygons.core.artifact_io import sha256_file

    assert summary.country_summaries[0].result_sha256 == sha256_file(
        tmp_path / "output" / "monaco.parquet"
    )
    assert summary.country_summaries[1].result_sha256 == sha256_file(
        tmp_path / "output" / "liechtenstein.parquet"
    )
    assert pq.read_table(tmp_path / "output" / "monaco.parquet")[
        "polygon_id"
    ].to_pylist() == ["monaco/way/10", "monaco/way/12"]


def test_manifest_requires_equal_sources_and_country_summaries(
    tmp_path: Path,
) -> None:
    from fineweb_polygons.directions.lexical.v1 import pipeline as pipeline_module

    source_path = tmp_path / "monaco.osm"
    liechtenstein_path = tmp_path / "liechtenstein.osm"
    shard_path = tmp_path / "shard.parquet"
    source_path.write_bytes(b"source")
    shard_path.write_bytes(b"shard")
    config = Direction2RunConfig(
        source_path,
        liechtenstein_path,
        shard_path,
        tmp_path / "output",
        tmp_path / "manifest.json",
        tmp_path / "card.md",
        tmp_path / "run.jsonl",
    )

    with pytest.raises(ValueError):
        pipeline_module._manifest(
            config=config,
            sources=(PolygonSource("monaco", source_path),),
            polygons=(),
            names_indexed=0,
            scan=pipeline_module._ScanResult(),
            country_summaries=(),
        )


def test_normalizer_starts_without_pending_separator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fineweb_polygons.directions.lexical import matching as matching_module

    seen: list[int | None] = []
    original = matching_module._append_folded_character

    def spy(
        characters: list[str],
        offsets: list[int],
        original_index: int,
        character: str,
        pending_separator: int | None,
    ) -> int | None:
        seen.append(pending_separator)
        return original(
            characters,
            offsets,
            original_index,
            character,
            pending_separator,
        )

    monkeypatch.setattr(matching_module, "_append_folded_character", spy)
    matching_module._normalize_with_offsets("A")

    assert seen == [None]


def test_normalizer_preserves_separator_source_offsets() -> None:
    from fineweb_polygons.directions.lexical import matching as matching_module

    characters = ["a"]
    offsets = [0]
    pending = matching_module._append_folded_character(
        characters,
        offsets,
        2,
        " ",
        None,
    )
    matching_module._append_folded_character(
        characters,
        offsets,
        3,
        "b",
        pending,
    )

    assert characters == ["a", " ", "b"]
    assert offsets == [0, 2, 3]


def test_polygon_moment_keeps_outer_and_hole_signs() -> None:
    from fineweb_polygons.directions.lexical import osm as osm_module

    outer = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    hole = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]

    area, _, _, _ = osm_module._polygon_moment([outer, hole])

    assert area == pytest.approx(15.0)
