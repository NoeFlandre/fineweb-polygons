import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.direction2.card import render_dataset_card
from fineweb_polygons.direction2.matching import (
    AhoCorasickPolygonMatcher,
)
from fineweb_polygons.direction2.models import (
    Direction2RunConfig,
    PolygonRecord,
    PolygonSource,
)
from fineweb_polygons.direction2.pipeline import _ParquetState, run_direction2
from fineweb_polygons.direction2.polygons import (
    centroid_from_geojson,
    read_polygon_records,
)
from fineweb_polygons.direction2.sentences import (
    context_for_match,
    split_sentences,
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
    tmp_path: Path,
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
                    "Intro. The Prince's Palace is visible. End.",
                    "Vaduz has a castle. Other text.",
                    "No polygon here.",
                ],
                "url": ["https://one", "https://two", "https://three"],
            }
        ),
        shard,
    )
    output_dir = tmp_path / "output"
    manifest = tmp_path / "runs" / "manifest.json"
    card = tmp_path / "output" / "dataset-card.md"
    log = tmp_path / "logs" / "run.jsonl"

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
    assert set(summary.output_paths) == {
        output_dir / "monaco.parquet",
        output_dir / "liechtenstein.parquet",
    }
    assert json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete"
    assert "run_completed" in log.read_text(encoding="utf-8")
    assert card.is_file()

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
