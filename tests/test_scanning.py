import json
from io import StringIO
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import FineWebDocument, PolygonProfile
from fineweb_polygons.scanning import (
    ScanStats,
    _as_text,
    _projected_columns,
    _row_start,
    _scan_batch,
    _scan_batches,
    _scan_partition,
    _select_row_groups,
    _validate_row_group_bounds,
    _write_matches,
    scan_row_group,
    scan_row_groups,
)


def write_fixture_shard(path: Path) -> Path:
    table = pa.table(
        {
            "id": ["doc-0", "doc-1"],
            "text": ["No relevant place.", "Fontvieille is in Monaco."],
            "url": ["https://example.test/other", "https://example.test/fontvieille"],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def write_multi_group_shard(path: Path) -> Path:
    table = pa.table(
        {
            "id": ["doc-0", "doc-1", "doc-2", "doc-3"],
            "text": [
                "Fontvieille is in Monaco.",
                "No relevant place.",
                "No relevant place.",
                "Fontvieille is in Monaco.",
            ],
            "url": ["", "", "", "https://example.test/fontvieille"],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def test_scan_row_group_writes_matching_evidence(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path / "shard.parquet")
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    stats = scan_row_group(
        shard,
        row_group_index=0,
        matcher=matcher,
        output_path=output,
    )

    assert stats.rows_scanned == 2
    assert stats.matches_written == 1
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["fineweb_row_index"] == 1
    assert record["fineweb_document_id"] == "doc-1"
    assert record["polygon_id"] == "way/1"
    assert record["url"] == "https://example.test/fontvieille"


def test_scan_row_group_requires_text_and_url(tmp_path: Path) -> None:
    shard = tmp_path / "missing-url.parquet"
    pq.write_table(pa.table({"text": ["Monaco"]}), shard)

    with pytest.raises(ValueError, match=r"text.*url"):
        scan_row_group(
            shard,
            row_group_index=0,
            matcher=EvidenceMatcher([]),
            output_path=tmp_path / "partition.jsonl",
        )


def test_scan_row_groups_preserves_global_row_indices(tmp_path: Path) -> None:
    shard = write_multi_group_shard(tmp_path / "multi-group.parquet")
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    stats = scan_row_groups(
        shard,
        row_group_indices=(0, 1),
        matcher=matcher,
        output_path=output,
    )

    assert stats.rows_scanned == 4
    assert stats.matches_written == 2
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["fineweb_row_index"] for record in records] == [0, 3]

    second_output = tmp_path / "second-partition.jsonl"
    second_stats = scan_row_groups(
        shard,
        row_group_indices=(1,),
        matcher=matcher,
        output_path=second_output,
    )
    assert second_stats == ScanStats(rows_scanned=2, matches_written=1)
    second_record = json.loads(second_output.read_text(encoding="utf-8"))
    assert second_record["fineweb_row_index"] == 3


def test_scan_row_group_forwards_batch_size(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_scan_row_groups(*args, **kwargs) -> ScanStats:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return ScanStats(rows_scanned=1, matches_written=0)

    import fineweb_polygons.scanning as scanning_module

    monkeypatch.setattr(scanning_module, "scan_row_groups", fake_scan_row_groups)

    result = scan_row_group(
        tmp_path / "shard.parquet",
        row_group_index=3,
        matcher=EvidenceMatcher([]),
        output_path=tmp_path / "partition.jsonl",
        batch_size=7,
    )

    assert result == ScanStats(rows_scanned=1, matches_written=0)
    assert captured["args"] == (tmp_path / "shard.parquet",)
    assert captured["kwargs"]["row_group_indices"] == (3,)
    assert captured["kwargs"]["batch_size"] == 7


def test_scan_row_groups_rejects_empty_and_noncontiguous_selection() -> None:
    with pytest.raises(ValueError, match=r"^row_group_indices must not be empty$"):
        _select_row_groups(())
    with pytest.raises(
        ValueError, match=r"^row_group_indices must be sorted and contiguous$"
    ):
        _select_row_groups((0, 2))
    assert _select_row_groups((2, 3)) == (2, 3)


def test_projected_columns_include_optional_id_and_report_all_missing_columns(
    tmp_path: Path,
) -> None:
    with_id = tmp_path / "with-id.parquet"
    without_id = tmp_path / "without-id.parquet"
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text": ["x"], "url": ["y"], "id": ["z"]}), with_id)
    pq.write_table(pa.table({"text": ["x"], "url": ["y"]}), without_id)
    pq.write_table(pa.table({"title": ["x"]}), missing)

    assert _projected_columns(pq.ParquetFile(with_id)) == ["text", "url", "id"]
    assert _projected_columns(pq.ParquetFile(without_id)) == ["text", "url"]
    with pytest.raises(ValueError, match=r"missing text, url"):
        _projected_columns(pq.ParquetFile(missing))


def test_row_group_bounds_reject_negative_and_past_end_indices(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path / "shard.parquet")
    parquet_file = pq.ParquetFile(shard)

    with pytest.raises(
        IndexError, match=r"^row_group_indices contains an invalid row-group index$"
    ):
        _validate_row_group_bounds(parquet_file, (-1,))
    with pytest.raises(
        IndexError, match=r"^row_group_indices contains an invalid row-group index$"
    ):
        _validate_row_group_bounds(parquet_file, (1,))
    assert _row_start(parquet_file, 1) == 2


def test_scan_batches_passes_the_bounded_reader_options() -> None:
    captured = {}

    class FakeParquet:
        def iter_batches(self, **kwargs):
            captured.update(kwargs)
            return iter(())

    stats = _scan_batches(
        cast(pq.ParquetFile, FakeParquet()),
        selected_row_groups=(2, 3),
        projected_columns=("text", "url"),
        row_start=7,
        matcher=EvidenceMatcher([]),
        output=StringIO(),
        batch_size=11,
    )

    assert stats == ScanStats(rows_scanned=0, matches_written=0)
    assert captured == {
        "batch_size": 11,
        "row_groups": [2, 3],
        "columns": ["text", "url"],
        "use_threads": True,
    }


def test_scan_batches_accumulates_multiple_batches(tmp_path: Path) -> None:
    shard = write_multi_group_shard(tmp_path / "multi-group.parquet")
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    stats = scan_row_groups(
        shard,
        row_group_indices=(0, 1),
        matcher=matcher,
        output_path=output,
        batch_size=2,
    )

    assert stats == ScanStats(rows_scanned=4, matches_written=2)
    records = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["fineweb_row_index"] for record in records] == [0, 3]


def test_scan_without_id_uses_null_document_ids(tmp_path: Path) -> None:
    shard = tmp_path / "without-id.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["Fontvieille is in Monaco."],
                "url": ["https://example.test/fontvieille"],
            }
        ),
        shard,
    )
    output = tmp_path / "partition.jsonl"

    scan_row_group(
        shard,
        row_group_index=0,
        matcher=EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")]),
        output_path=output,
    )

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["fineweb_document_id"] is None
    assert record["url"] == "https://example.test/fontvieille"


def test_scan_batch_requires_equal_projected_column_lengths(tmp_path: Path) -> None:
    class MismatchedBatch:
        num_rows = 2

        def to_pydict(self):
            return {"text": ["a", "b"], "url": ["u"]}

    with (
        tmp_path.joinpath("partition.jsonl").open("w", encoding="utf-8") as output,
        pytest.raises(ValueError, match=r"zip\(\) argument"),
    ):
        _scan_batch(
            MismatchedBatch(),
            row_start=0,
            matcher=EvidenceMatcher([]),
            output=output,
        )


def test_as_text_normalizes_null_and_non_string_values() -> None:
    assert _as_text(None) == ""
    assert _as_text(42) == "42"


def test_write_matches_preserves_unicode_and_sorted_json_keys(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Café")])
    document = FineWebDocument(1, "doc-1", "Café is in Monaco.", "")
    import fineweb_polygons.scanning as scanning_module

    original_dumps = scanning_module.json.dumps
    seen = {}

    def recording_dumps(value, *args, **kwargs):
        seen.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(scanning_module.json, "dumps", recording_dumps)

    with output.open("w", encoding="utf-8") as stream:
        assert _write_matches(document, matcher, stream) == 1

    line = output.read_text(encoding="utf-8").strip()
    assert "Café" in line
    assert line.startswith('{"context_fields"')
    assert seen["ensure_ascii"] is False
    assert seen["sort_keys"] is True


def test_scan_removes_temporary_output_after_a_failed_match(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path / "shard.parquet")
    output = tmp_path / "nested" / "deeper" / "partition.jsonl"

    class FailingMatcher:
        def match(self, document):
            raise RuntimeError("synthetic matcher failure")

    with pytest.raises(RuntimeError, match="synthetic matcher failure"):
        scan_row_group(
            shard,
            row_group_index=0,
            matcher=cast(EvidenceMatcher, FailingMatcher()),
            output_path=output,
        )

    assert not output.exists()
    assert not output.with_name(f".{output.name}.tmp").exists()


def test_scan_uses_explicit_utf8_for_temporary_output(
    monkeypatch, tmp_path: Path
) -> None:
    shard = write_fixture_shard(tmp_path / "shard.parquet")
    output = tmp_path / "partition.jsonl"
    original_open = Path.open
    calls = []

    def recording_open(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    scan_row_group(
        shard,
        row_group_index=0,
        matcher=EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")]),
        output_path=output,
    )

    temporary_calls = [call for call in calls if call[0].name == ".partition.jsonl.tmp"]
    assert temporary_calls
    assert temporary_calls[0][2]["encoding"] == "utf-8"


def test_scan_cleanup_preserves_an_open_failure(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "partition.jsonl"
    temporary_path = output.with_name(f".{output.name}.tmp")
    original_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == temporary_path:
            raise RuntimeError("synthetic open failure")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(RuntimeError, match="synthetic open failure"):
        _scan_partition(
            cast(pq.ParquetFile, None),
            selected_row_groups=(0,),
            projected_columns=("text", "url"),
            row_start=0,
            matcher=cast(EvidenceMatcher, None),
            output_path=output,
            batch_size=1,
        )
