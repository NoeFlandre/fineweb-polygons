import json
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fineweb_polygons.scanning as scanning_module
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
    assert record["polygon_id"] == "way/1"
    assert record["text"] == "Fontvieille is in Monaco."


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


def test_scan_row_group_forwards_batch_size(monkeypatch, tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([])
    captured: dict[str, object] = {}
    expected = ScanStats(rows_scanned=3, matches_written=1)

    def fake_scan_row_groups(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(scanning_module, "scan_row_groups", fake_scan_row_groups)

    assert (
        scanning_module.scan_row_group(
            shard,
            row_group_index=3,
            matcher=matcher,
            output_path=output,
            batch_size=7,
        )
        == expected
    )
    assert captured == {
        "path": shard,
        "row_group_indices": (3,),
        "matcher": matcher,
        "output_path": output,
        "batch_size": 7,
    }


def test_scan_row_groups_starts_at_the_selected_row_group(tmp_path: Path) -> None:
    shard = write_multi_group_shard(tmp_path / "selected-group.parquet")
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    stats = scan_row_groups(
        shard,
        row_group_indices=(1,),
        matcher=matcher,
        output_path=output,
    )

    assert stats.rows_scanned == 2
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["fineweb_row_index"] == 3


def test_select_row_groups_rejects_empty_and_noncontiguous_inputs() -> None:
    with pytest.raises(ValueError, match=r"\Arow_group_indices must not be empty\Z"):
        _select_row_groups(())
    with pytest.raises(
        ValueError, match=r"\Arow_group_indices must be sorted and contiguous\Z"
    ):
        _select_row_groups((1, 3))


def test_projected_columns_keep_optional_ids_and_report_missing_columns(
    tmp_path: Path,
) -> None:
    with_id = tmp_path / "with-id.parquet"
    without_id = tmp_path / "without-id.parquet"
    missing = tmp_path / "missing.parquet"
    pq.write_table(pa.table({"text": ["x"], "url": ["y"], "id": ["z"]}), with_id)
    pq.write_table(pa.table({"text": ["x"], "url": ["y"]}), without_id)
    pq.write_table(pa.table({"text": ["x"]}), missing)

    assert _projected_columns(pq.ParquetFile(with_id)) == ["text", "url", "id"]
    assert _projected_columns(pq.ParquetFile(without_id)) == ["text", "url"]
    with pytest.raises(
        ValueError,
        match=(r"\AParquet shard must contain text and url columns; missing url\Z"),
    ):
        _projected_columns(pq.ParquetFile(missing))
    missing_both = tmp_path / "missing-both.parquet"
    pq.write_table(pa.table({"id": ["z"]}), missing_both)
    with pytest.raises(
        ValueError,
        match=(
            r"\AParquet shard must contain text and url columns; missing text, url\Z"
        ),
    ):
        _projected_columns(pq.ParquetFile(missing_both))


def test_row_group_bounds_and_starts_are_exact(tmp_path: Path) -> None:
    shard = write_multi_group_shard(tmp_path / "bounds.parquet")
    parquet_file = pq.ParquetFile(shard)

    assert _row_start(parquet_file, 1) == 2
    with pytest.raises(
        IndexError, match=r"\Arow_group_indices contains an invalid row-group index\Z"
    ):
        _validate_row_group_bounds(parquet_file, (-1,))
    with pytest.raises(
        IndexError, match=r"\Arow_group_indices contains an invalid row-group index\Z"
    ):
        _validate_row_group_bounds(parquet_file, (2,))


def test_scan_batches_forwards_bounded_reader_arguments_and_accumulates(
    monkeypatch,
) -> None:
    calls: list[dict[str, object]] = []
    row_starts: list[int] = []

    class Batch:
        def __init__(self, num_rows: int) -> None:
            self.num_rows = num_rows

    class FakeParquetFile:
        def iter_batches(self, **kwargs: object):
            calls.append(kwargs)
            return iter((Batch(2), Batch(3)))

    def fake_scan_batch(batch, *, row_start, matcher, output):
        del batch, matcher, output
        row_starts.append(row_start)
        return 1

    monkeypatch.setattr(scanning_module, "_scan_batch", fake_scan_batch)
    stats = _scan_batches(
        cast(Any, FakeParquetFile()),
        selected_row_groups=(2, 3),
        projected_columns=("text", "url"),
        row_start=10,
        matcher=cast(Any, object()),
        output=StringIO(),
        batch_size=7,
    )

    assert calls == [
        {
            "batch_size": 7,
            "row_groups": [2, 3],
            "columns": ["text", "url"],
            "use_threads": True,
        }
    ]
    assert row_starts == [10, 12]
    assert stats == ScanStats(rows_scanned=5, matches_written=2)


def test_scan_batch_handles_optional_ids_and_strict_lengths() -> None:
    documents: list[FineWebDocument] = []

    class RecordingMatcher:
        def match(self, document: FineWebDocument) -> tuple[object, ...]:
            documents.append(document)
            return ()

    class Batch:
        num_rows = 2

        def to_pydict(self):
            return {
                "text": [None, "Café"],
                "url": [None, "https://x"],
                "id": [None, 42],
            }

    assert (
        _scan_batch(
            cast(Any, Batch()),
            row_start=4,
            matcher=cast(Any, RecordingMatcher()),
            output=StringIO(),
        )
        == 0
    )
    assert documents == [
        FineWebDocument(4, None, "", ""),
        FineWebDocument(5, "42", "Café", "https://x"),
    ]

    class ShortIds:
        num_rows = 2

        def to_pydict(self):
            return {"text": ["a", "b"], "url": ["", ""], "id": ["only"]}

    with pytest.raises(ValueError, match=r"zip\(\) argument 3 is shorter"):
        _scan_batch(
            cast(Any, ShortIds()),
            row_start=0,
            matcher=cast(Any, RecordingMatcher()),
            output=StringIO(),
        )


def test_scan_batch_without_ids_uses_none_document_ids() -> None:
    documents: list[FineWebDocument] = []

    class RecordingMatcher:
        def match(self, document: FineWebDocument) -> tuple[object, ...]:
            documents.append(document)
            return ()

    class Batch:
        num_rows = 1

        def to_pydict(self):
            return {"text": ["text"], "url": ["url"]}

    _scan_batch(
        cast(Any, Batch()),
        row_start=0,
        matcher=cast(Any, RecordingMatcher()),
        output=StringIO(),
    )

    assert documents == [FineWebDocument(0, None, "text", "url")]


def test_write_matches_uses_stable_unicode_json(monkeypatch) -> None:
    class Match:
        def to_record(self):
            return {"z": "é", "a": 1}

    class Matcher:
        def match(self, document):
            del document
            return (Match(),)

    output = StringIO()
    ensure_ascii_values: list[object] = []
    real_dumps = scanning_module.json.dumps

    def dumps(value, *args, **kwargs):
        ensure_ascii_values.append(kwargs.get("ensure_ascii"))
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(scanning_module.json, "dumps", dumps)

    assert (
        _write_matches(
            FineWebDocument(0, None, "", ""),
            cast(Any, Matcher()),
            output,
        )
        == 1
    )
    assert output.getvalue() == '{"a": 1, "z": "é"}\n'
    assert ensure_ascii_values == [False]
    assert _as_text(None) == ""
    assert _as_text(42) == "42"


def test_scan_partition_creates_nested_output_and_cleans_missing_temp(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "deep" / "nested" / "partition.jsonl"
    parquet_file = cast(Any, object())
    open_calls: list[tuple[object, object]] = []
    real_open = Path.open

    def fake_scan_batches(*args, **kwargs):
        del args, kwargs
        return ScanStats(rows_scanned=1, matches_written=0)

    monkeypatch.setattr(scanning_module, "_scan_batches", fake_scan_batches)

    def open_file(self, mode="r", *args, **kwargs):
        if self == output.with_name(f".{output.name}.tmp"):
            open_calls.append((mode, kwargs.get("encoding")))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    stats = _scan_partition(
        parquet_file,
        selected_row_groups=(0,),
        projected_columns=("text", "url"),
        row_start=0,
        matcher=cast(Any, object()),
        output_path=output,
        batch_size=1,
    )
    assert stats == ScanStats(rows_scanned=1, matches_written=0)
    assert output.exists()
    assert open_calls == [("w", "utf-8")]

    missing_output = tmp_path / "missing" / "partition.jsonl"
    temporary_path = missing_output.with_name(f".{missing_output.name}.tmp")
    real_open = Path.open

    def fail_temporary_open(self, *args, **kwargs):
        if self == temporary_path:
            raise RuntimeError("open failed")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_temporary_open)
    with pytest.raises(RuntimeError, match="open failed"):
        _scan_partition(
            parquet_file,
            selected_row_groups=(0,),
            projected_columns=("text", "url"),
            row_start=0,
            matcher=cast(Any, object()),
            output_path=missing_output,
            batch_size=1,
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
