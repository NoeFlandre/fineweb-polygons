"""Bounded, atomic scanning of FineWeb Parquet row groups."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import pyarrow.parquet as pq

from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import FineWebDocument

_REQUIRED_COLUMNS = ("text", "url")
_DEFAULT_BATCH_SIZE = 8192


@dataclass(frozen=True, slots=True)
class ScanStats:
    """Counters produced by one resumable scan partition."""

    rows_scanned: int
    matches_written: int


def scan_row_group(
    shard_path: Path,
    *,
    row_group_index: int,
    matcher: EvidenceMatcher,
    output_path: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ScanStats:
    """Scan one Parquet row group and atomically write its JSONL matches."""
    return scan_row_groups(
        shard_path,
        row_group_indices=(row_group_index,),
        matcher=matcher,
        output_path=output_path,
        batch_size=batch_size,
    )


def scan_row_groups(
    shard_path: Path,
    *,
    row_group_indices: Sequence[int],
    matcher: EvidenceMatcher,
    output_path: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> ScanStats:
    """Scan contiguous row groups with one Parquet reader and atomic output."""
    selected_row_groups = _select_row_groups(row_group_indices)
    parquet_file = pq.ParquetFile(shard_path)
    projected_columns = _projected_columns(parquet_file)
    _validate_row_group_bounds(parquet_file, selected_row_groups)
    row_start = _row_start(parquet_file, selected_row_groups[0])
    return _scan_partition(
        parquet_file,
        selected_row_groups=selected_row_groups,
        projected_columns=projected_columns,
        row_start=row_start,
        matcher=matcher,
        output_path=output_path,
        batch_size=batch_size,
    )


def _select_row_groups(row_group_indices: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(row_group_indices)
    if not selected:
        raise ValueError("row_group_indices must not be empty")
    if selected != tuple(range(selected[0], selected[-1] + 1)):
        raise ValueError("row_group_indices must be sorted and contiguous")
    return selected


def _projected_columns(parquet_file: pq.ParquetFile) -> list[str]:
    columns = set(parquet_file.schema_arrow.names)
    missing_columns = set(_REQUIRED_COLUMNS) - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Parquet shard must contain text and url columns; missing {missing}"
        )
    return ["text", "url", *(["id"] if "id" in columns else [])]


def _validate_row_group_bounds(
    parquet_file: pq.ParquetFile, selected_row_groups: Sequence[int]
) -> None:
    row_group_count = parquet_file.metadata.num_row_groups
    if selected_row_groups[0] < 0 or selected_row_groups[-1] >= row_group_count:
        raise IndexError("row_group_indices contains an invalid row-group index")


def _row_start(parquet_file: pq.ParquetFile, row_group_index: int) -> int:
    return sum(
        parquet_file.metadata.row_group(index).num_rows
        for index in range(row_group_index)
    )


def _scan_partition(
    parquet_file: pq.ParquetFile,
    *,
    selected_row_groups: Sequence[int],
    projected_columns: Sequence[str],
    row_start: int,
    matcher: EvidenceMatcher,
    output_path: Path,
    batch_size: int,
) -> ScanStats:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            stats = _scan_batches(
                parquet_file,
                selected_row_groups=selected_row_groups,
                projected_columns=projected_columns,
                row_start=row_start,
                matcher=matcher,
                output=output,
                batch_size=batch_size,
            )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return stats


def _scan_batches(
    parquet_file: pq.ParquetFile,
    *,
    selected_row_groups: Sequence[int],
    projected_columns: Sequence[str],
    row_start: int,
    matcher: EvidenceMatcher,
    output: TextIO,
    batch_size: int,
) -> ScanStats:
    rows_scanned = 0
    matches_written = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        row_groups=list(selected_row_groups),
        columns=list(projected_columns),
        use_threads=True,
    ):
        batch_matches = _scan_batch(
            batch,
            row_start=row_start + rows_scanned,
            matcher=matcher,
            output=output,
        )
        rows_scanned += batch.num_rows
        matches_written += batch_matches
    return ScanStats(rows_scanned=rows_scanned, matches_written=matches_written)


def _scan_batch(
    batch: Any,
    *,
    row_start: int,
    matcher: EvidenceMatcher,
    output: TextIO,
) -> int:
    values = batch.to_pydict()
    ids = values.get("id", [None] * batch.num_rows)
    matches_written = 0
    for offset, (raw_text, raw_url, raw_id) in enumerate(
        zip(values["text"], values["url"], ids, strict=True)
    ):
        document = FineWebDocument(
            row_index=row_start + offset,
            document_id=None if raw_id is None else str(raw_id),
            text=_as_text(raw_text),
            url=_as_text(raw_url),
        )
        matches_written += _write_matches(document, matcher, output)
    return matches_written


def _write_matches(
    document: FineWebDocument, matcher: EvidenceMatcher, output: TextIO
) -> int:
    matches = matcher.match(document)
    for match in matches:
        output.write(json.dumps(match.to_record(), ensure_ascii=False, sort_keys=True))
        output.write("\n")
    return len(matches)


def _as_text(value: object) -> str:
    return "" if value is None else str(value)
