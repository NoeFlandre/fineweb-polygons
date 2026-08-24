"""Bounded, atomic scanning of FineWeb Parquet row groups."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import FineWebDocument

_REQUIRED_COLUMNS = ("text", "url")
_DEFAULT_BATCH_SIZE = 8192


@dataclass(frozen=True, slots=True)
class ScanStats:
    """Counters produced by one row-group scan."""

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
    parquet_file = pq.ParquetFile(shard_path)
    columns = set(parquet_file.schema_arrow.names)
    missing_columns = set(_REQUIRED_COLUMNS) - columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Parquet shard must contain text and url columns; missing {missing}"
        )

    projected_columns = ["text", "url"]
    if "id" in columns:
        projected_columns.append("id")
    row_start = sum(
        parquet_file.metadata.row_group(index).num_rows
        for index in range(row_group_index)
    )
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_scanned = 0
    matches_written = 0
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                row_groups=[row_group_index],
                columns=projected_columns,
                use_threads=True,
            ):
                values = batch.to_pydict()
                ids = values.get("id", [None] * batch.num_rows)
                for offset, (raw_text, raw_url, raw_id) in enumerate(
                    zip(values["text"], values["url"], ids, strict=True)
                ):
                    document = FineWebDocument(
                        row_index=row_start + rows_scanned + offset,
                        document_id=None if raw_id is None else str(raw_id),
                        text=_as_text(raw_text),
                        url=_as_text(raw_url),
                    )
                    matches = matcher.match(document)
                    for match in matches:
                        output.write(
                            json.dumps(
                                match.to_record(),
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        )
                        output.write("\n")
                    matches_written += len(matches)
                rows_scanned += batch.num_rows
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return ScanStats(rows_scanned=rows_scanned, matches_written=matches_written)


def _as_text(value: object) -> str:
    return "" if value is None else str(value)
