"""Streaming lexical candidate generation for Direction 2."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fineweb_polygons.artifact_io import (
    atomic_json_write,
    atomic_text_output,
    deterministic_temporary_path,
    sha256_file,
    write_json_line,
)
from fineweb_polygons.direction2.card import render_dataset_card
from fineweb_polygons.direction2.matching import (
    AhoCorasickPolygonMatcher,
    count_unique_normalized_names,
)
from fineweb_polygons.direction2.models import (
    DIRECTION_VERSION,
    OUTPUT_COLUMNS,
    Direction2CountrySummary,
    Direction2RunConfig,
    Direction2RunSummary,
    PolygonNameMatch,
    PolygonRecord,
    PolygonSource,
)
from fineweb_polygons.direction2.polygons import read_polygon_records
from fineweb_polygons.direction2.sentences import (
    context_for_match,
    split_sentences,
)
from fineweb_polygons.normalization import NORMALIZATION_VERSION

_REQUIRED_COLUMNS = ("text", "url")
_OUTPUT_SCHEMA = pa.schema([(column, pa.string()) for column in OUTPUT_COLUMNS])


def run_direction2(config: Direction2RunConfig) -> Direction2RunSummary:
    """Run the lexical POC and atomically publish its result artifacts."""
    _validate_inputs(config)
    sources = _sources(config)
    output_paths = tuple(
        config.output_dir / f"{source.key}.parquet" for source in sources
    )
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with _log_file(config.log_path) as log:
        _log_event(log, "run_started", version=DIRECTION_VERSION)
        polygons = read_polygon_records(sources)
        matcher = AhoCorasickPolygonMatcher.build(polygons)
        _log_event(
            log,
            "polygons_loaded",
            names_indexed=matcher.names_indexed,
            polygons_read=len(polygons),
        )
        scan = _scan_fineweb(
            config.shard_path,
            matcher=matcher,
            output_paths=output_paths,
            batch_size=config.batch_size,
            output_batch_size=config.output_batch_size,
            log=log,
        )
        country_summaries = _country_summaries(
            sources, output_paths, polygons, scan.country_stats
        )
        summary = _summary(
            config=config,
            output_paths=output_paths,
            polygons=polygons,
            names_indexed=matcher.names_indexed,
            scan=scan,
            country_summaries=country_summaries,
        )
        manifest = _manifest(
            config=config,
            sources=sources,
            output_paths=output_paths,
            polygons=polygons,
            names_indexed=matcher.names_indexed,
            scan=scan,
            country_summaries=country_summaries,
        )
        atomic_json_write(config.manifest_path, manifest)
        _write_card(config.dataset_card_path, manifest)
        _log_event(log, "run_completed", **summary.to_record())
    return summary


def _sources(config: Direction2RunConfig) -> tuple[PolygonSource, ...]:
    return (
        PolygonSource("monaco", config.monaco_pbf),
        PolygonSource("liechtenstein", config.liechtenstein_pbf),
    )


def _validate_inputs(config: Direction2RunConfig) -> None:
    for path in (config.monaco_pbf, config.liechtenstein_pbf, config.shard_path):
        if not path.is_file():
            raise FileNotFoundError(path)


class _LogFile(AbstractContextManager[Any]):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._stream: Any = None

    def __enter__(self) -> Any:
        self._stream = self._path.open("w", encoding="utf-8")
        return self._stream

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._stream is not None:
            self._stream.close()


def _log_file(path: Path) -> _LogFile:
    return _LogFile(path)


def _log_event(stream: Any, event: str, **values: object) -> None:
    write_json_line(stream, {"event": event, **values})
    stream.flush()


class _ScanResult:
    def __init__(self) -> None:
        self.docs_scanned = 0
        self.matches_found = 0
        self.polygon_ids: set[str] = set()
        self.country_stats: dict[str, _CountryStats] = {}


class _CountryStats:
    def __init__(self) -> None:
        self.matches_found = 0
        self.polygon_ids: set[str] = set()


class _ParquetOutputs(AbstractContextManager["_ParquetOutputs"]):
    def __init__(self, paths: tuple[Path, ...], batch_size: int) -> None:
        self._states = {path.stem: _ParquetState(path, batch_size) for path in paths}

    def __enter__(self) -> _ParquetOutputs:
        for state in self._states.values():
            state.open()
        return self

    def add(self, source_key: str, row: dict[str, str], polygon_id: str) -> None:
        state = self._states[source_key]
        state.add(row, polygon_id)

    def stats(self, source_key: str) -> _CountryStats:
        state = self._states[source_key]
        return state.stats

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            for state in self._states.values():
                state.publish()
        else:
            for state in self._states.values():
                state.abort()


class _ParquetState:
    def __init__(self, path: Path, batch_size: int) -> None:
        self.path = path
        self.temporary = deterministic_temporary_path(path)
        self.batch_size = batch_size
        self.writer: Any = None
        self.rows: list[dict[str, str]] = []
        self.stats = _CountryStats()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self.writer = pq.ParquetWriter(
            self.temporary, _OUTPUT_SCHEMA, compression="zstd"
        )

    def add(self, row: dict[str, str], polygon_id: str) -> None:
        self.rows.append(row)
        self.stats.matches_found += 1
        self.stats.polygon_ids.add(polygon_id)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=_OUTPUT_SCHEMA)
        self.writer.write_table(table)
        self.rows.clear()

    def publish(self) -> None:
        self.flush()
        self.writer.close()
        os.replace(self.temporary, self.path)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
        self.temporary.unlink(missing_ok=True)


def _scan_fineweb(
    shard_path: Path,
    *,
    matcher: AhoCorasickPolygonMatcher,
    output_paths: tuple[Path, ...],
    batch_size: int,
    output_batch_size: int,
    log: Any,
) -> _ScanResult:
    parquet_file = pq.ParquetFile(shard_path)
    _require_columns(parquet_file)
    result = _ScanResult()
    with _ParquetOutputs(output_paths, output_batch_size) as outputs:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(_REQUIRED_COLUMNS),
            use_threads=True,
        ):
            _scan_batch(batch, matcher=matcher, outputs=outputs, result=result)
            _log_event(log, "progress", docs_scanned=result.docs_scanned)
        result.country_stats = {
            path.stem: outputs.stats(path.stem) for path in output_paths
        }
    return result


def _require_columns(parquet_file: pq.ParquetFile) -> None:
    available = set(parquet_file.schema_arrow.names)
    missing = set(_REQUIRED_COLUMNS) - available
    if missing:
        raise ValueError(
            "FineWeb shard must contain text and url columns; missing "
            + ", ".join(sorted(missing))
        )


def _scan_batch(
    batch: Any,
    *,
    matcher: AhoCorasickPolygonMatcher,
    outputs: _ParquetOutputs,
    result: _ScanResult,
) -> None:
    text_column = batch.column("text")
    url_column = batch.column("url")
    for index in range(batch.num_rows):
        text = _as_text(text_column[index].as_py())
        url = _as_text(url_column[index].as_py())
        result.docs_scanned += 1
        matches = matcher.find(text)
        if not matches:
            continue
        spans = split_sentences(text)
        for match in matches:
            row = _match_row(text, url, spans, match)
            outputs.add(match.polygon.source_key, row, match.polygon.polygon_id)
            result.matches_found += 1
            result.polygon_ids.add(match.polygon.polygon_id)


def _match_row(
    text: str,
    url: str,
    spans: tuple[Any, ...],
    match: PolygonNameMatch,
) -> dict[str, str]:
    window = context_for_match(text, spans, match_start=match.start)
    polygon = match.polygon
    return {
        "polygon_id": polygon.polygon_id,
        "polygon_name": polygon.name,
        "matched_alias": match.matched_alias,
        "osm_tags": polygon.tags_as_json(),
        "centroid": polygon.centroid_as_json(),
        "fineweb_url": url,
        "sentence": window.sentence,
        "context": window.context,
    }


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _country_summaries(
    sources: tuple[PolygonSource, ...],
    output_paths: tuple[Path, ...],
    polygons: tuple[PolygonRecord, ...],
    country_stats: Mapping[str, _CountryStats],
) -> tuple[Direction2CountrySummary, ...]:
    summaries = []
    for source, output_path in zip(sources, output_paths, strict=True):
        source_polygons = tuple(
            polygon for polygon in polygons if polygon.source_key == source.key
        )
        stats = country_stats[source.key]
        summaries.append(
            Direction2CountrySummary(
                source_key=source.key,
                output_path=output_path,
                polygons_read=len(source_polygons),
                names_indexed=count_unique_normalized_names(source_polygons),
                matches_found=stats.matches_found,
                unique_polygons_matched=len(stats.polygon_ids),
                result_sha256=sha256_file(output_path),
            )
        )
    return tuple(summaries)


def _summary(
    *,
    config: Direction2RunConfig,
    output_paths: tuple[Path, ...],
    polygons: tuple[PolygonRecord, ...],
    names_indexed: int,
    scan: _ScanResult,
    country_summaries: tuple[Direction2CountrySummary, ...],
) -> Direction2RunSummary:
    return Direction2RunSummary(
        output_paths=output_paths,
        manifest_path=config.manifest_path,
        dataset_card_path=config.dataset_card_path,
        log_path=config.log_path,
        polygons_read=len(polygons),
        names_indexed=names_indexed,
        fineweb_docs_scanned=scan.docs_scanned,
        matches_found=scan.matches_found,
        unique_polygons_matched=len(scan.polygon_ids),
        country_summaries=country_summaries,
    )


def _manifest(
    *,
    config: Direction2RunConfig,
    sources: tuple[PolygonSource, ...],
    output_paths: tuple[Path, ...],
    polygons: tuple[PolygonRecord, ...],
    names_indexed: int,
    scan: _ScanResult,
    country_summaries: tuple[Direction2CountrySummary, ...],
) -> dict[str, object]:
    return {
        "configuration": {
            "batch_size": config.batch_size,
            "geographic_disambiguation": False,
            "matcher": "Aho-Corasick",
            "normalization_version": NORMALIZATION_VERSION,
            "output_batch_size": config.output_batch_size,
            "sentence_context": "matching sentence plus one sentence on each side",
            "thematic_filtering": False,
        },
        "countries": {
            summary.source_key: summary.to_record() for summary in country_summaries
        },
        "direction": DIRECTION_VERSION,
        "polygon_inventory": {
            "named_polygons": sum(
                bool(polygon.candidate_names()) for polygon in polygons
            ),
            "names_indexed": names_indexed,
            "polygons_read": len(polygons),
        },
        "results": {
            "files": [
                {
                    "path": f"data/direction-2/lexical-v1/{source.key}.parquet",
                    "sha256": summary.result_sha256,
                    "source_key": source.key,
                }
                for source, summary in zip(sources, country_summaries, strict=True)
            ],
            "fineweb_docs_scanned": scan.docs_scanned,
            "matches_found": scan.matches_found,
            "unique_polygons_matched": len(scan.polygon_ids),
        },
        "schema": list(OUTPUT_COLUMNS),
        "sources": {
            "fineweb_shard": {
                "path": str(config.shard_path),
                "sha256": sha256_file(config.shard_path),
            },
            "osm_pbf": [
                {
                    "path": str(source.path),
                    "sha256": sha256_file(source.path),
                    "source_key": source.key,
                }
                for source in sources
            ],
        },
        "status": "complete",
    }


def _write_card(path: Path, manifest: Mapping[str, object]) -> None:
    with atomic_text_output(
        path, temporary_factory=deterministic_temporary_path
    ) as output:
        output.write(render_dataset_card(manifest))
