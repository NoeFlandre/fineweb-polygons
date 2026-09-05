"""Two-pass specificity-aware candidate generation for Direction 2 V2."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fineweb_polygons.artifact_io import (
    atomic_json_write,
    atomic_text_output,
    deterministic_temporary_path,
    read_json_object,
    sha256_file,
    write_json_line,
)
from fineweb_polygons.direction2.matching import AhoCorasickPatternMatcher
from fineweb_polygons.direction2.models import PolygonRecord, PolygonSource
from fineweb_polygons.direction2.polygons import read_polygon_records
from fineweb_polygons.direction2.sentences import (
    SentenceSpan,
    context_for_match,
    split_sentences,
)
from fineweb_polygons.direction2.v2_card import render_dataset_card
from fineweb_polygons.direction2.v2_matching import (
    V2NameMatch,
    V2NameMatcher,
    has_independent_country_match,
)
from fineweb_polygons.direction2.v2_models import (
    COUNTRY_NAMES,
    DIRECTION_V2_VERSION,
    OUTPUT_COLUMNS_V2,
    Direction2V2CountrySummary,
    Direction2V2RunConfig,
    Direction2V2RunSummary,
)
from fineweb_polygons.direction2.v2_specificity import (
    NameProfile,
    build_name_inventory,
    searchable_name_patterns,
)
from fineweb_polygons.normalization import NORMALIZATION_VERSION

_REQUIRED_COLUMNS = ("text", "url")
_OUTPUT_SCHEMA = pa.schema(
    [
        *[(column, pa.string()) for column in OUTPUT_COLUMNS_V2[:8]],
        ("name_match_class", pa.string()),
        ("osm_polygon_count", pa.int64()),
        ("fineweb_document_frequency", pa.int64()),
    ]
)


@dataclass(frozen=True, slots=True)
class _V2Source:
    key: str
    path: Path
    country_name: str


@dataclass(frozen=True, slots=True)
class _FrequencyResult:
    frequencies: dict[str, int]
    documents_scanned: int
    reused: bool


@dataclass
class _CountryStats:
    matches_found: int = 0
    distinctive_matches: int = 0
    generic_matches: int = 0
    polygon_ids: set[str] = field(default_factory=set)


@dataclass
class _ScanResult:
    documents_scanned: int = 0
    matches_found: int = 0
    distinctive_matches: int = 0
    generic_matches: int = 0
    polygon_ids: set[str] = field(default_factory=set)
    country_stats: dict[str, _CountryStats] = field(default_factory=dict)


def run_direction2_v2(config: Direction2V2RunConfig) -> Direction2V2RunSummary:
    """Run the V2 frequency pass and country-gated retrieval pass."""
    _validate_inputs(config)
    sources = _sources(config)
    output_paths = tuple(
        config.output_dir / f"{source.key}.parquet" for source in sources
    )
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("w", encoding="utf-8") as log:
        _log_event(log, "run_started", version=DIRECTION_V2_VERSION)
        polygons = read_polygon_records(
            tuple(PolygonSource(source.key, source.path) for source in sources)
        )
        fingerprints = _fingerprints(config, sources)
        initial_profiles = build_name_inventory(
            polygons,
            document_frequencies={},
            document_count=0,
            country_names=COUNTRY_NAMES,
        )
        frequency = _frequency_result(
            config,
            fingerprints=fingerprints,
            patterns=searchable_name_patterns(initial_profiles),
            log=log,
        )
        profiles = build_name_inventory(
            polygons,
            document_frequencies=frequency.frequencies,
            document_count=frequency.documents_scanned,
            country_names=COUNTRY_NAMES,
        )
        _write_inventory(
            config.name_inventory_path,
            profiles,
            fingerprints=fingerprints,
            documents_scanned=frequency.documents_scanned,
        )
        _log_event(
            log,
            "names_loaded",
            names_considered=len(profiles),
            names_discarded=sum(
                profile.decision.decision == "discard" for profile in profiles
            ),
            names_indexed=sum(
                profile.decision.decision != "discard" for profile in profiles
            ),
            frequency_pass_reused=frequency.reused,
        )
        scan = _scan_matches(
            config.shard_path,
            profiles=profiles,
            sources=sources,
            output_paths=output_paths,
            batch_size=config.batch_size,
            output_batch_size=config.output_batch_size,
            log=log,
        )
        country_summaries = _country_summaries(
            sources,
            output_paths,
            polygons,
            profiles,
            scan.country_stats,
        )
        summary = _summary(
            config=config,
            output_paths=output_paths,
            profiles=profiles,
            polygons_read=len(polygons),
            frequency=frequency,
            scan=scan,
            country_summaries=country_summaries,
        )
        manifest = _manifest(
            config=config,
            sources=sources,
            fingerprints=fingerprints,
            profiles=profiles,
            polygons_read=len(polygons),
            frequency=frequency,
            scan=scan,
            country_summaries=country_summaries,
        )
        atomic_json_write(config.manifest_path, manifest)
        _write_card(config.dataset_card_path, manifest)
        _log_event(log, "run_completed", **summary.to_record())
    return summary


def _sources(config: Direction2V2RunConfig) -> tuple[_V2Source, ...]:
    return (
        _V2Source("monaco", config.monaco_pbf, COUNTRY_NAMES["monaco"]),
        _V2Source(
            "liechtenstein",
            config.liechtenstein_pbf,
            COUNTRY_NAMES["liechtenstein"],
        ),
    )


def _validate_inputs(config: Direction2V2RunConfig) -> None:
    for path in (config.monaco_pbf, config.liechtenstein_pbf, config.shard_path):
        if not path.is_file():
            raise FileNotFoundError(path)


def _fingerprints(
    config: Direction2V2RunConfig,
    sources: tuple[_V2Source, ...],
) -> dict[str, object]:
    return {
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
    }


def _policy_record() -> dict[str, object]:
    return {
        "fineweb_document_frequency_ratio": 0.001,
        "generic_osm_polygon_count_threshold": 1,
        "generic_requires_country_in_same_sentence": True,
        "minimum_name_letters": 3,
        "normalization_version": NORMALIZATION_VERSION,
        "short_single_token_max_letters": 8,
    }


def _frequency_result(
    config: Direction2V2RunConfig,
    *,
    fingerprints: Mapping[str, object],
    patterns: tuple[str, ...],
    log: Any,
) -> _FrequencyResult:
    cached = _read_cached_frequencies(
        config.name_inventory_path,
        fingerprints=fingerprints,
    )
    if cached is not None:
        return _FrequencyResult(
            frequencies=cached[0],
            documents_scanned=cached[1],
            reused=True,
        )
    frequencies, documents_scanned = _count_document_frequencies(
        config.shard_path,
        patterns=patterns,
        batch_size=config.batch_size,
        log=log,
    )
    return _FrequencyResult(
        frequencies=frequencies,
        documents_scanned=documents_scanned,
        reused=False,
    )


def _read_cached_frequencies(
    path: Path,
    *,
    fingerprints: Mapping[str, object],
) -> tuple[dict[str, int], int] | None:
    record = read_json_object(path)
    if not isinstance(record, Mapping) or not _cache_header_matches(
        record, fingerprints
    ):
        return None
    return _cached_frequency_payload(record)


def _cached_frequency_payload(
    record: Mapping[str, object],
) -> tuple[dict[str, int], int] | None:
    documents_scanned = record.get("fineweb_docs_scanned")
    names = record.get("names")
    if not isinstance(documents_scanned, int) or not isinstance(names, list):
        return None
    frequencies = _cached_frequency_map(names)
    return None if frequencies is None else (frequencies, documents_scanned)


def _cache_header_matches(
    record: Mapping[str, object],
    fingerprints: Mapping[str, object],
) -> bool:
    return (
        record.get("status") == "complete"
        and record.get("direction") == DIRECTION_V2_VERSION
        and record.get("inputs") == fingerprints
        and record.get("policy") == _policy_record()
    )


def _cached_frequency_map(
    names: list[object],
) -> dict[str, int] | None:
    frequencies: dict[str, int] = {}
    for value in names:
        item = _cached_frequency_item(value)
        if item is None:
            return None
        normalized_name, document_frequency = item
        frequencies[normalized_name] = document_frequency
    return frequencies


def _cached_frequency_item(value: object) -> tuple[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    normalized_name = value.get("normalized_name")
    decision = value.get("decision")
    if not isinstance(normalized_name, str) or not isinstance(decision, Mapping):
        return None
    document_frequency = decision.get("document_frequency")
    if not isinstance(document_frequency, int):
        return None
    return normalized_name, document_frequency


def _count_document_frequencies(
    shard_path: Path,
    *,
    patterns: tuple[str, ...],
    batch_size: int,
    log: Any,
) -> tuple[dict[str, int], int]:
    parquet_file = pq.ParquetFile(shard_path)
    _require_columns(parquet_file)
    matcher = AhoCorasickPatternMatcher.build(patterns)
    frequencies = dict.fromkeys(patterns, 0)
    documents_scanned = 0
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["text"],
        use_threads=True,
    ):
        for raw_text in batch.column("text").to_pylist():
            documents_scanned += 1
            for pattern in matcher.find_unique_patterns(_as_text(raw_text)):
                frequencies[pattern] += 1
        _log_event(log, "frequency_progress", docs_scanned=documents_scanned)
    return frequencies, documents_scanned


def _write_inventory(
    path: Path,
    profiles: tuple[NameProfile, ...],
    *,
    fingerprints: Mapping[str, object],
    documents_scanned: int,
) -> None:
    name_counts = _name_decision_counts(profiles)
    atomic_json_write(
        path,
        {
            "direction": DIRECTION_V2_VERSION,
            "fineweb_docs_scanned": documents_scanned,
            "inputs": dict(fingerprints),
            "names": [profile.to_record() for profile in profiles],
            "policy": _policy_record(),
            "status": "complete",
            "summary": {
                **name_counts,
                "names_considered": len(profiles),
            },
        },
    )


def _name_decision_counts(profiles: tuple[NameProfile, ...]) -> dict[str, int]:
    counts = Counter(profile.decision.decision for profile in profiles)
    return {
        "generic_names": counts["generic"],
        "names_discarded": counts["discard"],
        "names_indexed": counts["generic"] + counts["distinctive"],
    }


def _scan_matches(
    shard_path: Path,
    *,
    profiles: tuple[NameProfile, ...],
    sources: tuple[_V2Source, ...],
    output_paths: tuple[Path, ...],
    batch_size: int,
    output_batch_size: int,
    log: Any,
) -> _ScanResult:
    parquet_file = pq.ParquetFile(shard_path)
    _require_columns(parquet_file)
    matcher = V2NameMatcher.build(profiles)
    country_matchers = {
        source.key: AhoCorasickPatternMatcher.build((source.country_name,))
        for source in sources
    }
    result = _ScanResult()
    with _ParquetOutputs(
        tuple(source.key for source in sources),
        output_paths,
        output_batch_size,
    ) as outputs:
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(_REQUIRED_COLUMNS),
            use_threads=True,
        ):
            _scan_batch(
                batch,
                matcher=matcher,
                country_matchers=country_matchers,
                outputs=outputs,
                result=result,
            )
            _log_event(log, "match_progress", docs_scanned=result.documents_scanned)
        result.country_stats = {
            source.key: outputs.stats(source.key) for source in sources
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
    matcher: V2NameMatcher,
    country_matchers: Mapping[str, AhoCorasickPatternMatcher],
    outputs: _ParquetOutputs,
    result: _ScanResult,
) -> None:
    text_column = batch.column("text")
    url_column = batch.column("url")
    for index in range(batch.num_rows):
        result.documents_scanned += 1
        _scan_document(
            _as_text(text_column[index].as_py()),
            _as_text(url_column[index].as_py()),
            matcher=matcher,
            country_matchers=country_matchers,
            outputs=outputs,
            result=result,
        )


def _scan_document(
    text: str,
    url: str,
    *,
    matcher: V2NameMatcher,
    country_matchers: Mapping[str, AhoCorasickPatternMatcher],
    outputs: _ParquetOutputs,
    result: _ScanResult,
) -> None:
    matches = matcher.find(text)
    if not matches:
        return
    spans = split_sentences(text)
    for match in matches:
        if _write_document_match(
            text,
            url,
            spans,
            match,
            country_matchers=country_matchers,
            outputs=outputs,
        ):
            _record_match(result, match)


def _write_document_match(
    text: str,
    url: str,
    spans: tuple[SentenceSpan, ...],
    match: V2NameMatch,
    *,
    country_matchers: Mapping[str, AhoCorasickPatternMatcher],
    outputs: _ParquetOutputs,
) -> bool:
    span = _containing_span(spans, match.start)
    country_matcher = country_matchers[match.candidate.polygon.source_key]
    if not _keep_match(text, span, match, country_matcher=country_matcher):
        return False
    outputs.add(
        match.candidate.polygon.source_key,
        _match_row(text, url, spans, match),
        match.candidate.polygon.polygon_id,
        match.profile.decision.decision,
    )
    return True


def _record_match(result: _ScanResult, match: V2NameMatch) -> None:
    result.matches_found += 1
    result.polygon_ids.add(match.candidate.polygon.polygon_id)
    if match.profile.decision.decision == "generic":
        result.generic_matches += 1
    else:
        result.distinctive_matches += 1


def _containing_span(
    spans: tuple[SentenceSpan, ...],
    match_start: int,
) -> SentenceSpan:
    for span in spans:
        if span.start <= match_start < span.end:
            return span
    raise ValueError("match_start is outside the document sentences")


def _keep_match(
    text: str,
    span: SentenceSpan,
    match: V2NameMatch,
    *,
    country_matcher: AhoCorasickPatternMatcher,
) -> bool:
    if match.profile.decision.decision == "distinctive":
        return True
    sentence = text[span.start : span.end]
    country_matches = country_matcher.find(sentence)
    return has_independent_country_match(
        country_matches,
        name_start=match.start - span.start,
        name_end=match.end - span.start,
    )


def _match_row(
    text: str,
    url: str,
    spans: tuple[SentenceSpan, ...],
    match: V2NameMatch,
) -> dict[str, object]:
    window = context_for_match(text, spans, match_start=match.start)
    profile = match.profile
    polygon = match.candidate.polygon
    match_class = (
        "generic_name_with_country"
        if profile.decision.decision == "generic"
        else "distinctive_name"
    )
    return {
        "polygon_id": polygon.polygon_id,
        "polygon_name": polygon.name,
        "matched_alias": match.candidate.alias,
        "osm_tags": polygon.tags_as_json(),
        "centroid": polygon.centroid_as_json(),
        "fineweb_url": url,
        "sentence": window.sentence,
        "context": window.context,
        "name_match_class": match_class,
        "osm_polygon_count": profile.osm_polygon_count,
        "fineweb_document_frequency": profile.decision.document_frequency,
    }


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _country_summaries(
    sources: tuple[_V2Source, ...],
    output_paths: tuple[Path, ...],
    polygons: tuple[PolygonRecord, ...],
    profiles: tuple[NameProfile, ...],
    country_stats: Mapping[str, _CountryStats],
) -> tuple[Direction2V2CountrySummary, ...]:
    return tuple(
        _country_summary(
            source,
            output_path,
            polygons=polygons,
            profiles=profiles,
            stats=country_stats[source.key],
        )
        for source, output_path in zip(sources, output_paths, strict=True)
    )


def _country_summary(
    source: _V2Source,
    output_path: Path,
    *,
    polygons: tuple[PolygonRecord, ...],
    profiles: tuple[NameProfile, ...],
    stats: _CountryStats,
) -> Direction2V2CountrySummary:
    source_profiles = _profiles_for_source(profiles, source.key)
    return Direction2V2CountrySummary(
        source_key=source.key,
        output_path=output_path,
        polygons_read=sum(polygon.source_key == source.key for polygon in polygons),
        names_indexed=sum(
            profile.decision.decision != "discard" for profile in source_profiles
        ),
        matches_found=stats.matches_found,
        distinctive_matches=stats.distinctive_matches,
        generic_matches=stats.generic_matches,
        unique_polygons_matched=len(stats.polygon_ids),
        result_sha256=sha256_file(output_path),
    )


def _profiles_for_source(
    profiles: tuple[NameProfile, ...],
    source_key: str,
) -> tuple[NameProfile, ...]:
    return tuple(
        profile
        for profile in profiles
        if any(
            candidate.polygon.source_key == source_key
            for candidate in profile.candidates
        )
    )


def _summary(
    *,
    config: Direction2V2RunConfig,
    output_paths: tuple[Path, ...],
    profiles: tuple[NameProfile, ...],
    polygons_read: int,
    frequency: _FrequencyResult,
    scan: _ScanResult,
    country_summaries: tuple[Direction2V2CountrySummary, ...],
) -> Direction2V2RunSummary:
    name_counts = _name_decision_counts(profiles)
    return Direction2V2RunSummary(
        output_paths=output_paths,
        manifest_path=config.manifest_path,
        dataset_card_path=config.dataset_card_path,
        log_path=config.log_path,
        name_inventory_path=config.name_inventory_path,
        polygons_read=polygons_read,
        names_considered=len(profiles),
        names_indexed=name_counts["names_indexed"],
        names_discarded=name_counts["names_discarded"],
        generic_names=name_counts["generic_names"],
        fineweb_docs_frequency_pass=frequency.documents_scanned,
        fineweb_docs_match_pass=scan.documents_scanned,
        matches_found=scan.matches_found,
        distinctive_matches=scan.distinctive_matches,
        generic_matches=scan.generic_matches,
        unique_polygons_matched=len(scan.polygon_ids),
        country_summaries=country_summaries,
    )


def _manifest(
    *,
    config: Direction2V2RunConfig,
    sources: tuple[_V2Source, ...],
    fingerprints: Mapping[str, object],
    profiles: tuple[NameProfile, ...],
    polygons_read: int,
    frequency: _FrequencyResult,
    scan: _ScanResult,
    country_summaries: tuple[Direction2V2CountrySummary, ...],
) -> dict[str, object]:
    name_counts = _name_decision_counts(profiles)
    return {
        "configuration": {
            **_policy_record(),
            "batch_size": config.batch_size,
            "frequency_pass_reused": frequency.reused,
            "matcher": "Aho-Corasick",
            "output_batch_size": config.output_batch_size,
            "sentence_context": "matching sentence plus one sentence on each side",
            "url_is_condition": False,
        },
        "countries": {
            summary.source_key: summary.to_record() for summary in country_summaries
        },
        "direction": DIRECTION_V2_VERSION,
        "name_inventory": {
            "path": str(config.name_inventory_path),
            "sha256": sha256_file(config.name_inventory_path),
        },
        "polygon_inventory": {
            **name_counts,
            "names_considered": len(profiles),
            "polygons_read": polygons_read,
        },
        "results": {
            "files": [
                {
                    "path": f"data/direction-2/lexical-v2/{source.key}.parquet",
                    "sha256": summary.result_sha256,
                    "source_key": source.key,
                }
                for source, summary in zip(sources, country_summaries, strict=True)
            ],
            "fineweb_docs_frequency_pass": frequency.documents_scanned,
            "fineweb_docs_match_pass": scan.documents_scanned,
            "generic_matches": scan.generic_matches,
            "matches_found": scan.matches_found,
            "distinctive_matches": scan.distinctive_matches,
            "unique_polygons_matched": len(scan.polygon_ids),
        },
        "schema": list(OUTPUT_COLUMNS_V2),
        "sources": dict(fingerprints),
        "status": "complete",
    }


def _write_card(path: Path, manifest: Mapping[str, object]) -> None:
    with atomic_text_output(
        path,
        temporary_factory=deterministic_temporary_path,
    ) as output:
        output.write(render_dataset_card(manifest))


def _log_event(stream: Any, event: str, **values: object) -> None:
    write_json_line(stream, {"event": event, **values})
    stream.flush()


class _ParquetOutputs(AbstractContextManager["_ParquetOutputs"]):
    def __init__(
        self,
        source_keys: tuple[str, ...],
        paths: tuple[Path, ...],
        batch_size: int,
    ) -> None:
        self._states = {
            source_key: _ParquetState(path, batch_size)
            for source_key, path in zip(source_keys, paths, strict=True)
        }

    def __enter__(self) -> _ParquetOutputs:
        for state in self._states.values():
            state.open()
        return self

    def add(
        self,
        source_key: str,
        row: dict[str, object],
        polygon_id: str,
        match_type: str,
    ) -> None:
        self._states[source_key].add(row, polygon_id, match_type)

    def stats(self, source_key: str) -> _CountryStats:
        return self._states[source_key].stats

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            for state in self._states.values():
                state.publish()
        else:
            for state in self._states.values():
                state.abort()


class _ParquetState:
    """Own one country output, buffering rows before atomic publication."""

    def __init__(self, path: Path, batch_size: int) -> None:
        self.path = path
        self.temporary = deterministic_temporary_path(path)
        self.batch_size = batch_size
        self.writer: Any = None
        self.rows: list[dict[str, object]] = []
        self.stats = _CountryStats()

    def open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary.unlink(missing_ok=True)
        self.writer = pq.ParquetWriter(
            self.temporary,
            _OUTPUT_SCHEMA,
            compression="zstd",  # pragma: no mutate: PyArrow accepts case variants.
        )

    def add(self, row: dict[str, object], polygon_id: str, match_type: str) -> None:
        self.rows.append(row)
        self.stats.matches_found += 1
        self.stats.polygon_ids.add(polygon_id)
        if match_type == "generic":
            self.stats.generic_matches += 1
        else:
            self.stats.distinctive_matches += 1
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
