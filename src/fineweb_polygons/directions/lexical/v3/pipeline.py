"""Two-pass scored candidate generation for Direction 2 lexical V3."""

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

from fineweb_polygons.core.artifact_io import (
    atomic_json_write,
    atomic_text_output,
    deterministic_temporary_path,
    read_json_object,
    sha256_file,
    write_json_line,
)
from fineweb_polygons.core.normalization import NORMALIZATION_VERSION
from fineweb_polygons.directions.lexical.matching import AhoCorasickPatternMatcher
from fineweb_polygons.directions.lexical.models import PolygonRecord, PolygonSource
from fineweb_polygons.directions.lexical.osm import read_polygon_records
from fineweb_polygons.directions.lexical.sentences import (
    SentenceSpan,
    context_for_match,
    split_sentences,
)
from fineweb_polygons.directions.lexical.v2.specificity import (
    NameProfile,
    build_name_inventory,
    searchable_name_patterns,
)
from fineweb_polygons.directions.lexical.v3.card import render_dataset_card
from fineweb_polygons.directions.lexical.v3.evidence import (
    CandidateEvidence,
    score_candidate,
)
from fineweb_polygons.directions.lexical.v3.matching import V3NameMatch, V3NameMatcher
from fineweb_polygons.directions.lexical.v3.models import (
    COUNTRY_NAMES,
    DATA_PREFIX,
    DIRECTION_V3_VERSION,
    OUTPUT_COLUMNS_V3,
    Direction2V3CountrySummary,
    Direction2V3RunConfig,
    Direction2V3RunSummary,
)

_REQUIRED_COLUMNS = ("text", "url")
_OUTPUT_SCHEMA = pa.schema(
    [
        *[(column, pa.string()) for column in OUTPUT_COLUMNS_V3[:8]],
        ("name_match_class", pa.string()),
        ("osm_polygon_count", pa.int64()),
        ("fineweb_document_frequency", pa.int64()),
        ("decision_tier", pa.string()),
        ("evidence_score", pa.int64()),
        ("evidence_reasons", pa.string()),
        ("name_in_url", pa.bool_()),
        ("country_in_sentence", pa.bool_()),
        ("country_in_context", pa.bool_()),
        ("same_polygon_alias_nearby", pa.bool_()),
        ("other_polygon_name_nearby", pa.bool_()),
    ]
)


@dataclass(frozen=True, slots=True)
class _V3Source:
    key: str
    path: Path


@dataclass(frozen=True, slots=True)
class _FrequencyResult:
    frequencies: dict[str, int]
    documents_scanned: int
    reused: bool


@dataclass
class _CountryStats:
    matches_found: int = 0
    high_confidence_matches: int = 0
    possible_matches: int = 0
    rejected_matches: int = 0
    polygon_ids: set[str] = field(default_factory=set)


@dataclass
class _ScanResult:
    documents_scanned: int = 0
    matches_found: int = 0
    high_confidence_matches: int = 0
    possible_matches: int = 0
    rejected_matches: int = 0
    polygon_ids: set[str] = field(default_factory=set)
    country_stats: dict[str, _CountryStats] = field(default_factory=dict)


def run_direction2_v3(config: Direction2V3RunConfig) -> Direction2V3RunSummary:
    """Run the V3 frequency pass and scored retrieval pass."""
    _validate_inputs(config)
    sources = _sources(config)
    output_paths = tuple(
        config.output_dir / f"{source.key}.parquet" for source in sources
    )
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    with config.log_path.open("w", encoding="utf-8") as log:
        _log_event(log, "run_started", version=DIRECTION_V3_VERSION)
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


def _sources(config: Direction2V3RunConfig) -> tuple[_V3Source, ...]:
    return (
        _V3Source("monaco", config.monaco_pbf),
        _V3Source("liechtenstein", config.liechtenstein_pbf),
    )


def _validate_inputs(config: Direction2V3RunConfig) -> None:
    for path in (config.monaco_pbf, config.liechtenstein_pbf, config.shard_path):
        if not path.is_file():
            raise FileNotFoundError(path)


def _fingerprints(
    config: Direction2V3RunConfig,
    sources: tuple[_V3Source, ...],
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
        "normalization_version": NORMALIZATION_VERSION,
        "tier_thresholds": {"high_confidence": 4, "possible": 2},
    }


def _frequency_result(
    config: Direction2V3RunConfig,
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
        return _FrequencyResult(cached[0], cached[1], reused=True)
    frequencies, documents_scanned = _count_document_frequencies(
        config.shard_path,
        patterns=patterns,
        batch_size=config.batch_size,
        log=log,
    )
    return _FrequencyResult(frequencies, documents_scanned, reused=False)


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
    documents_scanned = record.get("fineweb_docs_scanned")
    names = record.get("names")
    payload = _cache_payload(documents_scanned, names)
    if payload is None:
        return None
    cached_names, cached_documents = payload
    frequencies = _cached_frequencies(cached_names)
    if frequencies is None:
        return None
    return frequencies, cached_documents


def _cache_payload(
    documents_scanned: object,
    names: object,
) -> tuple[list[object], int] | None:
    if not isinstance(documents_scanned, int) or not isinstance(names, list):
        return None
    return names, documents_scanned


def _cached_frequencies(names: list[object]) -> dict[str, int] | None:
    frequencies: dict[str, int] = {}
    for value in names:
        item = _cached_frequency(value)
        if item is None:
            return None
        normalized_name, document_frequency = item
        frequencies[normalized_name] = document_frequency
    return frequencies


def _cached_frequency(value: object) -> tuple[str, int] | None:
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


def _cache_header_matches(
    record: Mapping[str, object],
    fingerprints: Mapping[str, object],
) -> bool:
    return (
        record.get("status") == "complete"
        and record.get("direction") == DIRECTION_V3_VERSION
        and record.get("inputs") == fingerprints
        and record.get("policy") == _policy_record()
    )


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
    counts = Counter(profile.decision.decision for profile in profiles)
    atomic_json_write(
        path,
        {
            "direction": DIRECTION_V3_VERSION,
            "fineweb_docs_scanned": documents_scanned,
            "inputs": dict(fingerprints),
            "names": [profile.to_record() for profile in profiles],
            "policy": _policy_record(),
            "status": "complete",
            "summary": {
                "generic_names": counts["generic"],
                "names_considered": len(profiles),
                "names_discarded": counts["discard"],
                "names_indexed": counts["generic"] + counts["distinctive"],
            },
        },
    )


def _scan_matches(
    shard_path: Path,
    *,
    profiles: tuple[NameProfile, ...],
    sources: tuple[_V3Source, ...],
    output_paths: tuple[Path, ...],
    batch_size: int,
    output_batch_size: int,
    log: Any,
) -> _ScanResult:
    parquet_file = pq.ParquetFile(shard_path)
    _require_columns(parquet_file)
    matcher = V3NameMatcher.build(profiles)
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
            _scan_batch(batch, matcher=matcher, outputs=outputs, result=result)
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
    matcher: V3NameMatcher,
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
            outputs=outputs,
            result=result,
        )


def _scan_document(
    text: str,
    url: str,
    *,
    matcher: V3NameMatcher,
    outputs: _ParquetOutputs,
    result: _ScanResult,
) -> None:
    matches = matcher.find(text)
    if not matches:
        return
    spans = split_sentences(text)
    for index, match in enumerate(matches):
        window = context_for_match(text, spans, match_start=match.start)
        same_alias, other_name = _nearby_evidence(spans, matches, index)
        evidence = score_candidate(
            match.profile,
            alias=match.candidate.alias,
            sentence=window.sentence,
            context=window.context,
            url=url,
            country_name=COUNTRY_NAMES[match.candidate.polygon.source_key],
            same_polygon_alias_nearby=same_alias,
            other_polygon_name_nearby=other_name,
        )
        outputs.add(
            match.candidate.polygon.source_key,
            _match_row(text, url, spans, match, evidence),
            match.candidate.polygon.polygon_id,
            evidence,
        )
        _record_match(result, match, evidence)


def _nearby_evidence(
    spans: tuple[SentenceSpan, ...],
    matches: tuple[V3NameMatch, ...],
    current_index: int,
) -> tuple[bool, bool]:
    current = matches[current_index]
    context_start, context_end = _context_bounds(spans, current.start)
    nearby = tuple(
        candidate
        for index, candidate in enumerate(matches)
        if _is_nearby(index, current_index, candidate, context_start, context_end)
    )
    return (
        any(_is_same_polygon_alias(current, candidate) for candidate in nearby),
        any(_is_other_polygon_name(current, candidate) for candidate in nearby),
    )


def _is_nearby(
    index: int,
    current_index: int,
    candidate: V3NameMatch,
    context_start: int,
    context_end: int,
) -> bool:
    return index != current_index and context_start <= candidate.start < context_end


def _is_same_polygon_alias(current: V3NameMatch, candidate: V3NameMatch) -> bool:
    return (
        candidate.candidate.polygon.polygon_id == current.candidate.polygon.polygon_id
        and candidate.profile.normalized_name != current.profile.normalized_name
    )


def _is_other_polygon_name(current: V3NameMatch, candidate: V3NameMatch) -> bool:
    return (
        candidate.candidate.polygon.source_key == current.candidate.polygon.source_key
        and candidate.candidate.polygon.polygon_id
        != current.candidate.polygon.polygon_id
    )


def _context_bounds(
    spans: tuple[SentenceSpan, ...],
    match_start: int,
) -> tuple[int, int]:
    sentence_index = next(
        index
        for index, span in enumerate(spans)
        if span.start <= match_start < span.end
    )
    first = max(0, sentence_index - 1)
    last = min(len(spans) - 1, sentence_index + 1)
    return spans[first].start, spans[last].end


def _match_row(
    text: str,
    url: str,
    spans: tuple[SentenceSpan, ...],
    match: V3NameMatch,
    evidence: CandidateEvidence,
) -> dict[str, object]:
    window = context_for_match(text, spans, match_start=match.start)
    profile = match.profile
    polygon = match.candidate.polygon
    match_class = (
        "generic_name" if profile.decision.decision == "generic" else "distinctive_name"
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
        **evidence.to_record(),
    }


def _record_match(
    result: _ScanResult,
    match: V3NameMatch,
    evidence: CandidateEvidence,
) -> None:
    result.matches_found += 1
    result.polygon_ids.add(match.candidate.polygon.polygon_id)
    if evidence.tier == "high_confidence":
        result.high_confidence_matches += 1
    elif evidence.tier == "possible":
        result.possible_matches += 1
    else:
        result.rejected_matches += 1


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _country_summaries(
    sources: tuple[_V3Source, ...],
    output_paths: tuple[Path, ...],
    polygons: tuple[PolygonRecord, ...],
    profiles: tuple[NameProfile, ...],
    country_stats: Mapping[str, _CountryStats],
) -> tuple[Direction2V3CountrySummary, ...]:
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
    source: _V3Source,
    output_path: Path,
    *,
    polygons: tuple[PolygonRecord, ...],
    profiles: tuple[NameProfile, ...],
    stats: _CountryStats,
) -> Direction2V3CountrySummary:
    source_profiles = _profiles_for_source(profiles, source.key)
    return Direction2V3CountrySummary(
        source_key=source.key,
        output_path=output_path,
        polygons_read=sum(polygon.source_key == source.key for polygon in polygons),
        names_indexed=sum(
            profile.decision.decision != "discard" for profile in source_profiles
        ),
        matches_found=stats.matches_found,
        high_confidence_matches=stats.high_confidence_matches,
        possible_matches=stats.possible_matches,
        rejected_matches=stats.rejected_matches,
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
    config: Direction2V3RunConfig,
    output_paths: tuple[Path, ...],
    profiles: tuple[NameProfile, ...],
    polygons_read: int,
    frequency: _FrequencyResult,
    scan: _ScanResult,
    country_summaries: tuple[Direction2V3CountrySummary, ...],
) -> Direction2V3RunSummary:
    counts = Counter(profile.decision.decision for profile in profiles)
    return Direction2V3RunSummary(
        output_paths=output_paths,
        manifest_path=config.manifest_path,
        dataset_card_path=config.dataset_card_path,
        log_path=config.log_path,
        name_inventory_path=config.name_inventory_path,
        polygons_read=polygons_read,
        names_considered=len(profiles),
        names_indexed=counts["generic"] + counts["distinctive"],
        names_discarded=counts["discard"],
        generic_names=counts["generic"],
        fineweb_docs_frequency_pass=frequency.documents_scanned,
        fineweb_docs_match_pass=scan.documents_scanned,
        matches_found=scan.matches_found,
        high_confidence_matches=scan.high_confidence_matches,
        possible_matches=scan.possible_matches,
        rejected_matches=scan.rejected_matches,
        unique_polygons_matched=len(scan.polygon_ids),
        country_summaries=country_summaries,
    )


def _manifest(
    *,
    config: Direction2V3RunConfig,
    sources: tuple[_V3Source, ...],
    fingerprints: Mapping[str, object],
    profiles: tuple[NameProfile, ...],
    polygons_read: int,
    frequency: _FrequencyResult,
    scan: _ScanResult,
    country_summaries: tuple[Direction2V3CountrySummary, ...],
) -> dict[str, object]:
    counts = Counter(profile.decision.decision for profile in profiles)
    return {
        "configuration": {
            **_policy_record(),
            "batch_size": config.batch_size,
            "frequency_pass_reused": frequency.reused,
            "matcher": "Aho-Corasick",
            "output_batch_size": config.output_batch_size,
            "sentence_context": "matching sentence plus one sentence on each side",
        },
        "countries": {
            summary.source_key: summary.to_record() for summary in country_summaries
        },
        "direction": DIRECTION_V3_VERSION,
        "name_inventory": {
            "path": str(config.name_inventory_path),
            "sha256": sha256_file(config.name_inventory_path),
        },
        "polygon_inventory": {
            "generic_names": counts["generic"],
            "names_considered": len(profiles),
            "names_discarded": counts["discard"],
            "names_indexed": counts["generic"] + counts["distinctive"],
            "polygons_read": polygons_read,
        },
        "results": {
            "files": [
                {
                    "path": f"{DATA_PREFIX}/{source.key}.parquet",
                    "sha256": summary.result_sha256,
                    "source_key": source.key,
                }
                for source, summary in zip(sources, country_summaries, strict=True)
            ],
            "fineweb_docs_frequency_pass": frequency.documents_scanned,
            "fineweb_docs_match_pass": scan.documents_scanned,
            "high_confidence_matches": scan.high_confidence_matches,
            "matches_found": scan.matches_found,
            "possible_matches": scan.possible_matches,
            "rejected_matches": scan.rejected_matches,
            "unique_polygons_matched": len(scan.polygon_ids),
        },
        "schema": list(OUTPUT_COLUMNS_V3),
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
        evidence: CandidateEvidence,
    ) -> None:
        self._states[source_key].add(row, polygon_id, evidence.tier)

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
    """Own one source output, buffering rows before atomic publication."""

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

    def add(self, row: dict[str, object], polygon_id: str, tier: str) -> None:
        self.rows.append(row)
        self.stats.matches_found += 1
        self.stats.polygon_ids.add(polygon_id)
        if tier == "high_confidence":
            self.stats.high_confidence_matches += 1
        elif tier == "possible":
            self.stats.possible_matches += 1
        else:
            self.stats.rejected_matches += 1
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
