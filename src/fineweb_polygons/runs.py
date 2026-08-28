"""Resumable coordination for one FineWeb shard scan."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import pyarrow.parquet as pq

from fineweb_polygons.artifact_io import (
    atomic_json_write as _shared_atomic_json_write,
)
from fineweb_polygons.artifact_io import (
    atomic_text_output as _atomic_text_output,
)
from fineweb_polygons.artifact_io import (
    deterministic_temporary_path as _deterministic_temporary_path,
)
from fineweb_polygons.artifact_io import (
    sha256_file as _sha256_file,
)
from fineweb_polygons.artifact_io import (
    write_json_line as _write_json_line,
)
from fineweb_polygons.deduplication import deduplicate_matches
from fineweb_polygons.foundation import validate_data_path, validate_external_data_root
from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.normalization import NORMALIZATION_VERSION
from fineweb_polygons.polygons import (
    read_named_polygon_profiles,
    read_v2_polygon_profiles,
    read_v3_polygon_profiles,
)
from fineweb_polygons.run_models import (
    _SCHEMA_VERSION,
    RunSummary,
    ScanRunConfig,
    _Partition,
    _ProfileRunData,
    _RowGroup,
    _RunCounters,
    _RunLayout,
)
from fineweb_polygons.scanning import ScanStats, scan_row_groups
from fineweb_polygons.specificity import (
    FINEWEB_DOCUMENT_FREQUENCY_RATIO,
    NameFrequency,
    SpecificityResult,
    count_fineweb_document_frequencies,
    filter_specific_profiles,
    frequency_threshold,
)
from fineweb_polygons.versions import get_retrieval_definition


def execute_run(
    config: ScanRunConfig,
    *,
    profiles: Sequence[PolygonProfile] | None = None,
) -> RunSummary:
    """Execute or resume a chunked FineWeb scan."""
    pbf_path, shard_path = _validated_inputs(config)
    layout = _RunLayout.from_config(config)
    layout.run_dir.mkdir(exist_ok=True)
    layout.partitions_dir.mkdir(exist_ok=True)
    _log(layout.log_path, "run_started", run_id=config.run_id)
    row_groups = _inspect_row_groups(shard_path)
    partitions = _make_partitions(
        row_groups, groups_per_partition=config.row_groups_per_partition
    )
    source_fingerprints = {
        "pbf": {"path": str(pbf_path), "sha256": _sha256_file(pbf_path)},
        "shard": {"path": str(shard_path), "sha256": _sha256_file(shard_path)},
    }
    definition = get_retrieval_definition(config.retrieval_version)
    profile_data = _prepare_profile_data(
        config=config,
        definition=definition,
        layout=layout,
        pbf_path=pbf_path,
        shard_path=shard_path,
        profiles=profiles,
        source_shard_sha256=str(source_fingerprints["shard"]["sha256"]),
    )
    configuration = _run_configuration(
        config=config,
        definition=definition,
        layout=layout,
        profile_data=profile_data,
    )
    expected_manifest = _new_manifest(
        run_id=config.run_id,
        layout=layout,
        partitions=partitions,
        source_fingerprints=source_fingerprints,
        profiles=profile_data.profiles,
        configuration=configuration,
        named_count=profile_data.named_count,
        unnamed_count=profile_data.unnamed_count,
        filtered_count=profile_data.filtered_count,
    )
    manifest = _load_or_create_manifest(layout.manifest_path, expected_manifest)
    matcher = _matcher_for_run(config, definition, profile_data.profiles)
    started = perf_counter()
    counters = _process_partitions(
        config=config,
        layout=layout,
        manifest=manifest,
        partitions=partitions,
        shard_path=shard_path,
        matcher=matcher,
    )
    _merge_partitions(layout, partitions)
    matches_written = counters.matches_written
    if definition.deduplicate_documents:
        matches_written = deduplicate_matches(layout.result_path)
    final_counters = _RunCounters(
        partitions_completed=counters.partitions_completed,
        partitions_skipped=counters.partitions_skipped,
        rows_scanned=counters.rows_scanned,
        matches_written=matches_written,
    )
    _complete_run(layout, manifest, final_counters, elapsed=perf_counter() - started)
    return RunSummary(
        result_path=layout.result_path,
        manifest_path=layout.manifest_path,
        partitions_completed=final_counters.partitions_completed,
        partitions_skipped=final_counters.partitions_skipped,
        rows_scanned=final_counters.rows_scanned,
        matches_written=final_counters.matches_written,
    )


def _validated_inputs(config: ScanRunConfig) -> tuple[Path, Path]:
    validate_external_data_root(config.paths)
    config.paths.ensure_data_layout()
    pbf_path = validate_data_path(config.paths, config.pbf_path)
    shard_path = validate_data_path(config.paths, config.shard_path)
    for path in (pbf_path, shard_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return pbf_path, shard_path


def _prepare_profile_data(
    *,
    config: ScanRunConfig,
    definition: Any,
    layout: _RunLayout,
    pbf_path: Path,
    shard_path: Path,
    profiles: Sequence[PolygonProfile] | None,
    source_shard_sha256: str,
) -> _ProfileRunData:
    include_occurrences = definition.requires_name_specificity
    if include_occurrences:
        selection = _select_profiles(
            pbf_path,
            profiles,
            retrieval_version=config.retrieval_version,
            include_name_occurrences=True,
        )
    else:
        selection = _select_profiles(
            pbf_path,
            profiles,
            retrieval_version=config.retrieval_version,
        )
    # pragma: no mutate start
    selected_profiles = cast(tuple[PolygonProfile, ...], selection[0])
    # pragma: no mutate end
    name_occurrences = _selection_occurrences(selection)
    # pragma: no mutate start
    base_data = _ProfileRunData(
        profiles=selected_profiles,
        named_count=cast(int, selection[1]),  # pragma: no mutate
        unnamed_count=cast(int, selection[2]),  # pragma: no mutate
        filtered_count=cast(int, selection[3]),  # pragma: no mutate
        name_occurrences=name_occurrences,
    )
    # pragma: no mutate end
    if not include_occurrences:
        return base_data
    if not name_occurrences:
        name_occurrences = {profile.normalized_name: 1 for profile in selected_profiles}
    result, artifact_sha256 = _load_or_build_frequency_artifact(
        layout=layout,
        shard_path=shard_path,
        profiles=selected_profiles,
        osm_name_occurrences=name_occurrences,
        source_shard_sha256=source_shard_sha256,
        country_name=config.country_name,
        batch_size=config.batch_size,
    )
    return _ProfileRunData(
        profiles=result.profiles,
        named_count=len(result.profiles),
        unnamed_count=base_data.unnamed_count,
        filtered_count=base_data.filtered_count + result.removed_count,
        name_occurrences=name_occurrences,
        frequency_result=result,
        frequency_artifact_sha256=artifact_sha256,
    )


def _selection_occurrences(
    selection: tuple[object, ...],
) -> dict[str, int]:  # pragma: no mutate block
    if len(selection) < 5:
        return {}
    raw = cast(  # pragma: no mutate
        tuple[tuple[str, int], ...], selection[4]
    )
    return dict(raw)


def _run_configuration(
    *,
    config: ScanRunConfig,
    definition: Any,
    layout: _RunLayout,
    profile_data: _ProfileRunData,
) -> dict[str, object]:
    configuration: dict[str, object] = {
        "batch_size": config.batch_size,
        "deduplicate_documents": definition.deduplicate_documents,
        "row_groups_per_partition": config.row_groups_per_partition,
        "matcher_version": definition.matcher_version,
        "normalization_version": NORMALIZATION_VERSION,
        "polygon_profile_version": definition.polygon_profile_version,
        "require_url_name": definition.requires_url_name,
        "retrieval_version": config.retrieval_version,
        "retrieval_definition": definition.to_record(),
    }
    if definition.requires_text_name:
        configuration["require_text_name"] = True
    if definition.max_name_country_distance is not None:
        configuration["max_name_country_distance"] = (
            definition.max_name_country_distance
        )
    if definition.requires_name_specificity:
        configuration.update(_specificity_configuration(config, layout, profile_data))
    return configuration


def _specificity_configuration(
    config: ScanRunConfig,
    layout: _RunLayout,
    profile_data: _ProfileRunData,
) -> dict[str, object]:
    result = profile_data.frequency_result
    artifact_sha256 = profile_data.frequency_artifact_sha256
    if result is None or artifact_sha256 is None:
        raise ValueError("V5 profile data is missing its frequency artifact")
    return {
        "base_polygon_profile_count": len(profile_data.name_occurrences),
        "country_name": config.country_name,
        "fineweb_document_frequency_ratio": FINEWEB_DOCUMENT_FREQUENCY_RATIO,
        "fineweb_document_frequency_threshold": frequency_threshold(
            result.documents_scanned
        ),
        "name_frequency_artifact": str(layout.name_frequency_path),
        "name_frequency_artifact_sha256": artifact_sha256,
        "name_specificity_rule": (
            "keep OSM-unique names at or below the FineWeb 0.1% "
            "document-frequency cutoff; use the country name as context only"
        ),
    }


def _matcher_for_run(
    config: ScanRunConfig,
    definition: Any,
    profiles: Sequence[PolygonProfile],
) -> EvidenceMatcher:
    matcher_options: dict[str, Any] = {
        "require_text_context": definition.requires_text_context,
        "require_text_name": definition.requires_text_name,
        "require_url_name": definition.requires_url_name,
    }
    if definition.requires_name_specificity:
        matcher_options["context_name"] = config.country_name
    if definition.max_name_country_distance is not None:
        matcher_options["max_name_country_distance"] = (
            definition.max_name_country_distance
        )
    return EvidenceMatcher(profiles, **matcher_options)


def _select_profiles(
    pbf_path: Path,
    profiles: Sequence[PolygonProfile] | None,
    *,
    retrieval_version: str,
    include_name_occurrences: bool = False,
) -> tuple[object, ...]:
    if profiles is None:
        readers = {
            "v1": read_named_polygon_profiles,
            "v2": read_v2_polygon_profiles,
            "v3": read_v3_polygon_profiles,
            "v4": read_v3_polygon_profiles,
            "v5": read_v3_polygon_profiles,
            "v6": read_v3_polygon_profiles,
        }
        reader = readers[retrieval_version]
        result = reader(pbf_path)
        selection: tuple[object, ...] = (
            result.profiles,
            result.named_count,
            result.unnamed_count,
            result.filtered_count,
        )
        if include_name_occurrences:
            selection += (result.name_occurrences,)
        return selection
    selected = tuple(profiles)
    selection = (selected, len(selected), 0, 0)
    if include_name_occurrences:
        selection += (tuple((profile.normalized_name, 1) for profile in selected),)
    return selection


def _load_or_build_frequency_artifact(
    *,
    layout: _RunLayout,
    shard_path: Path,
    profiles: Sequence[PolygonProfile],
    osm_name_occurrences: Mapping[str, int],
    source_shard_sha256: str,
    country_name: str,
    batch_size: int,
) -> tuple[SpecificityResult, str]:
    """Load or create the resumable V5 name-frequency artifact."""
    frequency_path = _frequency_path(layout)
    metadata = _frequency_metadata(
        profiles=profiles,
        osm_name_occurrences=osm_name_occurrences,
        source_shard_sha256=source_shard_sha256,
        country_name=country_name,
        batch_size=batch_size,
    )
    if frequency_path.exists():
        result = _read_frequency_result(
            frequency_path,
            profiles,
            metadata,
            country_name=country_name,
        )
        _log(
            layout.log_path,
            "name_frequency_skipped",
            documents_scanned=result.documents_scanned,
            profiles=len(profiles),
        )
        return result, _sha256_file(frequency_path)

    _log(layout.log_path, "name_frequency_started", profiles=len(profiles))
    result, artifact = _build_frequency_result(
        shard_path=shard_path,
        profiles=profiles,
        osm_name_occurrences=osm_name_occurrences,
        country_name=country_name,
        batch_size=batch_size,
        metadata=metadata,
    )
    _atomic_json_write(frequency_path, artifact)
    _log(
        layout.log_path,
        "name_frequency_complete",
        documents_scanned=result.documents_scanned,
        profiles=len(profiles),
        retained_profiles=len(result.profiles),
        threshold=frequency_threshold(result.documents_scanned),
    )
    return result, _sha256_file(frequency_path)


def _frequency_path(layout: _RunLayout) -> Path:
    if layout.name_frequency_path is None:
        raise ValueError("V5 runs require a name-frequency artifact path")
    return layout.name_frequency_path


def _frequency_metadata(
    *,
    profiles: Sequence[PolygonProfile],
    osm_name_occurrences: Mapping[str, int],
    source_shard_sha256: str,
    country_name: str,
    batch_size: int,
) -> dict[str, object]:
    profile_records = [
        {
            "polygon_id": profile.polygon_id,
            "name": profile.name,
            "normalized_name": profile.normalized_name,
            "osm_occurrences": int(
                osm_name_occurrences.get(profile.normalized_name, 1)
            ),
        }
        for profile in profiles
    ]
    return {
        "schema_version": 1,
        "shard_sha256": source_shard_sha256,
        "base_polygon_profile_sha256": _sha256_payload(profile_records),
        "country_name": country_name,
        "batch_size": batch_size,
        "fineweb_document_frequency_ratio": FINEWEB_DOCUMENT_FREQUENCY_RATIO,
        "profiles": profile_records,
    }


def _read_frequency_result(
    frequency_path: Path,
    profiles: Sequence[PolygonProfile],
    metadata: Mapping[str, object],
    *,
    country_name: str,
) -> SpecificityResult:
    artifact = json.loads(frequency_path.read_text(encoding="utf-8"))
    _validate_frequency_metadata(artifact, metadata)
    documents_scanned = int(artifact["documents_scanned"])
    threshold = int(artifact["fineweb_document_frequency_threshold"])
    _validate_frequency_threshold(threshold, documents_scanned)
    records = _frequency_records_from_artifact(artifact)
    return _apply_frequency_filter(
        profiles,
        records,
        country_name=country_name,
        threshold=threshold,
        documents_scanned=documents_scanned,
    )


def _validate_frequency_metadata(
    artifact: Mapping[str, object], metadata: Mapping[str, object]
) -> None:
    for key, expected in metadata.items():
        if artifact.get(key) != expected:
            raise ValueError(f"Name-frequency artifact fingerprint conflict in {key}")


def _validate_frequency_threshold(threshold: int, documents_scanned: int) -> None:
    if threshold != frequency_threshold(documents_scanned):
        raise ValueError(
            "Name-frequency artifact has an invalid document-frequency threshold"
        )


def _frequency_records_from_artifact(
    artifact: Mapping[str, object],
) -> tuple[NameFrequency, ...]:  # pragma: no mutate block
    raw_records = cast(Sequence[Mapping[str, object]], artifact["frequencies"])
    return tuple(
        NameFrequency(
            normalized_name=str(record["normalized_name"]),
            osm_occurrences=int(  # pragma: no mutate
                cast(int, record["osm_occurrences"])
            ),
            fineweb_document_frequency=int(
                cast(  # pragma: no mutate
                    int, record["fineweb_document_frequency"]
                )
            ),
        )
        for record in raw_records
    )


def _build_frequency_result(
    *,
    shard_path: Path,
    profiles: Sequence[PolygonProfile],
    osm_name_occurrences: Mapping[str, int],
    country_name: str,
    batch_size: int,
    metadata: Mapping[str, object],
) -> tuple[SpecificityResult, dict[str, object]]:
    fineweb_frequencies, documents_scanned = count_fineweb_document_frequencies(
        shard_path,
        profiles,
        batch_size=batch_size,
    )
    records = _frequency_records(
        profiles,
        osm_name_occurrences,
        fineweb_frequencies,
    )
    threshold = frequency_threshold(documents_scanned)
    result = _apply_frequency_filter(
        profiles,
        records,
        country_name=country_name,
        threshold=threshold,
        documents_scanned=documents_scanned,
    )
    artifact = {
        **metadata,
        "documents_scanned": documents_scanned,
        "fineweb_document_frequency_threshold": threshold,
        "frequencies": [record.to_record() for record in records],
    }
    return result, artifact


def _frequency_records(
    profiles: Sequence[PolygonProfile],
    osm_name_occurrences: Mapping[str, int],
    fineweb_frequencies: Mapping[str, int],
) -> tuple[NameFrequency, ...]:
    return tuple(
        NameFrequency(
            normalized_name=profile.normalized_name,
            osm_occurrences=int(osm_name_occurrences.get(profile.normalized_name, 1)),
            fineweb_document_frequency=int(
                fineweb_frequencies.get(profile.normalized_name, 0)
            ),
        )
        for profile in profiles
    )


def _apply_frequency_filter(
    profiles: Sequence[PolygonProfile],
    records: Sequence[NameFrequency],
    *,
    country_name: str,
    threshold: int,
    documents_scanned: int,
) -> SpecificityResult:
    result = filter_specific_profiles(
        profiles,
        {record.normalized_name: record for record in records},
        country_name=country_name,
        fineweb_document_frequency_threshold=threshold,
    )
    return SpecificityResult(
        profiles=result.profiles,
        frequencies=result.frequencies,
        documents_scanned=documents_scanned,
    )


def _process_partitions(
    *,
    config: ScanRunConfig,
    layout: _RunLayout,
    manifest: dict[str, Any],
    partitions: Sequence[_Partition],
    shard_path: Path,
    matcher: EvidenceMatcher,
) -> _RunCounters:
    completed = skipped = rows = matches = 0
    for partition_spec in partitions:
        was_skipped, stats = _process_partition(
            config=config,
            layout=layout,
            manifest=manifest,
            partition_spec=partition_spec,
            shard_path=shard_path,
            matcher=matcher,
        )
        if was_skipped:
            skipped += 1
        else:
            completed += 1
        rows += stats.rows_scanned
        matches += stats.matches_written
    return _RunCounters(completed, skipped, rows, matches)


def _process_partition(
    *,
    config: ScanRunConfig,
    layout: _RunLayout,
    manifest: dict[str, Any],
    partition_spec: _Partition,
    shard_path: Path,
    matcher: EvidenceMatcher,
) -> tuple[bool, ScanStats]:
    partition = manifest["partitions"][partition_spec.index]
    partition_path = layout.partitions_dir / (
        f"partition-{partition_spec.index:05d}.jsonl"
    )
    partition["path"] = str(partition_path)
    if _partition_is_complete(partition, partition_path):
        stats = _stats_from_record(partition)
        _log(layout.log_path, "partition_skipped", partition=partition_spec.index)
        return True, stats
    partition["status"] = "running"
    _atomic_json_write(layout.manifest_path, manifest)
    try:
        stats = scan_row_groups(
            shard_path,
            row_group_indices=partition_spec.row_group_indices,
            matcher=matcher,
            output_path=partition_path,
            batch_size=config.batch_size,
        )
    except Exception as error:
        partition["status"] = "failed"
        partition["error"] = str(error)
        _atomic_json_write(layout.manifest_path, manifest)
        _log(
            layout.log_path,
            "partition_failed",
            partition=partition_spec.index,
            error=str(error),
        )
        raise
    partition["status"] = "complete"
    partition["stats"] = {
        "rows_scanned": stats.rows_scanned,
        "matches_written": stats.matches_written,
    }
    _atomic_json_write(layout.manifest_path, manifest)
    _log(
        layout.log_path,
        "partition_complete",
        partition=partition_spec.index,
        rows_scanned=stats.rows_scanned,
        matches=stats.matches_written,
    )
    return False, stats


def _partition_is_complete(partition: Mapping[str, Any], path: Path) -> bool:
    return partition["status"] == "complete" and path.is_file()


def _stats_from_record(partition: Mapping[str, Any]) -> ScanStats:
    stats = partition["stats"]
    return ScanStats(
        rows_scanned=int(stats["rows_scanned"]),
        matches_written=int(stats["matches_written"]),
    )


def _complete_run(
    layout: _RunLayout,
    manifest: dict[str, Any],
    counters: _RunCounters,
    *,
    elapsed: float,
) -> None:
    manifest["status"] = "complete"
    manifest["rows_scanned"] = counters.rows_scanned
    manifest["matches_written"] = counters.matches_written
    manifest["elapsed_seconds"] = elapsed
    manifest["result_sha256"] = _sha256_file(layout.result_path)
    manifest["completed_at"] = _timestamp()
    _atomic_json_write(layout.manifest_path, manifest)
    _log(
        layout.log_path,
        "run_complete",
        rows_scanned=counters.rows_scanned,
        matches=counters.matches_written,
        elapsed_seconds=elapsed,
    )


def _inspect_row_groups(shard_path: Path) -> tuple[_RowGroup, ...]:
    parquet_file = pq.ParquetFile(shard_path)
    missing_columns = {"text", "url"} - set(parquet_file.schema_arrow.names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Parquet shard must contain text and url columns; missing {missing}"
        )
    row_groups: list[_RowGroup] = []
    row_start = 0
    for index in range(parquet_file.metadata.num_row_groups):
        row_count = parquet_file.metadata.row_group(index).num_rows
        row_groups.append(_RowGroup(index, row_start, row_count))
        row_start += row_count
    return tuple(row_groups)


def _make_partitions(
    row_groups: Sequence[_RowGroup], *, groups_per_partition: int
) -> tuple[_Partition, ...]:
    return tuple(
        _Partition(index, tuple(row_groups[start : start + groups_per_partition]))
        for index, start in enumerate(range(0, len(row_groups), groups_per_partition))
    )


def _new_manifest(
    *,
    run_id: str,
    layout: _RunLayout,
    partitions: Sequence[_Partition],
    source_fingerprints: Mapping[str, object],
    profiles: Sequence[PolygonProfile],
    configuration: Mapping[str, object],
    named_count: int,
    unnamed_count: int,
    filtered_count: int,
) -> dict[str, Any]:
    partition_records = [
        {
            "index": partition.index,
            "row_group_indices": list(partition.row_group_indices),
            "row_start": partition.row_start,
            "row_count": partition.row_count,
            "status": "pending",
            "stats": None,
            "path": str(
                layout.partitions_dir / f"partition-{partition.index:05d}.jsonl"
            ),
        }
        for partition in partitions
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "sources": source_fingerprints,
        "configuration": configuration,
        "configuration_sha256": _sha256_payload(configuration),
        "polygon_profile_sha256": _sha256_payload(
            [
                {
                    "polygon_id": profile.polygon_id,
                    "name": profile.name,
                    "normalized_name": profile.normalized_name,
                }
                for profile in profiles
            ]
        ),
        "polygon_counts": {
            "named": named_count,
            "unnamed": unnamed_count,
            "filtered": filtered_count,
        },
        "partitions": partition_records,
        "result_path": str(layout.result_path),
    }


def _load_or_create_manifest(
    manifest_path: Path,
    expected: Mapping[str, object],
) -> dict[str, Any]:
    if not manifest_path.exists():
        created = dict(expected)
        _atomic_json_write(manifest_path, created)
        return created
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "run_id",
        "sources",
        "configuration",
        "configuration_sha256",
        "polygon_profile_sha256",
    ):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"Run manifest fingerprint conflict in {key}")
    if _partition_structure(manifest["partitions"]) != _partition_structure(
        expected["partitions"]
    ):
        raise ValueError("Run manifest fingerprint conflict in partitions")
    return manifest


def _partition_structure(partitions: object) -> list[dict[str, object]]:
    if not isinstance(partitions, list):
        raise ValueError("Run manifest partitions must be a list")
    return [
        {
            key: partition[key]
            for key in ("index", "row_group_indices", "row_start", "row_count", "path")
        }
        for partition in partitions
        if isinstance(partition, dict)
    ]


def _merge_partitions(layout: _RunLayout, partitions: Sequence[_Partition]) -> None:
    with _atomic_text_output(
        layout.result_path,
        temporary_factory=_deterministic_temporary_path,
    ) as output:
        for partition in partitions:
            partition_path = layout.partitions_dir / (
                f"partition-{partition.index:05d}.jsonl"
            )
            if partition_path.exists():
                with partition_path.open("r", encoding="utf-8") as partition:
                    shutil.copyfileobj(partition, output)


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    _shared_atomic_json_write(
        path,
        value,
        temporary_factory=_deterministic_temporary_path,
    )


def _log(path: Path, event: str, **fields: object) -> None:
    record = {"timestamp": _timestamp(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as output:
        _write_json_line(output, record)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_payload(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
