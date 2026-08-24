"""Resumable coordination for one FineWeb shard scan."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pyarrow.parquet as pq

from fineweb_polygons.deduplication import deduplicate_matches
from fineweb_polygons.foundation import (
    ProjectPaths,
    validate_data_path,
    validate_external_data_root,
)
from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.normalization import NORMALIZATION_VERSION
from fineweb_polygons.polygons import (
    read_named_polygon_profiles,
    read_v2_polygon_profiles,
    read_v3_polygon_profiles,
)
from fineweb_polygons.scanning import ScanStats, scan_row_groups
from fineweb_polygons.versions import get_retrieval_definition

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCHEMA_VERSION = 2
_DEFAULT_ROW_GROUPS_PER_PARTITION = 32


@dataclass(frozen=True, slots=True)
class ScanRunConfig:
    """Immutable inputs and runtime settings for one resumable run."""

    paths: ProjectPaths
    pbf_path: Path
    shard_path: Path
    run_id: str
    batch_size: int = 8192
    row_groups_per_partition: int = _DEFAULT_ROW_GROUPS_PER_PARTITION
    retrieval_version: str = "v1"

    def __post_init__(self) -> None:
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError(
                "run_id must contain only letters, numbers, dots, dashes, "
                "or underscores"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.row_groups_per_partition < 1:
            raise ValueError("row_groups_per_partition must be positive")
        get_retrieval_definition(self.retrieval_version)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Stable summary returned by a completed or resumed run."""

    result_path: Path
    manifest_path: Path
    partitions_completed: int
    partitions_skipped: int
    rows_scanned: int
    matches_written: int


@dataclass(frozen=True, slots=True)
class _RunCounters:
    partitions_completed: int
    partitions_skipped: int
    rows_scanned: int
    matches_written: int


@dataclass(frozen=True, slots=True)
class _RowGroup:
    index: int
    row_start: int
    row_count: int


@dataclass(frozen=True, slots=True)
class _Partition:
    index: int
    row_groups: tuple[_RowGroup, ...]

    @property
    def row_start(self) -> int:
        return self.row_groups[0].row_start

    @property
    def row_count(self) -> int:
        return sum(row_group.row_count for row_group in self.row_groups)

    @property
    def row_group_indices(self) -> tuple[int, ...]:
        return tuple(row_group.index for row_group in self.row_groups)


@dataclass(frozen=True, slots=True)
class _RunLayout:
    run_dir: Path
    partitions_dir: Path
    manifest_path: Path
    result_path: Path
    log_path: Path

    @classmethod
    def from_config(cls, config: ScanRunConfig) -> _RunLayout:
        run_dir = config.paths.runs_dir / config.run_id
        return cls(
            run_dir=run_dir,
            partitions_dir=run_dir / "partitions",
            manifest_path=run_dir / "manifest.json",
            result_path=config.paths.artifacts_dir / f"{config.run_id}-matches.jsonl",
            log_path=config.paths.logs_dir / f"{config.run_id}.jsonl",
        )


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
    row_groups = _inspect_row_groups(shard_path)
    partitions = _make_partitions(
        row_groups, groups_per_partition=config.row_groups_per_partition
    )
    source_fingerprints = {
        "pbf": {"path": str(pbf_path), "sha256": _sha256_file(pbf_path)},
        "shard": {"path": str(shard_path), "sha256": _sha256_file(shard_path)},
    }
    definition = get_retrieval_definition(config.retrieval_version)
    (
        selected_profiles,
        named_count,
        unnamed_count,
        filtered_count,
    ) = _select_profiles(
        pbf_path,
        profiles,
        retrieval_version=config.retrieval_version,
    )

    configuration = {
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
    expected_manifest = _new_manifest(
        run_id=config.run_id,
        layout=layout,
        partitions=partitions,
        source_fingerprints=source_fingerprints,
        profiles=selected_profiles,
        configuration=configuration,
        named_count=named_count,
        unnamed_count=unnamed_count,
        filtered_count=filtered_count,
    )
    manifest = _load_or_create_manifest(layout.manifest_path, expected_manifest)
    _log(layout.log_path, "run_started", run_id=config.run_id)
    matcher = EvidenceMatcher(
        selected_profiles,
        require_text_context=definition.requires_text_context,
        require_text_name=definition.requires_text_name,
        require_url_name=definition.requires_url_name,
    )
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


def _select_profiles(
    pbf_path: Path,
    profiles: Sequence[PolygonProfile] | None,
    *,
    retrieval_version: str,
) -> tuple[tuple[PolygonProfile, ...], int, int, int]:
    if profiles is None:
        readers = {
            "v1": read_named_polygon_profiles,
            "v2": read_v2_polygon_profiles,
            "v3": read_v3_polygon_profiles,
            "v4": read_v3_polygon_profiles,
        }
        reader = readers[retrieval_version]
        result = reader(pbf_path)
        return (
            result.profiles,
            result.named_count,
            result.unnamed_count,
            result.filtered_count,
        )
    selected = tuple(profiles)
    return selected, len(selected), 0, 0


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
    temporary_path = layout.result_path.with_name(f".{layout.result_path.name}.tmp")
    layout.result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for partition in partitions:
                partition_path = layout.partitions_dir / (
                    f"partition-{partition.index:05d}.jsonl"
                )
                if partition_path.exists():
                    with partition_path.open("r", encoding="utf-8") as partition:
                        shutil.copyfileobj(partition, output)
        temporary_path.replace(layout.result_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _log(path: Path, event: str, **fields: object) -> None:
    record = {"timestamp": _timestamp(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
