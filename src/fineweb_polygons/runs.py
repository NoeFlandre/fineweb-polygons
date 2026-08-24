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

from fineweb_polygons.foundation import (
    ProjectPaths,
    validate_data_path,
    validate_external_data_root,
)
from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.normalization import NORMALIZATION_VERSION
from fineweb_polygons.polygons import read_named_polygon_profiles
from fineweb_polygons.scanning import scan_row_group

_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SCHEMA_VERSION = 1
_MATCHER_VERSION = "v1-exact-name-context"


@dataclass(frozen=True, slots=True)
class ScanRunConfig:
    """Immutable inputs and runtime settings for one resumable run."""

    paths: ProjectPaths
    pbf_path: Path
    shard_path: Path
    run_id: str
    batch_size: int = 8192

    def __post_init__(self) -> None:
        if not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError(
                "run_id must contain only letters, numbers, dots, dashes, "
                "or underscores"
            )
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


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
class _RowGroup:
    index: int
    row_start: int
    row_count: int


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
    """Execute or resume a row-group-partitioned FineWeb scan."""
    validate_external_data_root(config.paths)
    config.paths.ensure_data_layout()
    pbf_path = validate_data_path(config.paths, config.pbf_path)
    shard_path = validate_data_path(config.paths, config.shard_path)
    if not pbf_path.is_file():
        raise FileNotFoundError(pbf_path)
    if not shard_path.is_file():
        raise FileNotFoundError(shard_path)

    layout = _RunLayout.from_config(config)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    layout.partitions_dir.mkdir(parents=True, exist_ok=True)
    row_groups = _inspect_row_groups(shard_path)
    source_fingerprints = {
        "pbf": {"path": str(pbf_path), "sha256": _sha256_file(pbf_path)},
        "shard": {"path": str(shard_path), "sha256": _sha256_file(shard_path)},
    }
    if profiles is None:
        profile_result = read_named_polygon_profiles(pbf_path)
        selected_profiles = profile_result.profiles
        named_count = profile_result.named_count
        unnamed_count = profile_result.unnamed_count
    else:
        selected_profiles = tuple(profiles)
        named_count = len(selected_profiles)
        unnamed_count = 0

    configuration = {
        "batch_size": config.batch_size,
        "matcher_version": _MATCHER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
    }
    expected_manifest = _new_manifest(
        config=config,
        layout=layout,
        row_groups=row_groups,
        source_fingerprints=source_fingerprints,
        profiles=selected_profiles,
        configuration=configuration,
        named_count=named_count,
        unnamed_count=unnamed_count,
    )
    manifest = _load_or_create_manifest(layout.manifest_path, expected_manifest)
    _log(layout.log_path, "run_started", run_id=config.run_id)
    matcher = EvidenceMatcher(selected_profiles)
    partitions_completed = 0
    partitions_skipped = 0
    rows_scanned = 0
    matches_written = 0
    started = perf_counter()

    for row_group in row_groups:
        partition = manifest["partitions"][row_group.index]
        partition_path = layout.partitions_dir / (
            f"partition-{row_group.index:05d}.jsonl"
        )
        partition["path"] = str(partition_path)
        if partition["status"] == "complete" and partition_path.is_file():
            partitions_skipped += 1
            stats = partition["stats"]
            rows_scanned += int(stats["rows_scanned"])
            matches_written += int(stats["matches_written"])
            _log(layout.log_path, "partition_skipped", partition=row_group.index)
            continue

        partition["status"] = "running"
        _atomic_json_write(layout.manifest_path, manifest)
        try:
            stats = scan_row_group(
                shard_path,
                row_group_index=row_group.index,
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
                partition=row_group.index,
                error=str(error),
            )
            raise
        partition["status"] = "complete"
        partition["stats"] = {
            "rows_scanned": stats.rows_scanned,
            "matches_written": stats.matches_written,
        }
        _atomic_json_write(layout.manifest_path, manifest)
        partitions_completed += 1
        rows_scanned += stats.rows_scanned
        matches_written += stats.matches_written
        _log(
            layout.log_path,
            "partition_complete",
            partition=row_group.index,
            rows_scanned=stats.rows_scanned,
            matches=stats.matches_written,
        )

    _merge_partitions(layout, row_groups)
    manifest["status"] = "complete"
    manifest["rows_scanned"] = rows_scanned
    manifest["matches_written"] = matches_written
    manifest["elapsed_seconds"] = perf_counter() - started
    manifest["result_sha256"] = _sha256_file(layout.result_path)
    manifest["completed_at"] = _timestamp()
    _atomic_json_write(layout.manifest_path, manifest)
    _log(
        layout.log_path,
        "run_complete",
        rows_scanned=rows_scanned,
        matches=matches_written,
        elapsed_seconds=manifest["elapsed_seconds"],
    )
    return RunSummary(
        result_path=layout.result_path,
        manifest_path=layout.manifest_path,
        partitions_completed=partitions_completed,
        partitions_skipped=partitions_skipped,
        rows_scanned=rows_scanned,
        matches_written=matches_written,
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


def _new_manifest(
    *,
    config: ScanRunConfig,
    layout: _RunLayout,
    row_groups: Sequence[_RowGroup],
    source_fingerprints: Mapping[str, object],
    profiles: Sequence[PolygonProfile],
    configuration: Mapping[str, object],
    named_count: int,
    unnamed_count: int,
) -> dict[str, Any]:
    partition_records = [
        {
            "index": row_group.index,
            "row_start": row_group.row_start,
            "row_count": row_group.row_count,
            "status": "pending",
            "stats": None,
            "path": str(
                layout.partitions_dir / f"partition-{row_group.index:05d}.jsonl"
            ),
        }
        for row_group in row_groups
    ]
    return {
        "schema_version": _SCHEMA_VERSION,
        "run_id": config.run_id,
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
    manifest: dict[str, Any] = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    for key in (
        "schema_version",
        "run_id",
        "sources",
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
        {key: partition[key] for key in ("index", "row_start", "row_count", "path")}
        for partition in partitions
        if isinstance(partition, dict)
    ]


def _merge_partitions(layout: _RunLayout, row_groups: Sequence[_RowGroup]) -> None:
    temporary_path = layout.result_path.with_name(f".{layout.result_path.name}.tmp")
    layout.result_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for row_group in row_groups:
                partition_path = layout.partitions_dir / (
                    f"partition-{row_group.index:05d}.jsonl"
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
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
