"""Value objects used by the resumable FineWeb runner."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from fineweb_polygons.foundation import ProjectPaths
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.specificity import SpecificityResult
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
    country_name: str = "Monaco"

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
        if not self.country_name.strip():
            raise ValueError("country_name must not be empty")
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
    name_frequency_path: Path | None = None

    @classmethod
    def from_config(cls, config: ScanRunConfig) -> _RunLayout:
        run_dir = config.paths.runs_dir / config.run_id
        return cls(
            run_dir=run_dir,
            partitions_dir=run_dir / "partitions",
            manifest_path=run_dir / "manifest.json",
            result_path=config.paths.artifacts_dir / f"{config.run_id}-matches.jsonl",
            log_path=config.paths.logs_dir / f"{config.run_id}.jsonl",
            name_frequency_path=run_dir / "name-frequency.json",
        )


@dataclass(frozen=True, slots=True)
class _ProfileRunData:
    """Profiles and counters selected before a shard scan."""

    profiles: tuple[PolygonProfile, ...]
    named_count: int
    unnamed_count: int
    filtered_count: int
    name_occurrences: Mapping[str, int]
    frequency_result: SpecificityResult | None = None
    frequency_artifact_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class _ProfileSelection:
    """Profiles and source counts selected for one retrieval version."""

    profiles: tuple[PolygonProfile, ...]
    named_count: int
    unnamed_count: int
    filtered_count: int
    name_occurrences: Mapping[str, int]
