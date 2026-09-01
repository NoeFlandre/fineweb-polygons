"""Small immutable records for the isolated Direction 2 POC."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DIRECTION_VERSION = "direction-2-lexical-v1"
HF_CONFIG_NAME = "direction_2_lexical_v1"
OUTPUT_COLUMNS = (
    "polygon_id",
    "polygon_name",
    "matched_alias",
    "osm_tags",
    "centroid",
    "fineweb_url",
    "sentence",
    "context",
)


@dataclass(frozen=True, slots=True)
class PolygonSource:
    """One named OSM extract and its stable source key."""

    key: str
    path: Path

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("polygon source key must not be empty")


@dataclass(frozen=True, slots=True)
class PolygonRecord:
    """One OSM area, including the metadata needed for lexical evidence."""

    polygon_id: str
    source_key: str
    name: str
    aliases: tuple[str, ...]
    tags: tuple[tuple[str, str], ...]
    centroid: tuple[float, float] | None

    def candidate_names(self) -> tuple[str, ...]:
        """Return the main name followed by its non-empty aliases."""
        return tuple(name for name in (self.name, *self.aliases) if name.strip())

    def tags_as_json(self) -> str:
        """Serialize tags in a stable, viewer-friendly representation."""
        return json.dumps(
            dict(self.tags), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    def centroid_as_json(self) -> str:
        """Serialize longitude/latitude as a stable viewer-friendly object."""
        if self.centroid is None:
            return ""
        longitude, latitude = self.centroid
        return json.dumps(
            {"lat": latitude, "lon": longitude},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class PolygonNameMatch:
    """One polygon-name match with offsets into the original document text."""

    polygon: PolygonRecord
    matched_alias: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SentenceWindow:
    """The matching sentence and its bounded neighboring context."""

    sentence: str
    context: str


@dataclass(frozen=True, slots=True)
class Direction2RunConfig:
    """Inputs and output locations for one deterministic Direction 2 run."""

    monaco_pbf: Path
    liechtenstein_pbf: Path
    shard_path: Path
    output_dir: Path
    manifest_path: Path
    dataset_card_path: Path
    log_path: Path
    batch_size: int = 8192
    output_batch_size: int = 4096

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.output_batch_size <= 0:
            raise ValueError("output_batch_size must be positive")
        input_paths = {
            self.monaco_pbf.expanduser().resolve(),
            self.liechtenstein_pbf.expanduser().resolve(),
            self.shard_path.expanduser().resolve(),
        }
        if len(input_paths) != 3:
            raise ValueError("Direction 2 input paths must be different")


@dataclass(frozen=True, slots=True)
class Direction2CountrySummary:
    """Counts and digest for one source-specific result file."""

    source_key: str
    output_path: Path
    polygons_read: int
    names_indexed: int
    matches_found: int
    unique_polygons_matched: int
    result_sha256: str

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary record."""
        return {
            "matches_found": self.matches_found,
            "names_indexed": self.names_indexed,
            "output_path": str(self.output_path),
            "polygons_read": self.polygons_read,
            "result_sha256": self.result_sha256,
            "unique_polygons_matched": self.unique_polygons_matched,
        }


@dataclass(frozen=True, slots=True)
class Direction2RunSummary:
    """Stable counters and artifacts produced by a complete Direction 2 run."""

    output_paths: tuple[Path, ...]
    manifest_path: Path
    dataset_card_path: Path
    log_path: Path
    polygons_read: int
    names_indexed: int
    fineweb_docs_scanned: int
    matches_found: int
    unique_polygons_matched: int
    country_summaries: tuple[Direction2CountrySummary, ...]

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary record."""
        return {
            "countries": {
                summary.source_key: summary.to_record()
                for summary in self.country_summaries
            },
            "dataset_card_path": str(self.dataset_card_path),
            "fineweb_docs_scanned": self.fineweb_docs_scanned,
            "log_path": str(self.log_path),
            "manifest_path": str(self.manifest_path),
            "matches_found": self.matches_found,
            "names_indexed": self.names_indexed,
            "output_paths": [str(path) for path in self.output_paths],
            "polygons_read": self.polygons_read,
            "unique_polygons_matched": self.unique_polygons_matched,
        }
