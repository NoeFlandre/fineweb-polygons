"""Versioned records for Direction 2 lexical V2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fineweb_polygons.direction2.models import OUTPUT_COLUMNS

DIRECTION_V2_VERSION = "direction-2-lexical-v2"
HF_CONFIG_NAME_V2 = "direction_2_lexical_v2"
OUTPUT_COLUMNS_V2 = (
    *OUTPUT_COLUMNS,
    "name_match_class",
    "osm_polygon_count",
    "fineweb_document_frequency",
)
COUNTRY_NAMES = {
    "monaco": "Monaco",
    "liechtenstein": "Liechtenstein",
}


@dataclass(frozen=True, slots=True)
class Direction2V2RunConfig:
    """Inputs and output locations for a deterministic V2 run."""

    monaco_pbf: Path
    liechtenstein_pbf: Path
    shard_path: Path
    output_dir: Path
    manifest_path: Path
    dataset_card_path: Path
    log_path: Path
    name_inventory_path: Path
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
            raise ValueError("Direction 2 V2 input paths must be different")


@dataclass(frozen=True, slots=True)
class Direction2V2CountrySummary:
    """Counts and digest for one V2 source result."""

    source_key: str
    output_path: Path
    polygons_read: int
    names_indexed: int
    matches_found: int
    distinctive_matches: int
    generic_matches: int
    unique_polygons_matched: int
    result_sha256: str

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary."""
        return {
            "distinctive_matches": self.distinctive_matches,
            "generic_matches": self.generic_matches,
            "matches_found": self.matches_found,
            "names_indexed": self.names_indexed,
            "output_path": str(self.output_path),
            "polygons_read": self.polygons_read,
            "result_sha256": self.result_sha256,
            "unique_polygons_matched": self.unique_polygons_matched,
        }


@dataclass(frozen=True, slots=True)
class Direction2V2RunSummary:
    """Stable counters and artifacts produced by a V2 run."""

    output_paths: tuple[Path, ...]
    manifest_path: Path
    dataset_card_path: Path
    log_path: Path
    name_inventory_path: Path
    polygons_read: int
    names_considered: int
    names_indexed: int
    names_discarded: int
    generic_names: int
    fineweb_docs_frequency_pass: int
    fineweb_docs_match_pass: int
    matches_found: int
    distinctive_matches: int
    generic_matches: int
    unique_polygons_matched: int
    country_summaries: tuple[Direction2V2CountrySummary, ...]
    direction: str = DIRECTION_V2_VERSION

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary."""
        return {
            "countries": {
                summary.source_key: summary.to_record()
                for summary in self.country_summaries
            },
            "dataset_card_path": str(self.dataset_card_path),
            "direction": self.direction,
            "fineweb_docs_frequency_pass": self.fineweb_docs_frequency_pass,
            "fineweb_docs_match_pass": self.fineweb_docs_match_pass,
            "generic_matches": self.generic_matches,
            "generic_names": self.generic_names,
            "log_path": str(self.log_path),
            "manifest_path": str(self.manifest_path),
            "matches_found": self.matches_found,
            "name_inventory_path": str(self.name_inventory_path),
            "names_considered": self.names_considered,
            "names_discarded": self.names_discarded,
            "names_indexed": self.names_indexed,
            "output_paths": [str(path) for path in self.output_paths],
            "polygons_read": self.polygons_read,
            "unique_polygons_matched": self.unique_polygons_matched,
        }
