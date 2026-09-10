"""Versioned records for Direction 2 lexical V3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fineweb_polygons.directions.lexical.v2.models import OUTPUT_COLUMNS_V2

DIRECTION_V3_VERSION = "direction-2-lexical-v3"
HF_CONFIG_NAME_V3 = "direction_2_lexical_v3"
DATA_PREFIX = "data/direction-2-lexical/v3"
OUTPUT_COLUMNS_V3 = (
    *OUTPUT_COLUMNS_V2,
    "decision_tier",
    "evidence_score",
    "evidence_reasons",
    "name_in_url",
    "country_in_sentence",
    "country_in_context",
    "same_polygon_alias_nearby",
    "other_polygon_name_nearby",
)
COUNTRY_NAMES = {
    "monaco": "Monaco",
    "liechtenstein": "Liechtenstein",
}
DecisionTier = Literal["high_confidence", "possible", "rejected"]


@dataclass(frozen=True, slots=True)
class Direction2V3RunConfig:
    """Inputs and output locations for a deterministic V3 run."""

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
            raise ValueError("Direction 2 V3 input paths must be different")


@dataclass(frozen=True, slots=True)
class Direction2V3CountrySummary:
    """Counts and digest for one V3 source result."""

    source_key: str
    output_path: Path
    polygons_read: int
    names_indexed: int
    matches_found: int
    high_confidence_matches: int
    possible_matches: int
    rejected_matches: int
    unique_polygons_matched: int
    result_sha256: str

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible summary."""
        return {
            "high_confidence_matches": self.high_confidence_matches,
            "matches_found": self.matches_found,
            "names_indexed": self.names_indexed,
            "output_path": str(self.output_path),
            "polygons_read": self.polygons_read,
            "possible_matches": self.possible_matches,
            "rejected_matches": self.rejected_matches,
            "result_sha256": self.result_sha256,
            "unique_polygons_matched": self.unique_polygons_matched,
        }


@dataclass(frozen=True, slots=True)
class Direction2V3RunSummary:
    """Stable counters and artifacts produced by a V3 run."""

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
    high_confidence_matches: int
    possible_matches: int
    rejected_matches: int
    unique_polygons_matched: int
    country_summaries: tuple[Direction2V3CountrySummary, ...]
    direction: str = DIRECTION_V3_VERSION

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
            "generic_names": self.generic_names,
            "high_confidence_matches": self.high_confidence_matches,
            "log_path": str(self.log_path),
            "manifest_path": str(self.manifest_path),
            "matches_found": self.matches_found,
            "name_inventory_path": str(self.name_inventory_path),
            "names_considered": self.names_considered,
            "names_discarded": self.names_discarded,
            "names_indexed": self.names_indexed,
            "output_paths": [str(path) for path in self.output_paths],
            "possible_matches": self.possible_matches,
            "polygons_read": self.polygons_read,
            "rejected_matches": self.rejected_matches,
            "unique_polygons_matched": self.unique_polygons_matched,
        }
