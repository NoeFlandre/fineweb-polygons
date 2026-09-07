"""OSM records and evidence shapes shared by every lexical-candidate version."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
