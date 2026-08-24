"""Read the minimal named polygon profile from an OSM extract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import osmium

from fineweb_polygons.models import PolygonProfile

_POLYGON_RELATION_TYPES = frozenset({"boundary", "multipolygon"})


@dataclass(frozen=True, slots=True)
class PolygonReadResult:
    """Named profiles and counts from one OSM extract."""

    profiles: tuple[PolygonProfile, ...]
    named_count: int
    unnamed_count: int


class _NamedPolygonHandler(osmium.SimpleHandler):
    def __init__(self) -> None:
        super().__init__()
        self.profiles: list[PolygonProfile] = []
        self.named_count = 0
        self.unnamed_count = 0

    def way(self, entity: Any) -> None:
        if entity.is_closed():
            self._record("way", entity)

    def relation(self, entity: Any) -> None:
        if entity.tags.get("type") in _POLYGON_RELATION_TYPES:
            self._record("relation", entity)

    def _record(self, entity_kind: str, entity: Any) -> None:
        name = entity.tags.get("name")
        if not name:
            self.unnamed_count += 1
            return
        self.named_count += 1
        self.profiles.append(PolygonProfile.create(f"{entity_kind}/{entity.id}", name))


def read_named_polygon_profiles(pbf_path: Path) -> PolygonReadResult:
    """Read named closed ways and named polygon relations from an OSM file."""
    handler = _NamedPolygonHandler()
    handler.apply_file(str(pbf_path), locations=False)
    profiles = tuple(sorted(handler.profiles, key=lambda profile: profile.polygon_id))
    return PolygonReadResult(
        profiles=profiles,
        named_count=handler.named_count,
        unnamed_count=handler.unnamed_count,
    )
