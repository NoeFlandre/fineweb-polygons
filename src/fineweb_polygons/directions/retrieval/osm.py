"""Read named OSM areas and apply each retrieval version's polygon profile."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import osmium
import osmium.geom
import osmium.osm

from fineweb_polygons.core.normalization import normalize_for_search
from fineweb_polygons.directions.retrieval.models import PolygonProfile

_POLYGON_RELATION_TYPES = frozenset({"boundary", "multipolygon"})


@dataclass(frozen=True, slots=True)
class PolygonReadResult:
    """Named profiles and counts from one OSM extract."""

    profiles: tuple[PolygonProfile, ...]
    named_count: int
    unnamed_count: int
    filtered_count: int = 0
    name_occurrences: tuple[tuple[str, int], ...] = ()


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


_NAME_SEPARATOR_RE = re.compile(r"[\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class _AreaRecord:
    """Materialized OSM area data that survives the parser callback."""

    polygon_id: str
    name: str
    tags: dict[str, str]
    geometry: dict[str, Any]
    representative_point: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _AreaMaterialization:
    """Parser result with enough state for source counters."""

    is_area: bool
    has_name: bool
    record: _AreaRecord | None


def read_v2_polygon_profiles(pbf_path: Path) -> PolygonReadResult:
    """Read meaningful, in-boundary named areas from a raw OSM extract."""
    records, boundary, source_named_count, unnamed_count = _read_areas(pbf_path)
    if boundary is None:
        raise ValueError("V2 profile requires a Monaco admin_level=8 city boundary")
    candidates = [
        PolygonProfile.create(record.polygon_id, record.name)
        for record in records
        if _is_meaningful_name(record.name) and _is_in_monaco(record, boundary)
    ]
    profiles = _deduplicate_profiles(candidates)
    return PolygonReadResult(
        profiles=profiles,
        named_count=len(profiles),
        unnamed_count=unnamed_count,
        filtered_count=source_named_count - len(profiles),
    )


def _read_areas(
    pbf_path: Path,
) -> tuple[list[_AreaRecord], dict[str, Any] | None, int, int]:
    factory = osmium.geom.GeoJSONFactory()
    records: list[_AreaRecord] = []
    boundary: dict[str, Any] | None = None
    source_named_count = 0
    unnamed_count = 0
    for area in osmium.FileProcessor(str(pbf_path)).with_locations().with_areas():
        materialization = _materialize_area(factory, area)
        if not materialization.is_area:
            continue
        if not materialization.has_name:
            unnamed_count += 1
            continue
        source_named_count += 1
        record = materialization.record
        if record is None:
            continue
        records.append(record)
        boundary = _boundary_from_record(record, boundary)
    return records, boundary, source_named_count, unnamed_count


def _materialize_area(
    factory: osmium.geom.GeoJSONFactory, area: Any
) -> _AreaMaterialization:
    if not isinstance(area, osmium.osm.Area):
        return _AreaMaterialization(False, False, None)
    tags = dict(area.tags)
    name = tags.get("name")
    if not name:
        return _AreaMaterialization(True, False, None)
    return _materialize_named_area(factory, area, tags, name)


def _materialize_named_area(
    factory: osmium.geom.GeoJSONFactory,
    area: osmium.osm.Area,
    tags: dict[str, str],
    name: str,
) -> _AreaMaterialization:
    geometry = _read_geometry(factory, area)
    representative_point = None if geometry is None else _representative_point(geometry)
    if geometry is None or representative_point is None:
        return _AreaMaterialization(True, True, None)
    polygon_kind = "way" if area.from_way() else "relation"
    return _AreaMaterialization(
        True,
        True,
        _AreaRecord(
            polygon_id=f"{polygon_kind}/{area.orig_id()}",
            name=name,
            tags=tags,
            geometry=geometry,
            representative_point=representative_point,
        ),
    )


def _boundary_from_record(
    record: _AreaRecord, current: dict[str, Any] | None
) -> dict[str, Any] | None:
    if _is_monaco_boundary(record.name, record.tags):
        return record.geometry
    return current


def _read_geometry(
    factory: osmium.geom.GeoJSONFactory, area: Any
) -> dict[str, Any] | None:
    try:
        geometry = json.loads(factory.create_multipolygon(area))
    except (TypeError, ValueError, RuntimeError):
        return None
    return geometry if isinstance(geometry, dict) else None


def _is_monaco_boundary(name: str, tags: Mapping[str, str]) -> bool:
    return (
        normalize_for_search(name) == "monaco"
        and tags.get("boundary") == "administrative"
        and tags.get("admin_level") == "8"
        and tags.get("place") == "city"
    )


def _is_meaningful_name(name: str) -> bool:
    normalized = _NAME_SEPARATOR_RE.sub(
        " ", unicodedata.normalize("NFKC", name).casefold()
    ).strip()
    return (
        len(normalized) >= 3
        and not normalized.isdecimal()
        and any(character.isalpha() for character in normalized)
    )


def _is_in_monaco(record: _AreaRecord, boundary: Mapping[str, Any]) -> bool:
    if _is_monaco_boundary(record.name, record.tags):
        return True
    return _point_in_geometry(record.representative_point, boundary)


def _deduplicate_profiles(
    profiles: list[PolygonProfile],
) -> tuple[PolygonProfile, ...]:
    by_name: dict[str, PolygonProfile] = {}
    for profile in sorted(profiles, key=lambda item: item.polygon_id):
        by_name.setdefault(profile.normalized_name, profile)
    return tuple(sorted(by_name.values(), key=lambda item: item.polygon_id))


def _representative_point(
    geometry: Mapping[str, Any],
) -> tuple[float, float] | None:
    points = _points(geometry.get("coordinates"))
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _points(value: object) -> tuple[tuple[float, float], ...]:
    if _is_coordinate(value):
        assert isinstance(value, list)
        return ((float(value[0]), float(value[1])),)
    return _nested_points(value)


def _is_coordinate(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(item, (int, float)) for item in value)
    )


def _nested_points(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, list):
        return ()
    points: list[tuple[float, float]] = []
    for child in value:
        points.extend(_points(child))
    return tuple(points)


def _point_in_geometry(point: tuple[float, float], geometry: Mapping[str, Any]) -> bool:
    return any(
        _point_in_polygon(point, polygon)
        for polygon in _polygons(geometry.get("coordinates"))
    )


def _polygons(value: object) -> tuple[object, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _point_in_polygon(point: tuple[float, float], polygon: object) -> bool:
    rings = _rings(polygon)
    if not rings:
        return False
    outer, *holes = rings
    return _point_in_ring(point, outer) and not any(
        _point_in_ring(point, hole) for hole in holes
    )


def _rings(polygon: object) -> tuple[tuple[tuple[float, float], ...], ...]:
    if not isinstance(polygon, list) or not polygon:
        return ()
    return tuple(_points(ring) for ring in polygon)


def _point_in_ring(
    point: tuple[float, float], ring: tuple[tuple[float, float], ...]
) -> bool:
    if len(ring) < 3:
        return False
    x, y = point
    inside = False
    previous_x, previous_y = ring[-1]
    for current_x, current_y in ring:
        crosses = (previous_y > y) != (current_y > y)
        if crosses:
            boundary_x = (current_x - previous_x) * (y - previous_y) / (
                current_y - previous_y
            ) + previous_x
            if x < boundary_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


@dataclass(frozen=True, slots=True)
class _AreaSelection:
    is_area: bool
    has_name: bool
    profile: PolygonProfile | None


def read_v3_polygon_profiles(pbf_path: Path) -> PolygonReadResult:
    """Read meaningful named closed ways and valid polygon relations."""
    profiles, named_count, unnamed_count, name_occurrences = (
        _read_area_profiles_with_occurrences(pbf_path)
    )
    deduplicated = _deduplicate_profiles(profiles)
    return PolygonReadResult(
        profiles=deduplicated,
        named_count=len(deduplicated),
        unnamed_count=unnamed_count,
        filtered_count=named_count - len(deduplicated),
        name_occurrences=tuple(sorted(name_occurrences.items())),
    )


def _read_area_profiles_with_occurrences(
    pbf_path: Path,
) -> tuple[list[PolygonProfile], int, int, Counter[str]]:
    profiles: list[PolygonProfile] = []
    named_count = 0
    unnamed_count = 0
    name_occurrences: Counter[str] = Counter()
    for area in osmium.FileProcessor(str(pbf_path)).with_locations().with_areas():
        selection = _select_area(area)
        if not selection.is_area:
            continue
        if not selection.has_name:
            unnamed_count += 1
            continue
        named_count += 1
        if selection.profile is not None:
            profiles.append(selection.profile)
            name_occurrences[selection.profile.normalized_name] += 1
    return profiles, named_count, unnamed_count, name_occurrences


def _select_area(area: Any) -> _AreaSelection:
    if not isinstance(area, osmium.osm.Area):
        return _AreaSelection(False, False, None)
    name = area.tags.get("name")
    if not name:
        return _AreaSelection(True, False, None)
    if not _is_meaningful_name(name):
        return _AreaSelection(True, True, None)
    kind = "way" if area.from_way() else "relation"
    return _AreaSelection(
        True,
        True,
        PolygonProfile.create(f"{kind}/{area.orig_id()}", name),
    )
