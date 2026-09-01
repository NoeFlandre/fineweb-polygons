"""Read named and unnamed OSM areas with lightweight geometry metadata."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeGuard, cast

import osmium
from osmium.geom import GeoJSONFactory

from fineweb_polygons.direction2.models import PolygonRecord, PolygonSource

_GEOMETRY_EPSILON = 1e-15


def read_polygon_records(sources: Sequence[PolygonSource]) -> tuple[PolygonRecord, ...]:
    """Read every area emitted by osmium for the supplied extracts."""
    records = [record for source in sources for record in _read_source(source)]
    return tuple(sorted(records, key=lambda record: record.polygon_id))


def _read_source(source: PolygonSource) -> tuple[PolygonRecord, ...]:
    factory = GeoJSONFactory()
    records: list[PolygonRecord] = []
    entities = osmium.FileProcessor(str(source.path)).with_areas()
    for area in _as_entities(entities):
        if not area.is_area():
            continue
        records.append(_area_record(source, area, factory))
    return tuple(records)


def _area_record(
    source: PolygonSource,
    area: Any,
    factory: GeoJSONFactory,
) -> PolygonRecord:
    tags = _sorted_tags(area.tags)
    name = dict(tags).get("name", "")
    aliases = _aliases(tags)
    geometry = json.loads(factory.create_multipolygon(area))
    centroid = centroid_from_geojson(geometry)
    object_type = "way" if area.from_way() else "relation"
    polygon_id = f"{source.key}/{object_type}/{area.orig_id()}"
    return PolygonRecord(
        polygon_id=polygon_id,
        source_key=source.key,
        name=name,
        aliases=aliases,
        tags=tags,
        centroid=centroid,
    )


def _sorted_tags(tags: Any) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in dict(tags).items()))


def _aliases(tags: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    aliases: list[str] = []
    for key, value in tags:
        if not key.startswith("name:") or not value.strip() or value in seen:
            continue
        seen.add(value)
        aliases.append(value)
    return tuple(aliases)


def centroid_from_geojson(geometry: Mapping[str, object]) -> tuple[float, float] | None:
    """Return an area-weighted planar centroid as ``(longitude, latitude)``."""
    coordinates = geometry.get("coordinates")
    if not _is_sequence(coordinates):
        return None
    return _geometry_centroid(coordinates)


def _geometry_centroid(coordinates: Sequence[object]) -> tuple[float, float] | None:
    total_area = 0.0
    total_longitude = 0.0
    total_latitude = 0.0
    outer_points: list[tuple[float, float]] = []
    for polygon in coordinates:
        area, longitude, latitude, points = _polygon_moment(polygon)
        total_area += area
        total_longitude += longitude
        total_latitude += latitude
        outer_points.extend(points)

    if abs(total_area) > _GEOMETRY_EPSILON:
        return (total_longitude / total_area, total_latitude / total_area)
    return _mean_point(outer_points)


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _ring_points(ring: object) -> list[tuple[float, float]]:
    if not _is_sequence(ring):
        return []
    points = [point for point in (_coordinate(point) for point in ring) if point]
    return _without_closing_point(points)


def _coordinate(point: object) -> tuple[float, float] | None:
    if not _is_sequence(point) or len(point) < 2:
        return None
    try:
        return (_as_float(point[0]), _as_float(point[1]))
    except (TypeError, ValueError):
        return None


def _without_closing_point(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def _polygon_moment(
    polygon: object,
) -> tuple[float, float, float, list[tuple[float, float]]]:
    if not _is_sequence(polygon):
        return 0.0, 0.0, 0.0, []
    total_area = 0.0
    total_longitude = 0.0
    total_latitude = 0.0
    outer_points: list[tuple[float, float]] = []
    for ring_index, ring in enumerate(polygon):
        points = _ring_points(ring)
        area, longitude, latitude = _ring_moment(points, is_outer=ring_index == 0)
        total_area += area
        total_longitude += longitude
        total_latitude += latitude
        if ring_index == 0:
            outer_points.extend(points)
    return total_area, total_longitude, total_latitude, outer_points


def _ring_moment(
    points: Sequence[tuple[float, float]],
    *,
    is_outer: bool,
) -> tuple[float, float, float]:
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    area, longitude, latitude = _ring_centroid(points)
    weight = abs(area) * (1.0 if is_outer else -1.0)
    return weight, longitude * weight, latitude * weight


def _ring_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    cross_sum = 0.0
    longitude_sum = 0.0
    latitude_sum = 0.0
    for first, second in zip(points, (*points[1:], points[0]), strict=True):
        cross = first[0] * second[1] - second[0] * first[1]
        cross_sum += cross
        longitude_sum += (first[0] + second[0]) * cross
        latitude_sum += (first[1] + second[1]) * cross
    area = cross_sum / 2.0
    if abs(area) <= _GEOMETRY_EPSILON:
        return area, points[0][0], points[0][1]
    return area, longitude_sum / (6.0 * area), latitude_sum / (6.0 * area)


def _mean_point(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _as_entities(value: object) -> Iterator[Any]:
    return cast(Iterator[Any], value)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        raise TypeError("coordinate must be numeric")
    return float(value)
