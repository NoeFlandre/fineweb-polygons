"""Raw-PBF profile filtering for the V3 retrieval contract."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import osmium
import osmium.osm

from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.polygons import PolygonReadResult
from fineweb_polygons.v2_profiles import _deduplicate_profiles, _is_meaningful_name


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


def _read_area_profiles(
    pbf_path: Path,
) -> tuple[list[PolygonProfile], int, int]:
    profiles, named_count, unnamed_count, _ = _read_area_profiles_with_occurrences(
        pbf_path
    )
    return profiles, named_count, unnamed_count


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
