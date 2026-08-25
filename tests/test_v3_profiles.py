from pathlib import Path

import fineweb_polygons.v3_profiles as v3_module
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.v3_profiles import (
    _AreaSelection,
    _select_area,
    read_v3_polygon_profiles,
)

MINI_V3_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="43.70" lon="7.40" />
  <node id="2" lat="43.70" lon="7.41" />
  <node id="3" lat="43.71" lon="7.41" />
  <node id="4" lat="43.71" lon="7.40" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="Named Way" />
  </way>
  <way id="11">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
  </way>
  <way id="12">
    <nd ref="1" />
    <nd ref="2" />
    <tag k="name" v="Open Way" />
  </way>
  <relation id="20">
    <member type="way" ref="10" role="outer" />
    <tag k="type" v="multipolygon" />
    <tag k="name" v="Named Relation" />
  </relation>
</osm>
"""


MEANINGFUL_NAMES_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="43.70" lon="7.40" />
  <node id="2" lat="43.70" lon="7.41" />
  <node id="3" lat="43.71" lon="7.41" />
  <node id="4" lat="43.71" lon="7.40" />
  <way id="101">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="Palais" />
  </way>
  <way id="102">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v=" palais " />
  </way>
  <way id="103">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="A" />
  </way>
  <way id="104">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="2" />
  </way>
  <way id="105">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="123" />
  </way>
</osm>
"""


def test_v3_keeps_named_closed_ways_and_polygon_relations_without_boundary(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "mini.osm"
    pbf.write_text(MINI_V3_OSM_XML, encoding="utf-8")

    result = read_v3_polygon_profiles(pbf)

    assert [profile.polygon_id for profile in result.profiles] == [
        "relation/20",
        "way/10",
    ]
    assert [profile.name for profile in result.profiles] == [
        "Named Relation",
        "Named Way",
    ]
    assert result.named_count == 2
    assert result.unnamed_count == 0
    assert result.filtered_count == 0


def test_v3_keeps_meaningful_names_and_deduplicates_normalized_names(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "meaningful.osm"
    pbf.write_text(MEANINGFUL_NAMES_OSM_XML, encoding="utf-8")

    result = read_v3_polygon_profiles(pbf)

    assert [profile.polygon_id for profile in result.profiles] == ["way/101"]
    assert [profile.name for profile in result.profiles] == ["Palais"]
    assert result.named_count == 1
    assert result.unnamed_count == 0
    assert result.filtered_count == 4
    assert result.name_occurrences == (("palais", 2),)


class _FakeArea:
    def __init__(self, name: str | None = "Palais") -> None:
        self.tags = {} if name is None else {"name": name}

    def from_way(self) -> bool:
        return True

    def orig_id(self) -> int:
        return 42


def test_v3_area_selection_distinguishes_non_areas_and_unnamed_areas(
    monkeypatch,
) -> None:
    monkeypatch.setattr(v3_module.osmium.osm, "Area", _FakeArea)

    assert _select_area(object()) == _AreaSelection(False, False, None)
    assert _select_area(_FakeArea(None)) == _AreaSelection(True, False, None)


def test_v3_area_reader_counts_all_unnamed_areas_and_continues(
    monkeypatch,
) -> None:
    paths: list[str] = []

    class _FakeProcessor:
        def __init__(self, path: str) -> None:
            paths.append(path)

        def with_locations(self) -> "_FakeProcessor":
            return self

        def with_areas(self) -> list[object]:
            return [object(), object(), object()]

    selections = iter(
        (
            _AreaSelection(True, False, None),
            _AreaSelection(True, False, None),
            _AreaSelection(True, True, PolygonProfile.create("way/9", "Palais")),
        )
    )
    monkeypatch.setattr(v3_module.osmium, "FileProcessor", _FakeProcessor)
    monkeypatch.setattr(v3_module, "_select_area", lambda area: next(selections))

    profiles, named_count, unnamed_count = v3_module._read_area_profiles(
        Path("mini.osm.pbf")
    )

    assert profiles == [PolygonProfile.create("way/9", "Palais")]
    assert named_count == 1
    assert unnamed_count == 2
    assert paths == ["mini.osm.pbf"]
