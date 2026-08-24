from pathlib import Path
from typing import Any, cast

import fineweb_polygons.v2_profiles as v2_module
from fineweb_polygons.polygons import (
    read_named_polygon_profiles,
    read_v2_polygon_profiles,
)
from fineweb_polygons.v2_profiles import (
    _AreaMaterialization,
    _AreaRecord,
    _is_coordinate,
    _is_in_monaco,
    _is_meaningful_name,
    _is_monaco_boundary,
    _materialize_area,
    _materialize_named_area,
    _point_in_polygon,
    _point_in_ring,
    _read_areas,
    _rings,
)

MINI_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="43.70" lon="7.40" />
  <node id="2" lat="43.70" lon="7.41" />
  <node id="3" lat="43.71" lon="7.41" />
  <node id="4" lat="43.71" lon="7.40" />
  <node id="5" lat="43.72" lon="7.42" />
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
    <nd ref="5" />
    <tag k="name" v="Open Way" />
  </way>
  <relation id="20">
    <member type="way" ref="10" role="outer" />
    <tag k="type" v="multipolygon" />
    <tag k="name" v="Named Relation" />
  </relation>
</osm>
"""


def test_read_named_polygon_profiles_keeps_only_named_polygon_entities(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "mini.osm"
    pbf.write_text(MINI_OSM_XML, encoding="utf-8")

    result = read_named_polygon_profiles(pbf)

    assert [profile.polygon_id for profile in result.profiles] == [
        "relation/20",
        "way/10",
    ]
    assert result.named_count == 2
    assert result.unnamed_count == 1


V2_OSM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="test">
  <node id="1" lat="43.70" lon="7.40" />
  <node id="2" lat="43.70" lon="7.50" />
  <node id="3" lat="43.80" lon="7.50" />
  <node id="4" lat="43.80" lon="7.40" />
  <node id="5" lat="43.72" lon="7.42" />
  <node id="6" lat="43.72" lon="7.43" />
  <node id="7" lat="43.73" lon="7.43" />
  <node id="8" lat="43.73" lon="7.42" />
  <node id="9" lat="43.74" lon="8.00" />
  <node id="10" lat="43.74" lon="8.01" />
  <node id="11" lat="43.75" lon="8.01" />
  <node id="12" lat="43.75" lon="8.00" />
  <way id="100">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
  </way>
  <way id="101">
    <nd ref="5" />
    <nd ref="6" />
    <nd ref="7" />
    <nd ref="8" />
    <nd ref="5" />
    <tag k="name" v="Palais" />
  </way>
  <way id="102">
    <nd ref="5" />
    <nd ref="6" />
    <nd ref="7" />
    <nd ref="8" />
    <nd ref="5" />
    <tag k="name" v="A" />
  </way>
  <way id="103">
    <nd ref="9" />
    <nd ref="10" />
    <nd ref="11" />
    <nd ref="12" />
    <nd ref="9" />
    <tag k="name" v="Nice" />
  </way>
  <way id="104">
    <nd ref="5" />
    <nd ref="6" />
    <nd ref="7" />
    <nd ref="8" />
    <nd ref="5" />
    <tag k="building" v="yes" />
  </way>
  <relation id="200">
    <member type="way" ref="100" role="outer" />
    <tag k="type" v="multipolygon" />
    <tag k="boundary" v="administrative" />
    <tag k="admin_level" v="8" />
    <tag k="place" v="city" />
    <tag k="name" v="Monaco" />
  </relation>
</osm>
"""


def test_read_v2_polygon_profiles_keeps_meaningful_names_inside_monaco(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "mini.osm"
    pbf.write_text(V2_OSM_XML, encoding="utf-8")

    result = read_v2_polygon_profiles(pbf)

    assert [profile.polygon_id for profile in result.profiles] == [
        "relation/200",
        "way/101",
    ]
    assert [profile.name for profile in result.profiles] == ["Monaco", "Palais"]
    assert result.named_count == 2
    assert result.unnamed_count == 1
    assert result.filtered_count == 2


def test_read_v2_polygon_profiles_requires_the_boundary(tmp_path: Path) -> None:
    pbf = tmp_path / "without-boundary.osm"
    pbf.write_text(MINI_OSM_XML, encoding="utf-8")

    try:
        read_v2_polygon_profiles(pbf)
    except ValueError as error:
        assert str(error) == "V2 profile requires a Monaco admin_level=8 city boundary"
    else:
        raise AssertionError("expected a missing-boundary error")


def test_materialize_area_rejects_non_area_values() -> None:
    materialization = _materialize_area(cast(Any, object()), object())

    assert materialization.is_area is False
    assert materialization.has_name is False
    assert materialization.record is None


class _FakeArea:
    def __init__(self, *, from_way: bool = True, polygon_id: int = 42) -> None:
        self.tags = {"name": "Palais"}
        self._from_way = from_way
        self._polygon_id = polygon_id

    def from_way(self) -> bool:
        return self._from_way

    def orig_id(self) -> int:
        return self._polygon_id


class _FakeFactory:
    def __init__(self, geojson: str) -> None:
        self.geojson = geojson

    def create_multipolygon(self, area: object) -> str:
        return self.geojson


def test_materialize_area_marks_a_real_unnamed_area(monkeypatch) -> None:
    area = _FakeArea()
    area.tags = {}
    monkeypatch.setattr(v2_module.osmium.osm, "Area", _FakeArea)

    materialization = _materialize_area(cast(Any, _FakeFactory("{}")), area)

    assert materialization.is_area is True
    assert materialization.has_name is False
    assert materialization.record is None


def _materialized_record(polygon_id: str) -> _AreaMaterialization:
    return _AreaMaterialization(
        is_area=True,
        has_name=True,
        record=_AreaRecord(
            polygon_id=polygon_id,
            name="Palais",
            tags={},
            geometry={},
            representative_point=(0.0, 0.0),
        ),
    )


def _read_fake_areas(monkeypatch, materializations):
    class FakeProcessor:
        def with_locations(self):
            return self

        def with_areas(self):
            return tuple(range(len(materializations)))

    remaining = iter(materializations)
    monkeypatch.setattr(v2_module.osmium, "FileProcessor", lambda path: FakeProcessor())
    monkeypatch.setattr(
        v2_module,
        "_materialize_area",
        lambda factory, area: next(remaining),
    )
    return _read_areas(Path("unused.osm.pbf"))


def test_read_areas_counts_all_unnamed_areas_and_continues(monkeypatch) -> None:
    records, boundary, source_named_count, unnamed_count = _read_fake_areas(
        monkeypatch,
        [
            _AreaMaterialization(True, False, None),
            _AreaMaterialization(True, False, None),
            _materialized_record("way/3"),
        ],
    )

    assert [record.polygon_id for record in records] == ["way/3"]
    assert boundary is None
    assert source_named_count == 1
    assert unnamed_count == 2


def test_read_areas_continues_after_named_geometry_failure(monkeypatch) -> None:
    records, _, source_named_count, unnamed_count = _read_fake_areas(
        monkeypatch,
        [
            _AreaMaterialization(True, True, None),
            _materialized_record("way/2"),
        ],
    )

    assert [record.polygon_id for record in records] == ["way/2"]
    assert source_named_count == 2
    assert unnamed_count == 0


def test_materialize_named_area_keeps_geometry_and_way_identity() -> None:
    materialization = _materialize_named_area(
        cast(
            Any,
            _FakeFactory(
                '{"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [0, 2]]]}'
            ),
        ),
        cast(Any, _FakeArea()),
        {"name": "Palais"},
        "Palais",
    )

    assert materialization.is_area is True
    assert materialization.has_name is True
    assert materialization.record is not None
    assert materialization.record.polygon_id == "way/42"
    assert materialization.record.representative_point == (2 / 3, 2 / 3)


def test_materialize_named_area_uses_relation_identity() -> None:
    materialization = _materialize_named_area(
        cast(
            Any,
            _FakeFactory(
                '{"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [0, 2]]]}'
            ),
        ),
        cast(Any, _FakeArea(from_way=False)),
        {"name": "Palais"},
        "Palais",
    )

    assert materialization.record is not None
    assert materialization.record.polygon_id == "relation/42"


def test_materialize_named_area_drops_invalid_or_empty_geometry() -> None:
    invalid = _materialize_named_area(
        cast(Any, _FakeFactory("not-json")),
        cast(Any, _FakeArea()),
        {"name": "Palais"},
        "Palais",
    )
    empty = _materialize_named_area(
        cast(Any, _FakeFactory('{"type": "Polygon", "coordinates": []}')),
        cast(Any, _FakeArea()),
        {"name": "Palais"},
        "Palais",
    )

    for materialization in (invalid, empty):
        assert materialization.is_area is True
        assert materialization.has_name is True
        assert materialization.record is None


def test_monaco_boundary_requires_all_identity_tags() -> None:
    valid = {"boundary": "administrative", "admin_level": "8", "place": "city"}
    assert _is_monaco_boundary("Monaco", valid) is True
    assert _is_monaco_boundary("Monaco", {**valid, "boundary": "other"}) is False
    assert _is_monaco_boundary("Monaco", {**valid, "admin_level": "7"}) is False
    assert _is_monaco_boundary("Other", valid) is False


def test_meaningful_name_checks_separator_and_three_character_boundary() -> None:
    assert _is_meaningful_name("ABC") is True
    assert _is_meaningful_name("A-") is False


def test_in_monaco_keeps_the_boundary_even_when_point_is_outside() -> None:
    record = _AreaRecord(
        polygon_id="relation/1",
        name="Monaco",
        tags={"boundary": "administrative", "admin_level": "8", "place": "city"},
        geometry={},
        representative_point=(99.0, 99.0),
    )

    assert _is_in_monaco(record, {}) is True


def test_coordinate_requires_only_two_numeric_values() -> None:
    assert _is_coordinate([1, 2]) is True
    assert _is_coordinate([1, 2, "not-a-coordinate"]) is False


def test_rings_rejects_non_lists() -> None:
    assert _rings({"not": "a polygon"}) == ()


def test_point_in_polygon_rejects_invalid_and_hole_points() -> None:
    outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    hole = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]

    assert _point_in_polygon((20, 20), []) is False
    assert _point_in_polygon((2, 2), [outer, hole]) is True
    assert _point_in_polygon((5, 5), [outer, hole]) is False


def test_point_in_ring_handles_triangle_and_short_rings() -> None:
    triangle = ((0.0, 0.0), (4.0, 0.0), (0.0, 4.0))

    assert _point_in_ring((1.0, 1.0), triangle) is True
    assert _point_in_ring((3.0, 3.0), triangle) is False
    assert _point_in_ring((1.0, 1.0), ((0.0, 0.0), (1.0, 0.0))) is False


def test_point_in_ring_uses_the_previous_edge_for_asymmetric_rings() -> None:
    assert (
        _point_in_ring((1.0, 2.0), ((0.0, 1.0), (6.0, 2.0), (5.0, 6.0), (1.0, 5.0)))
        is True
    )
    assert (
        _point_in_ring((3.0, 4.0), ((0.0, 1.0), (6.0, 2.0), (5.0, 6.0), (1.0, 5.0)))
        is True
    )
    assert (
        _point_in_ring((3.0, 6.0), ((2.0, 3.0), (8.0, 3.0), (6.0, 9.0), (2.0, 7.0)))
        is True
    )
