from pathlib import Path

from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.polygons import _NamedPolygonHandler, read_named_polygon_profiles

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
  <relation id="21">
    <member type="way" ref="10" role="outer" />
    <tag k="type" v="boundary" />
  </relation>
</osm>
"""


def test_read_named_polygon_profiles_keeps_only_named_polygon_entities(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "mini.osm"
    pbf.write_text(MINI_OSM_XML, encoding="utf-8")

    result = read_named_polygon_profiles(pbf)

    assert [(profile.polygon_id, profile.name) for profile in result.profiles] == [
        ("relation/20", "Named Relation"),
        ("way/10", "Named Way"),
    ]
    assert result.named_count == 2
    assert result.unnamed_count == 2


def test_read_named_polygon_profiles_disables_location_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_apply_file(self, path, *, locations):
        captured["path"] = path
        captured["locations"] = locations
        self.profiles = [
            PolygonProfile.create("way/2", "B"),
            PolygonProfile.create("way/1", "A"),
        ]
        self.named_count = 2
        self.unnamed_count = 0

    monkeypatch.setattr(_NamedPolygonHandler, "apply_file", fake_apply_file)

    pbf = tmp_path / "mini.osm.pbf"
    result = read_named_polygon_profiles(pbf)

    assert captured == {"path": str(pbf), "locations": False}
    assert [profile.polygon_id for profile in result.profiles] == [
        "way/1",
        "way/2",
    ]
