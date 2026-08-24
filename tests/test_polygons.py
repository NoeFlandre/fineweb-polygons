from pathlib import Path

from fineweb_polygons.polygons import read_named_polygon_profiles

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
