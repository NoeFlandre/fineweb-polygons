from fineweb_polygons.models import PolygonProfile


def test_polygon_profile_retains_original_and_normalized_name() -> None:
    profile = PolygonProfile.create("way/7", "Palais  MONACO")

    assert profile.polygon_id == "way/7"
    assert profile.name == "Palais  MONACO"
    assert profile.normalized_name == "palais monaco"
