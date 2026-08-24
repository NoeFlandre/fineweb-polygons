from fineweb_polygons.normalization import has_monaco_marker, normalize_for_search


def test_normalization_ignores_case_and_repeated_separators() -> None:
    assert normalize_for_search("  Stade  Louis-II\n") == "stade louis ii"


def test_normalization_decodes_url_escapes() -> None:
    assert normalize_for_search("https://example.test/Palais%20Monaco") == (
        "https example test palais monaco"
    )


def test_normalization_returns_empty_for_none() -> None:
    assert normalize_for_search(None) == ""


def test_monaco_marker_is_case_insensitive_and_url_aware() -> None:
    assert has_monaco_marker("MONACO")
    encoded = "Principality%20of%20%4Donaco"
    assert has_monaco_marker(encoded, decode_url=True)
    assert not has_monaco_marker(encoded, decode_url=False)
    assert not has_monaco_marker(encoded)
    assert not has_monaco_marker(None)
