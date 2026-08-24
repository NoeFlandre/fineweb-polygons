from fineweb_polygons.normalization import normalize_for_search


def test_normalization_ignores_case_and_repeated_separators() -> None:
    assert normalize_for_search("  Stade  Louis-II\n") == "stade louis ii"


def test_normalization_decodes_url_escapes() -> None:
    assert normalize_for_search("https://example.test/Palais%20Monaco") == (
        "https example test palais monaco"
    )


def test_normalization_returns_empty_for_none() -> None:
    assert normalize_for_search(None) == ""
