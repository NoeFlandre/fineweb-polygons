from fineweb_polygons.normalization import has_monaco_marker, normalize_for_search


def test_normalization_ignores_case_and_repeated_separators() -> None:
    assert normalize_for_search("  Stade  Louis-II\n") == "stade louis ii"


def test_normalization_decodes_url_escapes() -> None:
    assert normalize_for_search("https://example.test/Palais%20Monaco") == (
        "https example test palais monaco"
    )


def test_normalization_casefolds_after_url_decoding() -> None:
    assert normalize_for_search("https://example.test/%4d%4f%4e%41%43%4f") == (
        "https example test monaco"
    )


def test_normalization_returns_empty_for_none() -> None:
    assert normalize_for_search(None) == ""


def test_monaco_marker_decodes_urls_and_rejects_none() -> None:
    encoded = "https://example.test/%4d%4f%4e%41%43%4f"

    assert has_monaco_marker(encoded) is True
    assert has_monaco_marker(encoded, decode_url=False) is False
    assert has_monaco_marker(None) is False
