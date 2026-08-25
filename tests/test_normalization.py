import fineweb_polygons.normalization as normalization
from fineweb_polygons.normalization import (
    has_context_marker,
    has_monaco_marker,
    normalize_for_search,
)


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


def test_context_marker_decodes_only_when_requested() -> None:
    encoded = "https://example.test/%4d%4f%4e%41%43%4f"

    assert has_context_marker(encoded, "Monaco") is True
    assert has_context_marker(encoded, "Monaco", decode_url=False) is False


def test_context_marker_normalizes_the_context_name_without_url_decoding(
    monkeypatch,
) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_normalize(value: object, *, decode_url: bool = True) -> str:
        calls.append((value, decode_url))
        return "monaco"

    monkeypatch.setattr(normalization, "normalize_for_search", fake_normalize)

    assert has_context_marker("text", "Monaco") is True
    assert calls == [("text", True), ("Monaco", False)]


def test_monaco_marker_keeps_the_canonical_context_name(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_context_marker(value, context_name, *, decode_url):
        captured.update(value=value, context_name=context_name, decode_url=decode_url)
        return True

    monkeypatch.setattr(normalization, "has_context_marker", fake_context_marker)

    assert has_monaco_marker("value", decode_url=False) is True
    assert captured == {
        "value": "value",
        "context_name": "Monaco",
        "decode_url": False,
    }
