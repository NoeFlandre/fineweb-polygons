from typing import cast

from fineweb_polygons.matching import (
    EvidenceMatcher,
    _context_candidates,
    _excerpt,
    _find_context,
    _find_names,
    _MultiPatternMatcher,
)
from fineweb_polygons.models import FineWebDocument, PolygonProfile


def test_name_in_url_and_context_in_text_is_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Casino de Monte Carlo")])
    document = FineWebDocument(
        row_index=4,
        document_id="doc-4",
        text="A report from Monaco describes the venue.",
        url="https://example.test/casino-de-monte-carlo",
    )

    matches = matcher.match(document)

    assert len(matches) == 1
    assert matches[0].matched_fields == ("url",)
    assert matches[0].context_fields == ("text",)
    assert matches[0].url_excerpt == document.url


def test_name_in_text_and_context_in_url_is_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(
        row_index=5,
        document_id=None,
        text="Fontvieille has a new report.",
        url="https://monaco.example.test/report",
    )

    assert matcher.match(document)[0].polygon_id == "way/1"


def test_url_encoded_context_is_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(
        row_index=7,
        document_id=None,
        text="Fontvieille has a new report.",
        url="https://example.test/Principality%20of%20Monaco/report",
    )

    assert matcher.match(document)[0].context_fields == ("url",)


def test_context_marker_without_an_accepted_phrase_is_not_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(8, None, "Fontvieille near Monacology.", "")

    assert matcher.match(document) == ()


def test_context_without_a_polygon_name_is_not_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(9, None, "A report from Monaco.", "")

    assert matcher.match(document) == ()


def test_name_without_context_is_not_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(6, "doc-6", "Fontvieille has a report.", "")

    assert matcher.match(document) == ()


def test_low_level_matcher_decodes_url_escapes_only_when_requested() -> None:
    matcher = _MultiPatternMatcher(("casino de monte carlo",))
    url = "https://example.test/Casino%20de%20Monte%20Carlo"

    assert matcher.find(url, decode_url=True) == frozenset({"casino de monte carlo"})
    assert matcher.find(url, decode_url=False) == frozenset()


def test_low_level_matcher_does_not_decode_urls_by_default() -> None:
    matcher = _MultiPatternMatcher(("casino de monte carlo",))

    assert (
        matcher.find("https://example.test/Casino%20de%20Monte%20Carlo") == frozenset()
    )


def test_context_helper_prefers_the_longest_context_phrase() -> None:
    class FakeMatcher:
        def find(self, value: str, *, decode_url: bool = False) -> frozenset[str]:
            return frozenset({"z", "aaaaaa"})

    contexts, phrase = _find_context(
        {"text": "Monaco", "url": ""}, cast(_MultiPatternMatcher, FakeMatcher())
    )

    assert contexts == {"text": frozenset({"z", "aaaaaa"})}
    assert phrase == "aaaaaa"


def test_context_helpers_keep_url_decoding_and_prefer_longest_phrase() -> None:
    values = {
        "text": "A report from Monaco.",
        "url": "https://example.test/Principality%20of%20%4Donaco/report",
    }
    matcher = _MultiPatternMatcher(("monaco", "principality of monaco"))

    assert _context_candidates(values) == ("text", "url")
    contexts, phrase = _find_context(values, matcher)

    assert contexts["text"] == frozenset({"monaco"})
    assert contexts["url"] == frozenset({"monaco", "principality of monaco"})
    assert phrase == "principality of monaco"


def test_name_helper_decodes_a_name_in_a_url() -> None:
    values = {
        "text": "No polygon name here.",
        "url": "https://example.test/Casino%20de%20Monte%20Carlo",
    }
    matcher = _MultiPatternMatcher(("casino de monte carlo",))

    assert _find_names(values, matcher) == {
        "text": frozenset(),
        "url": frozenset({"casino de monte carlo"}),
    }


def test_match_orders_same_name_profiles_and_preserves_all_evidence() -> None:
    matcher = EvidenceMatcher(
        [
            PolygonProfile.create("way/2", "Fontvieille"),
            PolygonProfile.create("way/1", "Fontvieille"),
        ]
    )
    document = FineWebDocument(
        row_index=10,
        document_id="doc-10",
        text=("Fontvieille and Monaco. " * 20),
        url=(
            "https://example.test/Fontvieille/Principality%20of%20%4Donaco/"
            + ("x/" * 150)
        ),
    )

    matches = matcher.match(document)

    assert [match.polygon_id for match in matches] == ["way/1", "way/2"]
    assert all(match.matched_fields == ("text", "url") for match in matches)
    assert all(match.context_fields == ("text", "url") for match in matches)
    assert all(match.context_phrase == "principality of monaco" for match in matches)
    assert all(match.polygon_name == "Fontvieille" for match in matches)
    assert all(match.matched_name == "Fontvieille" for match in matches)
    assert all(match.fineweb_document_id == "doc-10" for match in matches)
    assert all(match.url == document.url for match in matches)
    assert all(len(match.text_excerpt) == 240 for match in matches)
    assert all(match.text_excerpt.endswith("…") for match in matches)
    assert all(match.url_excerpt.endswith("…") for match in matches)


def test_match_excerpts_are_empty_for_fields_without_evidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    text_only = matcher.match(
        FineWebDocument(11, None, "Fontvieille in Monaco.", "https://example.test")
    )[0]
    url_only = matcher.match(
        FineWebDocument(
            12,
            None,
            "A general report.",
            "https://example.test/Fontvieille/Monaco",
        )
    )[0]

    assert text_only.text_excerpt == "Fontvieille in Monaco."
    assert text_only.url_excerpt == ""
    assert url_only.text_excerpt == ""
    assert url_only.url_excerpt == url_only.url


def test_excerpt_keeps_the_exact_boundary_and_truncates_after_it() -> None:
    at_limit = "x" * 240
    above_limit = "x" * 241

    assert _excerpt(at_limit) == at_limit
    assert _excerpt(above_limit) == ("x" * 239) + "…"
