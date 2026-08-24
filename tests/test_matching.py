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
    assert matches[0].context_phrase == "monaco"
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


def test_context_helpers_decode_only_urls_and_choose_the_longest_context() -> None:
    values = {
        "text": "Principality%20of%20M%6fnaco",
        "url": "https://example.test/Principality%20of%20%4d%6f%6e%61%63%6f",
    }
    context_matcher = _MultiPatternMatcher(("monaco", "principality of monaco"))

    assert _context_candidates(values) == ("url",)
    contexts, phrase = _find_context(values, context_matcher)
    assert contexts == {"url": frozenset({"monaco", "principality of monaco"})}
    assert phrase == "principality of monaco"


def test_context_helper_does_not_decode_text_values() -> None:
    values = {"text": "Principality%20of Monaco", "url": ""}
    context_matcher = _MultiPatternMatcher(("monaco", "principality of monaco"))

    contexts, phrase = _find_context(values, context_matcher)

    assert contexts == {"text": frozenset({"monaco"})}
    assert phrase == "monaco"


def test_context_helper_prefers_longest_phrase_over_lexical_order() -> None:
    class FakeContextMatcher:
        def find(self, value: str, *, decode_url: bool = True) -> frozenset[str]:
            del value, decode_url
            return frozenset({"zzzz", "a much longer phrase"})

    contexts, phrase = _find_context(
        {"text": "monaco", "url": ""}, cast(_MultiPatternMatcher, FakeContextMatcher())
    )

    assert contexts == {"text": frozenset({"zzzz", "a much longer phrase"})}
    assert phrase == "a much longer phrase"


def test_name_helper_decodes_urls_but_not_text() -> None:
    values = {
        "text": "fontv%69eille",
        "url": "https://example.test/fontv%69eille",
    }

    assert _find_names(values, _MultiPatternMatcher(("fontvieille",))) == {
        "text": frozenset(),
        "url": frozenset({"fontvieille"}),
    }


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


def test_v2_text_match_requires_monaco_context_in_the_same_text() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
    )
    document = FineWebDocument(
        10,
        "doc-10",
        "Fontvieille has a report.",
        "https://monaco.example.test/report",
    )

    assert matcher.match(document) == ()


def test_v2_text_match_accepts_monaco_context_in_the_same_text() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
    )
    document = FineWebDocument(
        11,
        "doc-11",
        "Fontvieille is in Monaco.",
        "https://example.test/report",
    )

    evidence = matcher.match(document)[0]

    assert evidence.matched_fields == ("text",)
    assert evidence.url_excerpt == ""


def test_v2_url_match_does_not_require_country_context() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
    )
    document = FineWebDocument(
        12,
        "doc-12",
        "A report about an unrelated place.",
        "https://example.test/fontvieille",
    )

    assert matcher.match(document)[0].matched_fields == ("url",)


def test_multi_pattern_matcher_decodes_encoded_values_by_default() -> None:
    matcher = _MultiPatternMatcher(["café"])

    assert matcher.find("https://example.test/caf%C3%A9") == frozenset({"café"})


def test_multi_pattern_matcher_can_skip_url_decoding() -> None:
    matcher = _MultiPatternMatcher(["café"])

    assert (
        matcher.find("https://example.test/caf%C3%A9", decode_url=False) == frozenset()
    )


def test_matcher_orders_profiles_with_the_same_normalized_name() -> None:
    matcher = EvidenceMatcher(
        [
            PolygonProfile.create("way/2", "Palais"),
            PolygonProfile.create("way/1", "Palais"),
        ]
    )
    document = FineWebDocument(
        13,
        "doc-13",
        "Palais is in Monaco.",
        "",
    )

    assert [evidence.polygon_id for evidence in matcher.match(document)] == [
        "way/1",
        "way/2",
    ]


def test_v2_url_only_evidence_records_an_empty_context_phrase() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
    )
    document = FineWebDocument(
        14,
        "doc-14",
        "An unrelated page.",
        "https://example.test/fontvieille",
    )

    assert matcher.match(document)[0].context_phrase == ""


def test_v2_evidence_records_the_complete_text() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
    )
    full_text = "Fontvieille is in Monaco. " * 40
    document = FineWebDocument(
        15,
        "doc-15",
        full_text,
        "https://example.test/fontvieille",
    )

    evidence = matcher.match(document)[0]
    record = evidence.to_record()

    assert record["polygon_name"] == "Fontvieille"
    assert record["fineweb_document_id"] == "doc-15"
    assert record["url"] == "https://example.test/fontvieille"
    assert record["matched_name"] == "Fontvieille"
    assert record["text"] == full_text
    assert record["text_excerpt"] == f"{full_text[:239]}…"


def test_v2_url_only_evidence_keeps_text_excerpt_empty() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
    )
    document = FineWebDocument(
        16,
        "doc-16",
        "An unrelated page.",
        "https://example.test/fontvieille",
    )

    assert matcher.match(document)[0].to_record()["text_excerpt"] == ""


def test_excerpt_keeps_the_exact_boundary_length() -> None:
    value = "x" * 240

    assert _excerpt(value) == value
