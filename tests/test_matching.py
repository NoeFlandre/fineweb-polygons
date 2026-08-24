from fineweb_polygons.matching import EvidenceMatcher
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
