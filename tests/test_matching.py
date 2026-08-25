import inspect
from typing import cast

import pytest

from fineweb_polygons.matching import (
    EvidenceMatcher,
    _closest_span_pair,
    _context_candidates,
    _excerpt,
    _find_context,
    _find_names,
    _MultiPatternMatcher,
    _normalized_sentence_ranges,
    _sentence_for_span,
    _span_distance,
    _v6_text_spans,
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


def test_span_helper_decodes_urls_only_when_requested() -> None:
    matcher = _MultiPatternMatcher(("monaco",))

    assert matcher.find_spans("mon%61co", decode_url=False) == {}
    assert matcher.find_spans("mon%61co", decode_url=True) == {"monaco": ((0, 6),)}
    assert matcher.find_spans("mon%61co") == {"monaco": ((0, 6),)}


def test_v6_text_span_helper_passes_false_url_decoding_to_both_matchers() -> None:
    class FakeMatcher:
        def __init__(self) -> None:
            self.decode_flags: list[bool] = []

        def find_spans(
            self, value: str, *, decode_url: bool = True
        ) -> dict[str, tuple[tuple[int, int], ...]]:
            del value
            self.decode_flags.append(decode_url)
            return {}

    name_matcher = FakeMatcher()
    context_matcher = FakeMatcher()

    assert _v6_text_spans(
        "text",
        cast(_MultiPatternMatcher, name_matcher),
        cast(_MultiPatternMatcher, context_matcher),
    ) == ({}, {})
    assert name_matcher.decode_flags == [False]
    assert context_matcher.decode_flags == [False]


def test_v6_keeps_url_matches_as_metadata_only() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=500,
    )

    evidence = matcher.match(
        FineWebDocument(
            26,
            "doc-26",
            "Fontvieille is in Monaco.",
            "https://example.test/fontv%69eille/Mon%61co",
        )
    )[0]

    assert evidence.matched_fields == ("text", "url")
    assert evidence.context_fields == ("text", "url")


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


def test_multi_pattern_matcher_accepts_no_patterns() -> None:
    assert _MultiPatternMatcher([]).find("anything") == frozenset()


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


def test_v3_requires_name_in_url_and_name_with_context_in_text() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_url_name=True,
    )
    document = FineWebDocument(
        17,
        "doc-17",
        "Fontvieille is in Monaco.",
        "https://example.test/fontvieille",
    )

    evidence = matcher.match(document)[0]

    assert evidence.matched_fields == ("text", "url")
    assert evidence.context_fields == ("text",)
    assert evidence.context_phrase == "monaco"


def test_v3_rejects_a_name_only_in_the_url() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_url_name=True,
    )
    document = FineWebDocument(
        18,
        "doc-18",
        "An unrelated page about Monaco.",
        "https://example.test/fontvieille",
    )

    assert matcher.match(document) == ()


def test_v3_rejects_a_name_only_in_text_even_when_url_has_monaco() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_url_name=True,
    )
    document = FineWebDocument(
        19,
        "doc-19",
        "Fontvieille is a venue in Monaco.",
        "https://monaco.example.test/report",
    )

    assert matcher.match(document) == ()


def test_v3_rejects_url_context_when_text_lacks_monaco_context() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_url_name=True,
    )
    document = FineWebDocument(
        20,
        "doc-20",
        "Fontvieille is a venue.",
        "https://monaco.example.test/fontvieille",
    )

    assert matcher.match(document) == ()


def test_v4_accepts_polygon_name_and_monaco_context_in_text_only() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
    )
    document = FineWebDocument(
        21,
        "doc-21",
        "Fontvieille is a district in Monaco.",
        "https://example.test/unrelated-article",
    )

    evidence = matcher.match(document)[0]

    assert evidence.matched_fields == ("text",)
    assert evidence.context_fields == ("text",)
    assert evidence.context_phrase == "monaco"
    assert evidence.url_excerpt == ""


def test_v4_rejects_polygon_name_only_in_url() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
    )
    document = FineWebDocument(
        22,
        "doc-22",
        "An unrelated article about Monaco.",
        "https://example.test/fontvieille",
    )

    assert matcher.match(document) == ()


def test_v4_rejects_polygon_name_without_text_context() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
    )
    document = FineWebDocument(
        23,
        "doc-23",
        "Fontvieille is a district.",
        "https://example.test/unrelated-article",
    )

    assert matcher.match(document) == ()


def test_v4_ignores_a_monaco_substring_without_raising() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Building")],
        require_text_context=True,
        require_text_name=True,
    )
    document = FineWebDocument(
        24,
        "doc-24",
        "Building was mentioned in monacobuilding.",
        "",
    )

    assert matcher.match(document) == ()


def test_v4_chooses_the_longest_text_context_phrase(monkeypatch) -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
    )

    class FakeContextMatcher:
        def find(self, value: str, *, decode_url: bool = True) -> frozenset[str]:
            del value, decode_url
            return frozenset({"z", "long phrase"})

    monkeypatch.setattr(
        matcher,
        "_context_matcher",
        cast(_MultiPatternMatcher, FakeContextMatcher()),
    )
    document = FineWebDocument(
        25,
        "doc-25",
        "Fontvieille Monaco.",
        "https://example.test/unrelated-article",
    )

    assert matcher.match(document)[0].context_phrase == "long phrase"


def test_v5_uses_the_configured_country_name_as_text_context() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Vaduz")],
        require_text_context=True,
        require_text_name=True,
        context_name="Liechtenstein",
    )

    evidence = matcher.match(
        FineWebDocument(
            40,
            "doc-40",
            "Vaduz is the capital of Liechtenstein.",
            "https://example.test/unrelated",
        )
    )[0]

    assert evidence.context_phrase == "liechtenstein"
    assert evidence.matched_fields == ("text",)
    assert evidence.context_fields == ("text",)


def test_v5_does_not_accept_monaco_as_liechtenstein_context() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Vaduz")],
        require_text_context=True,
        require_text_name=True,
        context_name="Liechtenstein",
    )

    assert matcher.match(FineWebDocument(41, "doc-41", "Vaduz is in Monaco.", "")) == ()


def test_v6_accepts_an_inclusive_500_character_normalized_text_gap() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=500,
    )
    document = FineWebDocument(
        43,
        "doc-43",
        f"Fontvieille {'x' * 498} Monaco",
        "https://example.test/unrelated",
    )

    evidence = matcher.match(document)[0]

    assert evidence.matched_fields == ("text",)
    assert evidence.context_fields == ("text",)
    assert evidence.name_country_distance == 500
    assert evidence.to_record()["name_country_distance"] == 500


def test_v6_rejects_a_501_character_normalized_text_gap() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=500,
    )
    document = FineWebDocument(
        44,
        "doc-44",
        f"Fontvieille {'x' * 499} Monaco",
        "https://example.test/fontvieille",
    )

    assert matcher.match(document) == ()


def test_v6_distance_uses_casefolded_separator_normalized_text() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Font Vieille")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=500,
    )

    evidence = matcher.match(
        FineWebDocument(45, "doc-45", "FONT-VIEILLE is in MONACO.", "")
    )[0]

    assert evidence.name_country_distance == 7


def test_v6_accepts_a_zero_distance_limit() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Monaco Telecom")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=0,
    )

    assert matcher.match(FineWebDocument(45, "doc-45", "Monaco Telecom", ""))


def test_v6_overlapping_name_and_country_spans_have_zero_distance() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Monaco Telecom")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=500,
    )

    evidence = matcher.match(FineWebDocument(46, "doc-46", "Monaco Telecom", ""))[0]

    assert evidence.name_country_distance == 0


def test_v6_serializes_the_original_sentences_without_excerpt_fields() -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Fontvieille")],
        require_text_context=True,
        require_text_name=True,
        max_name_country_distance=500,
    )
    text = "Fontvieille is a district. Monaco is the country context."

    record = matcher.match(FineWebDocument(47, "doc-47", text, ""))[0].to_record()

    assert record["text"] == text
    assert record["polygon_name_sentence"] == "Fontvieille is a district."
    assert record["country_name_sentence"] == "Monaco is the country context."
    assert record["context_phrase"] == "monaco"
    assert "text_excerpt" not in record
    assert "url_excerpt" not in record


def test_v6_rejects_a_negative_distance_limit() -> None:
    with pytest.raises(
        ValueError, match=r"\Amax_name_country_distance must be non-negative\Z"
    ):
        EvidenceMatcher([], max_name_country_distance=-1)


def test_span_distance_handles_both_directions_and_overlap() -> None:
    assert _span_distance((0, 10), (5, 12)) == 0
    assert _span_distance((5, 7), (1, 3)) == 2
    assert _span_distance((1, 3), (5, 7)) == 2


def test_closest_span_pair_prefers_the_smallest_distance() -> None:
    assert _closest_span_pair(((0, 3), (20, 23)), ((8, 10), (30, 32))) == (
        5,
        ((0, 3), (8, 10)),
    )


def test_closest_span_pair_uses_lexical_order_for_equal_distances() -> None:
    assert _closest_span_pair(((10, 12), (0, 2)), ((5, 7),)) == (
        3,
        ((0, 2), (5, 7)),
    )


def test_normalized_sentence_ranges_keep_exact_normalized_offsets() -> None:
    assert _normalized_sentence_ranges("One. Monaco here.") == (
        (0, 3, "One."),
        (4, 15, "Monaco here."),
    )


def test_normalized_sentence_ranges_skip_empty_sentences_and_continue() -> None:
    assert _normalized_sentence_ranges("...\nMonaco.") == ((0, 6, "Monaco."),)


def test_sentence_for_span_uses_the_start_of_a_half_open_span() -> None:
    ranges = ((0, 4, "first"), (4, 8, "second"))

    assert _sentence_for_span(ranges, (3, 6)) == "first"
    assert _sentence_for_span(ranges, (4, 5)) == "second"
    assert _sentence_for_span(ranges, (9, 10)) == ""


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"require_text_context": True},
        {"require_text_context": True, "require_url_name": True},
    ],
)
def test_configured_context_reaches_every_matching_version(options) -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Vaduz")],
        context_name="Liechtenstein",
        **options,
    )
    document = FineWebDocument(
        42,
        "doc-42",
        "Vaduz is in Liechtenstein.",
        "https://example.test/vaduz",
    )

    assert len(matcher.match(document)) == 1


def test_matcher_rejects_an_empty_context_name_with_a_stable_message() -> None:
    with pytest.raises(ValueError, match=r"\Acontext_name must not be empty\Z"):
        EvidenceMatcher([], context_name=" ")


def test_matcher_normalizes_the_default_context_name_without_url_decoding(
    monkeypatch,
) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_normalize(value: object, *, decode_url: bool = True) -> str:
        calls.append((value, decode_url))
        return "monaco"

    monkeypatch.setattr(
        "fineweb_polygons.matching.normalize_for_search", fake_normalize
    )

    EvidenceMatcher([])

    assert calls == [("Monaco", False)]


def test_v2_passes_the_configured_context_to_its_context_helper(monkeypatch) -> None:
    matcher = EvidenceMatcher(
        [PolygonProfile.create("way/1", "Vaduz")],
        require_text_context=True,
        context_name="Liechtenstein",
    )
    captured: dict[str, object] = {}
    real_find_context = _find_context

    def fake_find_context(values, context_matcher, *, context_name="Monaco"):
        captured["context_name"] = context_name
        return real_find_context(values, context_matcher, context_name=context_name)

    monkeypatch.setattr("fineweb_polygons.matching._find_context", fake_find_context)

    assert matcher.match(
        FineWebDocument(43, "doc-43", "Vaduz is in Liechtenstein.", "")
    )
    assert captured["context_name"] == "liechtenstein"


def test_context_helpers_pass_their_canonical_default_to_nested_calls(
    monkeypatch,
) -> None:
    context_calls: list[object] = []

    def fake_candidates(values, *, context_name="Monaco"):
        del values
        context_calls.append(context_name)
        return ()

    monkeypatch.setattr(
        "fineweb_polygons.matching._context_candidates", fake_candidates
    )
    _find_context({"text": "", "url": ""}, _MultiPatternMatcher(()))
    assert context_calls == ["Monaco"]

    marker_calls: list[object] = []

    def fake_marker(value, context_name, *, decode_url):
        del value, decode_url
        marker_calls.append(context_name)
        return False

    monkeypatch.setattr("fineweb_polygons.matching.has_context_marker", fake_marker)
    _context_candidates({"text": "", "url": ""})
    assert marker_calls == ["Monaco", "Monaco"]


def test_context_defaults_are_stable_public_defaults() -> None:
    assert inspect.signature(EvidenceMatcher).parameters["context_name"].default == (
        "Monaco"
    )
    assert inspect.signature(_find_context).parameters["context_name"].default == (
        "Monaco"
    )
    assert inspect.signature(_context_candidates).parameters[
        "context_name"
    ].default == ("Monaco")


def test_excerpt_keeps_the_exact_boundary_length() -> None:
    value = "x" * 240

    assert _excerpt(value) == value
