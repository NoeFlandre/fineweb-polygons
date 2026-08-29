"""Fast exact evidence matching for the versioned retrieval rules."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import ahocorasick

from fineweb_polygons.models import FineWebDocument, MatchEvidence, PolygonProfile
from fineweb_polygons.normalization import has_context_marker, normalize_for_search

_FIELD_ORDER = ("text", "url")
_EXCERPT_LIMIT = 240
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


class _MultiPatternMatcher:
    def __init__(self, patterns: Iterable[str]) -> None:
        unique_patterns = tuple(sorted({pattern for pattern in patterns if pattern}))
        self._has_patterns = bool(unique_patterns)
        self._automaton = ahocorasick.Automaton()
        for pattern in unique_patterns:
            self._automaton.add_word(f" {pattern} ", pattern)
        self._automaton.make_automaton()

    def find(self, value: str, *, decode_url: bool = True) -> frozenset[str]:
        normalized = normalize_for_search(value, decode_url=decode_url)
        return self.find_normalized(normalized)

    def find_normalized(self, normalized: str) -> frozenset[str]:
        """Find patterns in a value that already uses search normalization."""
        if not self._has_patterns:
            return frozenset()
        if not normalized:
            return frozenset()
        padded = f" {normalized} "
        return frozenset(pattern for _, pattern in self._automaton.iter(padded))

    def find_spans(
        self, value: str, *, decode_url: bool = True
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        """Return normalized half-open spans for every matched pattern."""
        normalized = normalize_for_search(value, decode_url=decode_url)
        return self.find_spans_normalized(normalized)

    def find_spans_normalized(
        self, normalized: str
    ) -> dict[str, tuple[tuple[int, int], ...]]:
        """Return spans for a value that already uses search normalization."""
        if not self._has_patterns:
            return {}
        if not normalized:
            return {}
        padded = f" {normalized} "
        spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for end, pattern in self._automaton.iter(padded):
            start = end - len(pattern) - 1
            spans[pattern].append((start, end - 1))
        return {pattern: tuple(matches) for pattern, matches in spans.items()}


class EvidenceMatcher:
    """Match polygon names and one configured country context."""

    def __init__(
        self,
        profiles: Sequence[PolygonProfile],
        *,
        require_text_context: bool = False,
        require_text_name: bool = False,
        require_url_name: bool = False,
        context_name: str = "Monaco",
        max_name_country_distance: int | None = None,
    ) -> None:
        self._require_text_context = require_text_context
        self._require_text_name = require_text_name
        self._require_url_name = require_url_name
        _validate_distance_limit(max_name_country_distance)
        self._max_name_country_distance = max_name_country_distance
        self._context_name = normalize_for_search(context_name, decode_url=False)
        if not self._context_name:
            raise ValueError("context_name must not be empty")
        profiles_by_name: dict[str, list[PolygonProfile]] = defaultdict(list)
        for profile in profiles:
            if profile.normalized_name:
                profiles_by_name[profile.normalized_name].append(profile)
        self._profiles_by_name = {
            name: tuple(sorted(items, key=lambda item: item.polygon_id))
            for name, items in profiles_by_name.items()
        }
        self._name_matcher = _MultiPatternMatcher(self._profiles_by_name)
        self._context_matcher = _MultiPatternMatcher((self._context_name,))

    def match(self, document: FineWebDocument) -> tuple[MatchEvidence, ...]:
        values = {"text": document.text, "url": document.url}
        if self._max_name_country_distance is not None:
            return self._match_v6(document, values)
        if self._require_url_name:
            return self._match_v3(document, values)
        if self._require_text_name:
            return self._match_v4(document, values)
        if self._require_text_context:
            return self._match_v2(document, values)
        return self._match_v1(document, values)

    def _match_v3(
        self,
        document: FineWebDocument,
        values: Mapping[str, str],
    ) -> tuple[MatchEvidence, ...]:
        contexts_by_field, context_phrase = _find_context(
            values, self._context_matcher, context_name=self._context_name
        )
        if context_phrase is None or not contexts_by_field.get("text"):
            return ()
        names_by_field = _find_names(values, self._name_matcher)
        accepted_names = _v3_accepted_names(names_by_field, contexts_by_field)
        if not accepted_names:
            return ()
        return _evidence_for_names(
            document,
            matched_names=accepted_names,
            profiles_by_name=self._profiles_by_name,
            names_by_field=names_by_field,
            contexts_by_field=contexts_by_field,
            context_phrase=context_phrase,
        )

    def _match_v1(
        self,
        document: FineWebDocument,
        values: Mapping[str, str],
    ) -> tuple[MatchEvidence, ...]:
        contexts_by_field, context_phrase = _find_context(
            values, self._context_matcher, context_name=self._context_name
        )
        if context_phrase is None:
            return ()
        names_by_field = _find_names(values, self._name_matcher)
        matched_names = set().union(*names_by_field.values())
        if not matched_names:
            return ()
        return _evidence_for_names(
            document,
            matched_names=matched_names,
            profiles_by_name=self._profiles_by_name,
            names_by_field=names_by_field,
            contexts_by_field=contexts_by_field,
            context_phrase=context_phrase,
        )

    def _match_v2(
        self,
        document: FineWebDocument,
        values: Mapping[str, str],
    ) -> tuple[MatchEvidence, ...]:
        names_by_field = _find_names(values, self._name_matcher)
        contexts_by_field, context_phrase = _find_context(
            values, self._context_matcher, context_name=self._context_name
        )
        accepted_names = _v2_accepted_names(names_by_field, contexts_by_field)
        if not accepted_names:
            return ()
        return _evidence_for_names(
            document,
            matched_names=accepted_names,
            profiles_by_name=self._profiles_by_name,
            names_by_field=names_by_field,
            contexts_by_field=contexts_by_field,
            context_phrase=context_phrase or "",
        )

    def _match_v4(
        self,
        document: FineWebDocument,
        values: Mapping[str, str],
    ) -> tuple[MatchEvidence, ...]:
        contexts_by_field, _ = _find_context(
            values, self._context_matcher, context_name=self._context_name
        )
        if "text" not in contexts_by_field:
            return ()
        text_contexts = contexts_by_field["text"]
        names_by_field = _find_names(values, self._name_matcher)
        accepted_names = set(names_by_field["text"])
        if not accepted_names:
            return ()
        context_phrase = max(text_contexts, key=lambda phrase: (len(phrase), phrase))
        return _evidence_for_names(
            document,
            matched_names=accepted_names,
            profiles_by_name=self._profiles_by_name,
            names_by_field=names_by_field,
            contexts_by_field=contexts_by_field,
            context_phrase=context_phrase,
        )

    def _match_v6(
        self,
        document: FineWebDocument,
        values: Mapping[str, str],
    ) -> tuple[MatchEvidence, ...]:
        max_distance = self._max_name_country_distance
        if max_distance is None:
            return ()
        text_name_spans, text_context_spans = _v6_text_spans(
            document.text, self._name_matcher, self._context_matcher
        )
        if not text_name_spans:
            return ()
        if not text_context_spans:
            return ()
        accepted_pairs = _accepted_v6_pairs(
            text_name_spans, text_context_spans, max_distance
        )
        if not accepted_pairs:
            return ()
        names_by_field, contexts_by_field = _v6_match_fields(
            values,
            name_matcher=self._name_matcher,
            context_matcher=self._context_matcher,
            text_name_spans=text_name_spans,
            text_context_spans=text_context_spans,
        )
        return _v6_evidence(
            document,
            accepted_pairs=accepted_pairs,
            profiles_by_name=self._profiles_by_name,
            names_by_field=names_by_field,
            contexts_by_field=contexts_by_field,
            context_phrase=self._context_name,
        )


def _v2_accepted_names(
    names_by_field: Mapping[str, frozenset[str]],
    contexts_by_field: Mapping[str, frozenset[str]],
) -> set[str]:
    accepted_names = set(names_by_field["url"])
    if contexts_by_field.get("text"):
        accepted_names.update(names_by_field["text"])
    return accepted_names


def _v3_accepted_names(
    names_by_field: Mapping[str, frozenset[str]],
    contexts_by_field: Mapping[str, frozenset[str]],
) -> set[str]:
    if not contexts_by_field.get("text"):
        return set()
    return set(names_by_field["url"]) & set(names_by_field["text"])


def _validate_distance_limit(max_distance: int | None) -> None:
    if max_distance is not None and max_distance < 0:
        raise ValueError("max_name_country_distance must be non-negative")


def _v6_text_spans(
    text: str,
    name_matcher: _MultiPatternMatcher,
    context_matcher: _MultiPatternMatcher,
) -> tuple[
    dict[str, tuple[tuple[int, int], ...]],
    dict[str, tuple[tuple[int, int], ...]],
]:
    # pragma: no mutate start
    normalized = normalize_for_search(text, decode_url=False)
    # pragma: no mutate end
    text_context_spans = context_matcher.find_spans_normalized(normalized)
    if not text_context_spans:
        return {}, {}
    return name_matcher.find_spans_normalized(normalized), text_context_spans


def _accepted_v6_pairs(
    name_spans: Mapping[str, Sequence[tuple[int, int]]],
    context_spans: Mapping[str, Sequence[tuple[int, int]]],
    max_distance: int,
) -> dict[str, tuple[int, tuple[tuple[int, int], tuple[int, int]]]]:
    country_spans = tuple(
        span for pattern_spans in context_spans.values() for span in pattern_spans
    )
    accepted: dict[str, tuple[int, tuple[tuple[int, int], tuple[int, int]]]] = {}
    for name, spans in name_spans.items():
        accepted_pair = _accepted_v6_pair(spans, country_spans, max_distance)
        if accepted_pair is not None:
            accepted[name] = accepted_pair
    return accepted


def _accepted_v6_pair(
    name_spans: Sequence[tuple[int, int]],
    country_spans: Iterable[tuple[int, int]],
    max_distance: int,
) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]] | None:
    closest = _closest_span_pair(name_spans, country_spans)
    if closest is None or closest[0] > max_distance:
        return None
    return closest


def _v6_match_fields(
    values: Mapping[str, str],
    *,
    name_matcher: _MultiPatternMatcher,
    context_matcher: _MultiPatternMatcher,
    text_name_spans: Mapping[str, Sequence[tuple[int, int]]],
    text_context_spans: Mapping[str, Sequence[tuple[int, int]]],
) -> tuple[dict[str, frozenset[str]], dict[str, frozenset[str]]]:
    # Keep URL decoding explicit: V6 text spans select the row, while URL
    # matches are retained only as secondary metadata.
    normalized_url = normalize_for_search(values["url"])
    # pragma: no mutate start
    url_names = name_matcher.find_normalized(normalized_url)
    # pragma: no mutate end
    names_by_field = {
        "text": frozenset(text_name_spans),
        "url": url_names,
    }
    contexts_by_field = {"text": frozenset(text_context_spans)}
    # pragma: no mutate start
    url_contexts = context_matcher.find_normalized(normalized_url)
    # pragma: no mutate end
    if url_contexts:
        contexts_by_field["url"] = url_contexts
    return names_by_field, contexts_by_field


def _v6_evidence(
    document: FineWebDocument,
    *,
    accepted_pairs: Mapping[str, tuple[int, tuple[tuple[int, int], tuple[int, int]]]],
    profiles_by_name: Mapping[str, Sequence[PolygonProfile]],
    names_by_field: Mapping[str, frozenset[str]],
    contexts_by_field: Mapping[str, frozenset[str]],
    context_phrase: str,
) -> tuple[MatchEvidence, ...]:
    sentence_ranges = _normalized_sentence_ranges(document.text)
    distances = {name: pair[0] for name, pair in accepted_pairs.items()}
    sentences = {
        name: _sentences_for_pair(sentence_ranges, pair)
        for name, pair in accepted_pairs.items()
    }
    return _evidence_for_names(
        document,
        matched_names=set(accepted_pairs),
        profiles_by_name=profiles_by_name,
        names_by_field=names_by_field,
        contexts_by_field=contexts_by_field,
        context_phrase=context_phrase,
        name_country_distances=distances,
        name_country_sentences=sentences,
    )


def _evidence_for_names(
    document: FineWebDocument,
    *,
    matched_names: set[str],
    profiles_by_name: Mapping[str, Sequence[PolygonProfile]],
    names_by_field: Mapping[str, frozenset[str]],
    contexts_by_field: Mapping[str, frozenset[str]],
    context_phrase: str,
    name_country_distances: Mapping[str, int] | None = None,
    name_country_sentences: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[MatchEvidence, ...]:
    return tuple(
        evidence
        for normalized_name in sorted(matched_names)
        for evidence in _matches_for_name(
            document,
            normalized_name=normalized_name,
            profiles=profiles_by_name[normalized_name],
            names_by_field=names_by_field,
            contexts_by_field=contexts_by_field,
            context_phrase=context_phrase,
            name_country_distance=(
                None
                if name_country_distances is None
                else name_country_distances.get(normalized_name)
            ),
            name_country_sentences=(
                None
                if name_country_sentences is None
                else name_country_sentences.get(normalized_name)
            ),
        )
    )


def _find_context(
    values: Mapping[str, str],
    matcher: _MultiPatternMatcher,
    *,
    context_name: str = "Monaco",
) -> tuple[dict[str, frozenset[str]], str | None]:
    candidates = _context_candidates(values, context_name=context_name)
    if not candidates:
        return {}, None
    contexts_by_field = {
        field: matcher.find(values[field], decode_url=field == "url")
        for field in candidates
    }
    phrases = set().union(*contexts_by_field.values())
    if not phrases:
        return contexts_by_field, None
    return contexts_by_field, max(phrases, key=lambda phrase: (len(phrase), phrase))


def _context_candidates(
    values: Mapping[str, str], *, context_name: str = "Monaco"
) -> tuple[str, ...]:
    return tuple(
        field
        for field in _FIELD_ORDER
        if has_context_marker(values[field], context_name, decode_url=field == "url")
    )


def _find_names(
    values: Mapping[str, str], matcher: _MultiPatternMatcher
) -> dict[str, frozenset[str]]:
    return {
        field: matcher.find(values[field], decode_url=field == "url")
        for field in _FIELD_ORDER
    }


def _matches_for_name(
    document: FineWebDocument,
    *,
    normalized_name: str,
    profiles: Sequence[PolygonProfile],
    names_by_field: Mapping[str, frozenset[str]],
    contexts_by_field: Mapping[str, frozenset[str]],
    context_phrase: str,
    name_country_distance: int | None = None,
    name_country_sentences: tuple[str, str] | None = None,
) -> tuple[MatchEvidence, ...]:
    matched_fields = _fields_containing(normalized_name, names_by_field)
    context_fields = _fields_with_context(contexts_by_field)
    evidence_fields = set(matched_fields) | set(context_fields)
    return tuple(
        _make_evidence(
            document,
            profile=profile,
            matched_fields=matched_fields,
            context_fields=context_fields,
            context_phrase=context_phrase,
            evidence_fields=evidence_fields,
            name_country_distance=name_country_distance,
            name_country_sentences=name_country_sentences,
        )
        for profile in profiles
    )


def _fields_containing(
    value: str, values_by_field: Mapping[str, frozenset[str]]
) -> tuple[str, ...]:
    return tuple(field for field in _FIELD_ORDER if value in values_by_field[field])


def _fields_with_context(
    values_by_field: Mapping[str, frozenset[str]],
) -> tuple[str, ...]:
    return tuple(field for field in _FIELD_ORDER if values_by_field.get(field))


def _make_evidence(
    document: FineWebDocument,
    *,
    profile: PolygonProfile,
    matched_fields: tuple[str, ...],
    context_fields: tuple[str, ...],
    context_phrase: str,
    evidence_fields: set[str],
    name_country_distance: int | None = None,
    name_country_sentences: tuple[str, str] | None = None,
) -> MatchEvidence:
    return MatchEvidence(
        polygon_id=profile.polygon_id,
        polygon_name=profile.name,
        fineweb_row_index=document.row_index,
        fineweb_document_id=document.document_id,
        url=document.url,
        matched_fields=matched_fields,
        context_fields=context_fields,
        matched_name=profile.name,
        context_phrase=context_phrase,
        text=document.text,
        text_excerpt=_excerpt(document.text if "text" in evidence_fields else ""),
        url_excerpt=_excerpt(document.url if "url" in evidence_fields else ""),
        name_country_distance=name_country_distance,
        polygon_name_sentence=(
            None if name_country_sentences is None else name_country_sentences[0]
        ),
        country_name_sentence=(
            None if name_country_sentences is None else name_country_sentences[1]
        ),
    )


def _closest_span_pair(
    name_spans: Sequence[tuple[int, int]],
    context_spans: Iterable[tuple[int, int]],
) -> tuple[int, tuple[tuple[int, int], tuple[int, int]]] | None:
    best: tuple[int, tuple[tuple[int, int], tuple[int, int]]] | None = None
    for name_span in name_spans:
        for context_span in context_spans:
            candidate = (
                _span_distance(name_span, context_span),
                (name_span, context_span),
            )
            if best is None or candidate < best:  # pragma: no mutate
                best = candidate
    return best


def _span_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, right[0] - left[1], left[0] - right[1])


def _normalized_sentence_ranges(
    text: str,
) -> tuple[tuple[int, int, str], ...]:
    ranges: list[tuple[int, int, str]] = []
    cursor = 0
    for sentence in _SENTENCE_SPLIT_RE.split(text):
        original = sentence.strip()
        # pragma: no mutate start
        normalized = normalize_for_search(original, decode_url=False)
        # pragma: no mutate end
        if not normalized:
            continue
        start = cursor
        end = start + len(normalized)
        ranges.append((start, end, original))
        cursor = end + 1
    return tuple(ranges)


def _sentences_for_pair(
    ranges: Sequence[tuple[int, int, str]],
    pair: tuple[int, tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[str, str]:
    name_span, country_span = pair[1]
    return (
        _sentence_for_span(ranges, name_span),
        _sentence_for_span(ranges, country_span),
    )


def _sentence_for_span(
    ranges: Sequence[tuple[int, int, str]], span: tuple[int, int]
) -> str:
    for start, end, sentence in ranges:
        if start <= span[0] < end:
            return sentence
    return ""


def _excerpt(value: str) -> str:
    if len(value) <= _EXCERPT_LIMIT:
        return value
    return f"{value[: _EXCERPT_LIMIT - 1]}…"
