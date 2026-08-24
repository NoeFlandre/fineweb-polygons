"""Fast exact evidence matching for the V1 retrieval rule."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import ahocorasick

from fineweb_polygons.models import FineWebDocument, MatchEvidence, PolygonProfile
from fineweb_polygons.normalization import has_monaco_marker, normalize_for_search

_FIELD_ORDER = ("text", "url")
_CONTEXT_PHRASES = ("monaco", "principality of monaco")
_EXCERPT_LIMIT = 240


class _MultiPatternMatcher:
    def __init__(self, patterns: Iterable[str]) -> None:
        unique_patterns = tuple(sorted({pattern for pattern in patterns if pattern}))
        self._automaton = ahocorasick.Automaton()
        for pattern in unique_patterns:
            self._automaton.add_word(f" {pattern} ", pattern)
        self._automaton.make_automaton()

    def find(self, value: str, *, decode_url: bool = True) -> frozenset[str]:
        normalized = normalize_for_search(value, decode_url=decode_url)
        if not normalized:
            return frozenset()
        padded = f" {normalized} "
        return frozenset(pattern for _, pattern in self._automaton.iter(padded))


class EvidenceMatcher:
    """Match polygon names and Monaco context in either FineWeb field."""

    def __init__(
        self,
        profiles: Sequence[PolygonProfile],
        *,
        require_text_context: bool = False,
        require_url_name: bool = False,
    ) -> None:
        self._require_text_context = require_text_context
        self._require_url_name = require_url_name
        profiles_by_name: dict[str, list[PolygonProfile]] = defaultdict(list)
        for profile in profiles:
            if profile.normalized_name:
                profiles_by_name[profile.normalized_name].append(profile)
        self._profiles_by_name = {
            name: tuple(sorted(items, key=lambda item: item.polygon_id))
            for name, items in profiles_by_name.items()
        }
        self._name_matcher = _MultiPatternMatcher(self._profiles_by_name)
        self._context_matcher = _MultiPatternMatcher(_CONTEXT_PHRASES)

    def match(self, document: FineWebDocument) -> tuple[MatchEvidence, ...]:
        values = {"text": document.text, "url": document.url}
        if self._require_url_name:
            return self._match_v3(document, values)
        if self._require_text_context:
            return self._match_v2(document, values)
        return self._match_v1(document, values)

    def _match_v3(
        self,
        document: FineWebDocument,
        values: Mapping[str, str],
    ) -> tuple[MatchEvidence, ...]:
        names_by_field = _find_names(values, self._name_matcher)
        contexts_by_field, context_phrase = _find_context(values, self._context_matcher)
        accepted_names = _v3_accepted_names(names_by_field, contexts_by_field)
        if context_phrase is None:
            return ()
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
        contexts_by_field, context_phrase = _find_context(values, self._context_matcher)
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
        contexts_by_field, context_phrase = _find_context(values, self._context_matcher)
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


def _evidence_for_names(
    document: FineWebDocument,
    *,
    matched_names: set[str],
    profiles_by_name: Mapping[str, Sequence[PolygonProfile]],
    names_by_field: Mapping[str, frozenset[str]],
    contexts_by_field: Mapping[str, frozenset[str]],
    context_phrase: str,
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
        )
    )


def _find_context(
    values: Mapping[str, str], matcher: _MultiPatternMatcher
) -> tuple[dict[str, frozenset[str]], str | None]:
    candidates = _context_candidates(values)
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


def _context_candidates(values: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        field
        for field in _FIELD_ORDER
        if has_monaco_marker(values[field], decode_url=field == "url")
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
    )


def _excerpt(value: str) -> str:
    if len(value) <= _EXCERPT_LIMIT:
        return value
    return f"{value[: _EXCERPT_LIMIT - 1]}…"
