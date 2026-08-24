"""Fast exact evidence matching for the V1 retrieval rule."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import ahocorasick

from fineweb_polygons.models import FineWebDocument, MatchEvidence, PolygonProfile
from fineweb_polygons.normalization import normalize_for_search

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

    def find(self, value: str) -> frozenset[str]:
        normalized = normalize_for_search(value)
        if not normalized:
            return frozenset()
        padded = f" {normalized} "
        return frozenset(pattern for _, pattern in self._automaton.iter(padded))


class EvidenceMatcher:
    """Match polygon names and Monaco context in either FineWeb field."""

    def __init__(self, profiles: Sequence[PolygonProfile]) -> None:
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
        names_by_field = {
            field: self._name_matcher.find(values[field]) for field in _FIELD_ORDER
        }
        contexts_by_field = {
            field: self._context_matcher.find(values[field]) for field in _FIELD_ORDER
        }
        matched_names = set().union(*names_by_field.values())
        if not matched_names:
            return ()
        context_phrases = set().union(*contexts_by_field.values())
        if not context_phrases:
            return ()
        context_phrase = max(context_phrases, key=lambda phrase: (len(phrase), phrase))
        results: list[MatchEvidence] = []
        for normalized_name in sorted(matched_names):
            matched_fields = tuple(
                field
                for field in _FIELD_ORDER
                if normalized_name in names_by_field[field]
            )
            context_fields = tuple(
                field for field in _FIELD_ORDER if contexts_by_field[field]
            )
            evidence_fields = set(matched_fields) | set(context_fields)
            for profile in self._profiles_by_name[normalized_name]:
                results.append(
                    MatchEvidence(
                        polygon_id=profile.polygon_id,
                        polygon_name=profile.name,
                        fineweb_row_index=document.row_index,
                        fineweb_document_id=document.document_id,
                        url=document.url,
                        matched_fields=matched_fields,
                        context_fields=context_fields,
                        matched_name=profile.name,
                        context_phrase=context_phrase,
                        text_excerpt=_excerpt(
                            document.text if "text" in evidence_fields else ""
                        ),
                        url_excerpt=_excerpt(
                            document.url if "url" in evidence_fields else ""
                        ),
                    )
                )
        return tuple(results)


def _excerpt(value: str) -> str:
    if len(value) <= _EXCERPT_LIMIT:
        return value
    return f"{value[: _EXCERPT_LIMIT - 1]}…"
