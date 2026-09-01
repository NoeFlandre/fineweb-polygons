"""Aho-Corasick matching for polygon names and OSM name aliases."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Any

import ahocorasick

from fineweb_polygons.direction2.models import PolygonNameMatch, PolygonRecord
from fineweb_polygons.normalization import normalize_for_search


@dataclass(frozen=True, slots=True)
class _NormalizedText:
    text: str
    offsets: tuple[int, ...]


class AhoCorasickPolygonMatcher:
    """Match every indexed polygon name without scanning once per polygon."""

    def __init__(self, automaton: Any, names_indexed: int) -> None:
        self._automaton = automaton
        self.names_indexed = names_indexed

    @classmethod
    def build(cls, polygons: tuple[PolygonRecord, ...]) -> AhoCorasickPolygonMatcher:
        """Build a matcher over unique normalized names and aliases."""
        patterns = _indexed_patterns(polygons)

        automaton = ahocorasick.Automaton()
        for normalized, values in sorted(patterns.items()):
            automaton.add_word(normalized, (normalized, tuple(values)))
        automaton.make_automaton()
        return cls(automaton, names_indexed=len(patterns))

    def find(self, text: str) -> tuple[PolygonNameMatch, ...]:
        """Return boundary-aware matches with original-text character offsets."""
        fast_normalized = normalize_for_search(text, decode_url=False)
        if not fast_normalized:
            return ()
        candidates = _candidate_spans(self._automaton, fast_normalized)
        if not candidates:
            return ()
        mapped = _normalize_with_offsets(text)
        if mapped.text != fast_normalized:
            candidates = _candidate_spans(self._automaton, mapped.text)
        return _materialize_matches(candidates, mapped.offsets)


def count_unique_normalized_names(polygons: tuple[PolygonRecord, ...]) -> int:
    """Count unique searchable names and aliases without building an automaton."""
    return len(_indexed_patterns(polygons))


def _indexed_patterns(
    polygons: tuple[PolygonRecord, ...],
) -> dict[str, list[tuple[PolygonRecord, str]]]:
    patterns: dict[str, list[tuple[PolygonRecord, str]]] = {}
    for polygon in polygons:
        for alias in polygon.candidate_names():
            normalized = normalize_for_search(alias, decode_url=False)
            if not normalized:
                continue
            values = patterns.setdefault(normalized, [])
            candidate = (polygon, alias)
            if candidate not in values:
                values.append(candidate)
    return patterns


def _normalize_with_offsets(value: str) -> _NormalizedText:
    characters: list[str] = []
    offsets: list[int] = []
    pending_separator: int | None = None
    for original_index, character in enumerate(value):
        pending_separator = _append_folded_character(
            characters,
            offsets,
            original_index,
            character,
            pending_separator,
        )
    return _NormalizedText("".join(characters), tuple(offsets))


def _append_folded_character(
    characters: list[str],
    offsets: list[int],
    original_index: int,
    character: str,
    pending_separator: int | None,
) -> int | None:
    folded = unicodedata.normalize("NFKC", character).casefold()
    for folded_character in folded:
        if not folded_character.isalnum():
            pending_separator = original_index
            continue
        if characters and pending_separator is not None:
            characters.append(" ")
            offsets.append(pending_separator)
        characters.append(folded_character)
        offsets.append(original_index)
        pending_separator = None
    return pending_separator


def _has_boundaries(text: str, start: int, end: int) -> bool:
    before_is_word = start > 0 and text[start - 1].isalnum()
    after_is_word = end + 1 < len(text) and text[end + 1].isalnum()
    return not before_is_word and not after_is_word


def _candidate_spans(
    automaton: Any,
    text: str,
) -> list[tuple[int, int, tuple[tuple[PolygonRecord, str], ...]]]:
    candidates = []
    for end, (pattern, values) in automaton.iter(text):
        start = end - len(pattern) + 1
        if _has_boundaries(text, start, end):
            candidates.append((start, end, values))
    return candidates


def _materialize_matches(
    candidates: list[tuple[int, int, tuple[tuple[PolygonRecord, str], ...]]],
    offsets: tuple[int, ...],
) -> tuple[PolygonNameMatch, ...]:
    matches = [
        PolygonNameMatch(
            polygon=polygon,
            matched_alias=alias,
            start=offsets[start],
            end=offsets[end] + 1,
        )
        for start, end, values in candidates
        for polygon, alias in values
    ]
    return tuple(
        sorted(
            matches,
            key=lambda match: (
                match.start,
                match.end,
                match.polygon.polygon_id,
                match.matched_alias,
            ),
        )
    )
