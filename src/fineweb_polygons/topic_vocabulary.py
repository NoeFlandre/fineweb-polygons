"""Validated strong-term vocabulary matching for V8."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fineweb_polygons.matching import _MultiPatternMatcher
from fineweb_polygons.normalization import normalize_for_search


@dataclass(frozen=True, slots=True)
class VocabularyMatch:
    """One unique strong term matched in a document's text."""

    category: str
    term: str


@dataclass(frozen=True, slots=True)
class TopicVocabulary:
    """A validated, reusable vocabulary and its exact matcher."""

    version: str
    language: str
    matching: dict[str, object]
    categories: dict[str, tuple[str, ...]]
    _matcher: _MultiPatternMatcher = field(init=False, repr=False, compare=False)
    _normalized_terms: dict[str, str] = field(init=False, repr=False, compare=False)
    _term_categories: dict[str, str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        normalized_terms: dict[str, str] = {}
        term_categories: dict[str, str] = {}
        for category, terms in self.categories.items():
            for term in terms:
                normalized = normalize_for_search(term, decode_url=False)
                if normalized in normalized_terms:
                    raise ValueError(f"duplicate vocabulary term: {term}")
                normalized_terms[normalized] = term
                term_categories[normalized] = category
        object.__setattr__(self, "_normalized_terms", normalized_terms)
        object.__setattr__(self, "_term_categories", term_categories)
        object.__setattr__(self, "_matcher", _MultiPatternMatcher(normalized_terms))

    @classmethod
    def from_mapping(cls, value: object) -> TopicVocabulary:
        """Build a vocabulary from the approved JSON object contract."""
        mapping = _vocabulary_mapping(value)
        version = _required_string(mapping, "version")
        language = _required_string(mapping, "language")
        matching = _matching_mapping(mapping)
        categories = _categories_mapping(mapping)
        return cls(
            version=version,
            language=language,
            matching=dict(matching),
            categories=categories,
        )

    @property
    def term_count(self) -> int:
        """Return the number of unique vocabulary terms."""
        return len(self._normalized_terms)

    def match_text(self, text: str) -> tuple[VocabularyMatch, ...]:
        """Return unique strong terms in their first textual occurrence order."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        spans = self._matcher.find_spans(text, decode_url=False)
        ordered_normalized_terms = sorted(
            spans,
            key=lambda term: (min(start for start, _ in spans[term]), term),
        )
        return tuple(
            VocabularyMatch(
                category=self._term_categories[normalized],
                term=self._normalized_terms[normalized],
            )
            for normalized in ordered_normalized_terms
        )

    def to_record(self) -> dict[str, object]:
        """Return the vocabulary contract for a reproducibility manifest."""
        return {
            "version": self.version,
            "language": self.language,
            "matching": dict(self.matching),
            "categories": {
                category: list(terms) for category, terms in self.categories.items()
            },
            "term_count": self.term_count,
        }


def load_vocabulary(path: Path) -> TopicVocabulary:
    """Load and validate a vocabulary JSON file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read vocabulary: {path}") from error
    return TopicVocabulary.from_mapping(value)


def _vocabulary_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("vocabulary must be a JSON object")
    return value


def _matching_mapping(value: Mapping[str, object]) -> dict[str, object]:
    matching = value.get("matching")
    if not isinstance(matching, Mapping):
        raise ValueError("matching must be a JSON object")
    _validate_matching_contract(matching)
    return dict(matching)


def _categories_mapping(
    value: Mapping[str, object],
) -> dict[str, tuple[str, ...]]:
    raw_categories = value.get("categories")
    if not isinstance(raw_categories, Mapping) or not raw_categories:
        raise ValueError("categories must be a non-empty JSON object")
    categories: dict[str, tuple[str, ...]] = {}
    for raw_category, raw_terms in raw_categories.items():
        category, terms = _category_record(raw_category, raw_terms)
        categories[category] = terms
    return categories


def _category_record(
    raw_category: object, raw_terms: object
) -> tuple[str, tuple[str, ...]]:
    category = _category_name(raw_category)
    terms = _category_terms(category, raw_terms)
    return category, terms


def _category_name(raw_category: object) -> str:
    if not isinstance(raw_category, str) or not raw_category.strip():
        raise ValueError("category names must be non-empty strings")

    return raw_category


def _category_terms(category: str, raw_terms: object) -> tuple[str, ...]:
    if isinstance(raw_terms, str) or not isinstance(raw_terms, Sequence):
        raise ValueError(f"category {category!r} must contain a list")
    terms = tuple(_validate_term(term) for term in raw_terms)
    if not terms:
        raise ValueError(f"category {category!r} must not be empty")
    return terms


def _required_string(value: Mapping[str, object], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return candidate


def _validate_term(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("vocabulary terms must be non-empty strings")
    if not normalize_for_search(value, decode_url=False):
        raise ValueError("vocabulary terms must contain searchable characters")
    return value.strip()


def _validate_matching_contract(matching: Mapping[str, object]) -> None:
    expected = (
        ("field", "text", "field must be text"),
        ("case_sensitive", False, "case_sensitive must be false"),
        ("unicode_normalization", "NFKC", "unicode_normalization must be NFKC"),
        ("word_boundaries", True, "word_boundaries must be true"),
        ("match_url", False, "match_url must be false"),
    )
    for key, required, message in expected:
        if matching.get(key) != required:
            raise ValueError(message)
