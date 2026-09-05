"""Data-driven name specificity rules for Direction 2 lexical V2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor
from typing import Literal

from fineweb_polygons.direction2.models import PolygonRecord
from fineweb_polygons.normalization import normalize_for_search

FINEWEB_DOCUMENT_FREQUENCY_RATIO = 0.001
SHORT_SINGLE_TOKEN_MAX_LETTERS = 8
MIN_NAME_LETTERS = 3
NameDecisionType = Literal["discard", "generic", "distinctive"]


@dataclass(frozen=True, slots=True)
class NameDecision:
    """The deterministic V2 decision for one normalized name."""

    normalized_name: str
    decision: NameDecisionType
    reason: str
    polygon_count: int
    document_frequency: int
    token_count: int
    letter_count: int
    frequency_cutoff: int

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible decision record."""
        return {
            "decision": self.decision,
            "document_frequency": self.document_frequency,
            "frequency_cutoff": self.frequency_cutoff,
            "letter_count": self.letter_count,
            "normalized_name": self.normalized_name,
            "polygon_count": self.polygon_count,
            "reason": self.reason,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class NameCandidate:
    """One original polygon alias represented by a normalized name."""

    polygon: PolygonRecord
    alias: str
    normalized_name: str

    @property
    def polygon_id(self) -> str:
        """Return the stable polygon identifier."""
        return self.polygon.polygon_id

    def to_record(self) -> dict[str, object]:
        """Return the candidate fields needed to audit an inventory."""
        return {
            "alias": self.alias,
            "normalized_name": self.normalized_name,
            "polygon_id": self.polygon.polygon_id,
            "source_key": self.polygon.source_key,
        }


@dataclass(frozen=True, slots=True)
class NameProfile:
    """All polygon aliases sharing one normalized name and its decision."""

    normalized_name: str
    candidates: tuple[NameCandidate, ...]
    osm_polygon_count: int
    decision: NameDecision

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible name profile."""
        return {
            "candidates": [candidate.to_record() for candidate in self.candidates],
            "decision": self.decision.to_record(),
            "normalized_name": self.normalized_name,
            "osm_polygon_count": self.osm_polygon_count,
        }


def searchable_name_patterns(
    profiles: Sequence[NameProfile],
) -> tuple[str, ...]:
    """Return names that have enough letters to search efficiently."""
    return tuple(
        profile.normalized_name
        for profile in profiles
        if profile.decision.reason not in {"no_letters", "too_few_letters"}
    )


def build_name_inventory(
    polygons: Sequence[PolygonRecord],
    *,
    document_frequencies: Mapping[str, int],
    document_count: int,
    country_names: Mapping[str, str],
) -> tuple[NameProfile, ...]:
    """Build stable name profiles from polygon records and frequencies."""
    grouped = _group_candidates(polygons)
    normalized_countries = _normalized_countries(country_names)
    return tuple(
        _build_profile(
            normalized_name,
            grouped[normalized_name],
            document_frequencies=document_frequencies,
            document_count=document_count,
            normalized_countries=normalized_countries,
            country_names=country_names,
        )
        for normalized_name in sorted(grouped)
    )


def _group_candidates(
    polygons: Sequence[PolygonRecord],
) -> dict[str, dict[tuple[str, str], NameCandidate]]:
    grouped: dict[str, dict[tuple[str, str], NameCandidate]] = {}
    for polygon in polygons:
        for alias in polygon.candidate_names():
            normalized_name = normalize_for_search(
                alias,
                decode_url=False,  # pragma: no mutate
            )
            if normalized_name:
                candidate = NameCandidate(polygon, alias, normalized_name)
                grouped.setdefault(normalized_name, {})[(polygon.polygon_id, alias)] = (
                    candidate
                )
    return grouped


def _normalized_countries(country_names: Mapping[str, str]) -> dict[str, str]:
    return {
        source_key: normalize_for_search(
            country,
            decode_url=False,  # pragma: no mutate
        )
        for source_key, country in country_names.items()
    }


def _build_profile(
    normalized_name: str,
    candidate_map: Mapping[tuple[str, str], NameCandidate],
    *,
    document_frequencies: Mapping[str, int],
    document_count: int,
    normalized_countries: Mapping[str, str],
    country_names: Mapping[str, str],
) -> NameProfile:
    candidates = tuple(
        sorted(
            candidate_map.values(),
            key=lambda candidate: (candidate.polygon_id, candidate.alias),
        )
    )
    polygon_count = len({candidate.polygon_id for candidate in candidates})
    country_name = _country_for_name(
        normalized_name,
        candidates,
        normalized_countries,
        country_names,
    )
    decision = classify_name(
        normalized_name,
        polygon_count=polygon_count,
        document_frequency=document_frequencies.get(normalized_name, 0),
        document_count=document_count,
        country_name=country_name,
    )
    return NameProfile(
        normalized_name=normalized_name,
        candidates=candidates,
        osm_polygon_count=polygon_count,
        decision=decision,
    )


def classify_name(
    name: str,
    *,
    polygon_count: int,
    document_frequency: int,
    document_count: int,
    country_name: str | None = None,
) -> NameDecision:
    """Classify a name as discard, generic, or distinctive."""
    _validate_counts(polygon_count, document_frequency, document_count)
    normalized_name = normalize_for_search(
        name,
        decode_url=False,  # pragma: no mutate
    )
    token_count = len(normalized_name.split())
    letter_count = sum(character.isalpha() for character in normalized_name)
    frequency_cutoff = fineweb_frequency_cutoff(document_count)
    discard_reason = _discard_reason(
        normalized_name,
        letter_count=letter_count,
        country_name=country_name,
    )
    generic_reason = (
        None
        if discard_reason is not None
        else _generic_reason(
            polygon_count=polygon_count,
            document_frequency=document_frequency,
            frequency_cutoff=frequency_cutoff,
            token_count=token_count,
            letter_count=letter_count,
        )
    )
    decision: NameDecisionType
    if discard_reason is not None:
        decision = "discard"
        reason = discard_reason
    elif generic_reason is not None:
        decision = "generic"
        reason = generic_reason
    else:
        decision = "distinctive"
        reason = "specific"
    return NameDecision(
        normalized_name=normalized_name,
        decision=decision,
        reason=reason,
        polygon_count=polygon_count,
        document_frequency=document_frequency,
        token_count=token_count,
        letter_count=letter_count,
        frequency_cutoff=frequency_cutoff,
    )


def _discard_reason(
    normalized_name: str,
    *,
    letter_count: int,
    country_name: str | None,
) -> str | None:
    if letter_count == 0:
        return "no_letters"
    if letter_count < MIN_NAME_LETTERS:
        return "too_few_letters"
    if country_name and normalized_name == normalize_for_search(
        country_name,
        decode_url=False,  # pragma: no mutate
    ):
        return "country_name"
    return None


def _generic_reason(
    *,
    polygon_count: int,
    document_frequency: int,
    frequency_cutoff: int,
    token_count: int,
    letter_count: int,
) -> str | None:
    if polygon_count > 1:
        return "osm_reuse"
    if document_frequency > frequency_cutoff:
        return "fineweb_frequency"
    if token_count == 1 and letter_count <= SHORT_SINGLE_TOKEN_MAX_LETTERS:
        return "short_single_token"
    return None


def _country_for_name(
    normalized_name: str,
    candidates: Sequence[NameCandidate],
    normalized_countries: Mapping[str, str],
    country_names: Mapping[str, str],
) -> str | None:
    for candidate in candidates:
        if normalized_countries.get(candidate.polygon.source_key) == normalized_name:
            return country_names[candidate.polygon.source_key]
    return None


def fineweb_frequency_cutoff(document_count: int) -> int:
    """Return the inclusive maximum frequency allowed for a specific name."""
    if document_count < 0:
        raise ValueError("document_count must be non-negative")
    if document_count == 0:
        return 0
    return max(1, floor(document_count * FINEWEB_DOCUMENT_FREQUENCY_RATIO))


def _validate_counts(
    polygon_count: int,
    document_frequency: int,
    document_count: int,
) -> None:
    for value, name in (
        (polygon_count, "polygon_count"),
        (document_frequency, "document_frequency"),
        (document_count, "document_count"),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
