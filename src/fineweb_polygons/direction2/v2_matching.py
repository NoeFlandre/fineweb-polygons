"""Specificity-aware matching helpers for Direction 2 lexical V2."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fineweb_polygons.direction2.matching import (
    AhoCorasickPatternMatcher,
    PatternMatch,
)
from fineweb_polygons.direction2.v2_specificity import NameCandidate, NameProfile


@dataclass(frozen=True, slots=True)
class V2NameMatch:
    """One accepted name occurrence mapped to a polygon candidate."""

    profile: NameProfile
    candidate: NameCandidate
    start: int
    end: int


class V2NameMatcher:
    """Match only names accepted by the V2 specificity policy."""

    def __init__(
        self,
        pattern_matcher: AhoCorasickPatternMatcher,
        profiles: dict[str, NameProfile],
    ) -> None:
        self._pattern_matcher = pattern_matcher
        self._profiles = profiles

    @classmethod
    def build(cls, profiles: Sequence[NameProfile]) -> V2NameMatcher:
        """Build one matcher over all non-discarded profiles."""
        selected = {
            profile.normalized_name: profile
            for profile in profiles
            if profile.decision.decision != "discard"
        }
        return cls(
            AhoCorasickPatternMatcher.build(selected),
            selected,
        )

    @property
    def names_indexed(self) -> int:
        """Return the count of accepted normalized names."""
        return self._pattern_matcher.patterns_indexed

    def find_unique_patterns(self, text: str) -> tuple[str, ...]:
        """Return accepted patterns once each in the document."""
        return self._pattern_matcher.find_unique_patterns(text)

    def find(self, text: str) -> tuple[V2NameMatch, ...]:
        """Return all accepted polygon/alias occurrences."""
        matches = [
            V2NameMatch(
                profile=self._profiles[pattern.pattern],
                candidate=candidate,
                start=pattern.start,
                end=pattern.end,
            )
            for pattern in self._pattern_matcher.find(text)
            for candidate in self._profiles[pattern.pattern].candidates
        ]
        return tuple(
            sorted(
                matches,
                key=lambda match: (
                    match.start,
                    match.end,
                    match.candidate.polygon_id,
                    match.candidate.alias,
                ),
            )
        )


def has_independent_country_match(
    country_matches: Iterable[PatternMatch],
    *,
    name_start: int,
    name_end: int,
) -> bool:
    """Return whether a country match exists outside the name span."""
    return any(
        match.end <= name_start or match.start >= name_end for match in country_matches
    )
