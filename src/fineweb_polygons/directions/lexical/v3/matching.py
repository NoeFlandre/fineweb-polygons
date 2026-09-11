"""Scored matching helpers for Direction 2 lexical V3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from fineweb_polygons.directions.lexical.matching import (
    AhoCorasickPatternMatcher,
)
from fineweb_polygons.directions.lexical.v2.specificity import (
    NameCandidate,
    NameProfile,
)


@dataclass(frozen=True, slots=True)
class V3NameMatch:
    """One indexed name occurrence mapped to a polygon candidate."""

    profile: NameProfile
    candidate: NameCandidate
    start: int
    end: int


class V3NameMatcher:
    """Match every V2-accepted name while retaining its measured profile."""

    def __init__(
        self,
        pattern_matcher: AhoCorasickPatternMatcher,
        profiles: dict[str, NameProfile],
    ) -> None:
        self._pattern_matcher = pattern_matcher
        self._profiles = profiles

    @classmethod
    def build(cls, profiles: Sequence[NameProfile]) -> V3NameMatcher:
        """Build an automaton over every non-discarded V2 profile."""
        selected = {
            profile.normalized_name: profile
            for profile in profiles
            if profile.decision.decision != "discard"
        }
        return cls(AhoCorasickPatternMatcher.build(selected), selected)

    @property
    def names_indexed(self) -> int:
        """Return the number of normalized names in the automaton."""
        return self._pattern_matcher.patterns_indexed

    def find_unique_patterns(self, text: str) -> tuple[str, ...]:
        """Return each matched normalized name once for a document."""
        return self._pattern_matcher.find_unique_patterns(text)

    def find(self, text: str) -> tuple[V3NameMatch, ...]:
        """Return all candidate occurrences in stable source order."""
        matches = [
            V3NameMatch(
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
