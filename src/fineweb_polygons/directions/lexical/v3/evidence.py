"""Pure evidence scoring for Direction 2 lexical V3."""

from __future__ import annotations

from dataclasses import dataclass

from fineweb_polygons.core.normalization import has_context_marker
from fineweb_polygons.directions.lexical.v2.specificity import NameProfile
from fineweb_polygons.directions.lexical.v3.models import DecisionTier


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """Auditable booleans, score, and tier for one lexical occurrence."""

    name_in_url: bool
    country_in_sentence: bool
    country_in_context: bool
    same_polygon_alias_nearby: bool
    other_polygon_name_nearby: bool
    score: int
    tier: DecisionTier
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        """Return the stable evidence fields written to Parquet."""
        return {
            "decision_tier": self.tier,
            "evidence_reasons": "|".join(self.reasons),
            "evidence_score": self.score,
            "name_in_url": self.name_in_url,
            "country_in_sentence": self.country_in_sentence,
            "country_in_context": self.country_in_context,
            "same_polygon_alias_nearby": self.same_polygon_alias_nearby,
            "other_polygon_name_nearby": self.other_polygon_name_nearby,
        }


def score_candidate(
    profile: NameProfile,
    *,
    alias: str,
    sentence: str,
    context: str,
    url: str,
    country_name: str,
    same_polygon_alias_nearby: bool,
    other_polygon_name_nearby: bool,
) -> CandidateEvidence:
    """Score one occurrence using only deterministic lexical evidence."""
    _validate_country_name(country_name)
    name_in_url, country_in_sentence, country_in_context = _text_signals(
        alias=alias,
        sentence=sentence,
        context=context,
        url=url,
        country_name=country_name,
    )
    signals = _signals(
        profile=profile,
        name_in_url=name_in_url,
        country_in_sentence=country_in_sentence,
        country_in_context=country_in_context,
        same_polygon_alias_nearby=same_polygon_alias_nearby,
        other_polygon_name_nearby=other_polygon_name_nearby,
    )
    score, reasons = _score_signals(signals)
    return CandidateEvidence(
        name_in_url=name_in_url,
        country_in_sentence=country_in_sentence,
        country_in_context=country_in_context,
        same_polygon_alias_nearby=same_polygon_alias_nearby,
        other_polygon_name_nearby=other_polygon_name_nearby,
        score=score,
        tier=_tier(score),
        reasons=reasons,
    )


def _validate_country_name(country_name: str) -> None:
    if not country_name.strip():
        raise ValueError("country_name must not be empty")


def _text_signals(
    *,
    alias: str,
    sentence: str,
    context: str,
    url: str,
    country_name: str,
) -> tuple[bool, bool, bool]:
    name_in_url = has_context_marker(url, alias)
    country_in_sentence = has_context_marker(sentence, country_name)
    country_in_context = not country_in_sentence and has_context_marker(
        context, country_name
    )
    return name_in_url, country_in_sentence, country_in_context


def _signals(
    *,
    profile: NameProfile,
    name_in_url: bool,
    country_in_sentence: bool,
    country_in_context: bool,
    same_polygon_alias_nearby: bool,
    other_polygon_name_nearby: bool,
) -> tuple[tuple[str, int, bool], ...]:
    return (
        ("distinctive_name", 2, profile.decision.decision == "distinctive"),
        ("name_in_url", 3, name_in_url),
        ("country_in_sentence", 3, country_in_sentence),
        ("country_in_context", 1, country_in_context),
        ("same_polygon_alias_nearby", 2, same_polygon_alias_nearby),
        ("other_polygon_name_nearby", 1, other_polygon_name_nearby),
    )


def _score_signals(
    signals: tuple[tuple[str, int, bool], ...],
) -> tuple[int, tuple[str, ...]]:
    active = tuple((name, points) for name, points, enabled in signals if enabled)
    return sum(points for _, points in active), tuple(name for name, _ in active)


def _tier(score: int) -> DecisionTier:
    if score >= 4:
        return "high_confidence"
    if score >= 2:
        return "possible"
    return "rejected"
