"""Immutable retrieval-version definitions used by runs and dataset releases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

RetrievalVersion = Literal["v1", "v2"]


@dataclass(frozen=True, slots=True)
class RetrievalDefinition:
    """The stable, human-readable contract for one retrieval version."""

    version: RetrievalVersion
    title: str
    polygon_profile_version: str
    matcher_version: str
    polygon_rule: str
    document_rule: str
    evidence_rule: str
    requires_text_context: bool

    def to_record(self) -> dict[str, object]:
        """Return the definition embedded in a run manifest."""
        return {
            "version": self.version,
            "title": self.title,
            "polygon_profile_version": self.polygon_profile_version,
            "matcher_version": self.matcher_version,
            "polygon_rule": self.polygon_rule,
            "document_rule": self.document_rule,
            "evidence_rule": self.evidence_rule,
            "requires_text_context": self.requires_text_context,
        }


_DEFINITIONS: Final[dict[RetrievalVersion, RetrievalDefinition]] = {
    "v1": RetrievalDefinition(
        version="v1",
        title="Named polygon exact matching",
        polygon_profile_version="v1-all-named-closed-ways-and-polygon-relations",
        matcher_version="v1-exact-name-context",
        polygon_rule=(
            "Keep every named closed way and polygon relation read from the PBF."
        ),
        document_rule=(
            "Keep a document when a polygon name and Monaco context appear in "
            "either FineWeb text or URL."
        ),
        evidence_rule=(
            "Use case-insensitive exact normalized matching and retain the full "
            "FineWeb text."
        ),
        requires_text_context=False,
    ),
    "v2": RetrievalDefinition(
        version="v2",
        title="Meaningful in-boundary polygon exact matching",
        polygon_profile_version="v2-in-boundary-meaningful-names",
        matcher_version="v2-exact-name-url-or-text-with-text-country-context",
        polygon_rule=(
            "Keep named area objects inside the Monaco admin_level=8 city "
            "boundary; reject numeric-only or shorter-than-three-character names "
            "and deduplicate normalized names."
        ),
        document_rule=(
            "Keep a document when the polygon name is in the URL, or when the "
            "name and Monaco or Principality of Monaco are both in the text."
        ),
        evidence_rule=(
            "Use case-insensitive exact normalized matching and retain the full "
            "FineWeb text plus evidence fields and excerpts."
        ),
        requires_text_context=True,
    ),
}


def get_retrieval_definition(version: str) -> RetrievalDefinition:
    """Return a frozen definition or reject an unknown version."""
    definition = _DEFINITIONS.get(version)
    if definition is None:
        raise ValueError("retrieval_version must be v1 or v2") from None
    return definition


def available_retrieval_versions() -> tuple[RetrievalVersion, ...]:
    """Return the version IDs in their stable publication order."""
    return tuple(_DEFINITIONS)
