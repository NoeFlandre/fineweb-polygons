"""Immutable retrieval-version definitions used by runs and dataset releases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

RetrievalVersion = Literal["v1", "v2", "v3", "v4", "v5", "v6"]


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
    requires_url_name: bool
    deduplicate_documents: bool
    max_name_country_distance: int | None = None

    @property
    def requires_text_name(self) -> bool:
        """Whether the polygon name must be found in FineWeb text."""
        return self.version in ("v4", "v5", "v6")

    @property
    def requires_name_specificity(self) -> bool:
        """Whether V5 frequency filtering must be applied to polygon names."""
        return self.version in ("v5", "v6")

    def to_record(self) -> dict[str, object]:
        """Return the definition embedded in a run manifest."""
        record = {
            "version": self.version,
            "title": self.title,
            "polygon_profile_version": self.polygon_profile_version,
            "matcher_version": self.matcher_version,
            "polygon_rule": self.polygon_rule,
            "document_rule": self.document_rule,
            "evidence_rule": self.evidence_rule,
            "requires_text_context": self.requires_text_context,
            "requires_url_name": self.requires_url_name,
            "deduplicate_documents": self.deduplicate_documents,
        }
        if self.requires_text_name:
            record["requires_text_name"] = True
        if self.requires_name_specificity:
            record["requires_name_specificity"] = True
        if self.max_name_country_distance is not None:
            record["max_name_country_distance"] = self.max_name_country_distance
        return record


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
        requires_url_name=False,
        deduplicate_documents=False,
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
        requires_url_name=False,
        deduplicate_documents=False,
    ),
    "v3": RetrievalDefinition(
        version="v3",
        title="All meaningful polygon areas with URL-and-text exact matching",
        polygon_profile_version="v3-all-meaningful-polygon-areas",
        matcher_version="v3-exact-name-url-and-text-context",
        polygon_rule=(
            "Keep every named closed way and valid polygon relation; reject "
            "numeric-only or shorter-than-three-character names and deduplicate "
            "normalized names."
        ),
        document_rule=(
            "Keep a document only when the polygon name is in the URL and the "
            "name plus Monaco or Principality of Monaco are both in the text."
        ),
        evidence_rule=(
            "Use case-insensitive exact normalized matching and retain the full "
            "FineWeb text plus evidence fields and excerpts."
        ),
        requires_text_context=True,
        requires_url_name=True,
        deduplicate_documents=True,
    ),
    "v4": RetrievalDefinition(
        version="v4",
        title="All meaningful polygon areas with text-only exact matching",
        polygon_profile_version="v4-all-meaningful-polygon-areas",
        matcher_version="v4-exact-name-and-text-context",
        polygon_rule=(
            "Keep every named closed way and valid polygon relation; reject "
            "numeric-only or shorter-than-three-character names and deduplicate "
            "normalized names."
        ),
        document_rule=(
            "Keep a document only when the polygon name and Monaco or "
            "Principality of Monaco both appear in FineWeb text; the URL is not "
            "a selection condition."
        ),
        evidence_rule=(
            "Use case-insensitive exact normalized matching in text and retain "
            "the full FineWeb text, URL, evidence fields, and excerpts."
        ),
        requires_text_context=True,
        requires_url_name=False,
        deduplicate_documents=True,
    ),
    "v5": RetrievalDefinition(
        version="v5",
        title="Specific polygon areas with country-in-text matching",
        polygon_profile_version="v5-specific-meaningful-polygon-areas",
        matcher_version="v5-exact-specific-name-and-country-in-text",
        polygon_rule=(
            "Start with every meaningful named closed way and polygon relation. "
            "Keep a name only when it is not the configured country name, occurs "
            "once in the PBF, and occurs in no more than 0.1% of FineWeb documents."
        ),
        document_rule=(
            "Keep a document only when the selected polygon name and the exact "
            "configured country name both appear in the FineWeb text. The URL "
            "does not select documents."
        ),
        evidence_rule=(
            "Use case-insensitive exact normalized matching and retain the full "
            "FineWeb text, URL, evidence fields, and excerpts."
        ),
        requires_text_context=True,
        requires_url_name=False,
        deduplicate_documents=True,
    ),
    "v6": RetrievalDefinition(
        version="v6",
        title="Specific polygon areas with local text-span matching",
        polygon_profile_version="v6-specific-meaningful-polygon-areas",
        matcher_version=(
            "v6-exact-specific-name-and-country-within-500-normalized-characters"
        ),
        polygon_rule=(
            "Start with every meaningful named closed way and polygon relation. "
            "Keep a name only when it is not the configured country name, occurs "
            "once in the PBF, and occurs in no more than 0.1% of FineWeb documents."
        ),
        document_rule=(
            "Keep a document only when the selected polygon name and the exact "
            "configured country name both appear in the FineWeb text with at most "
            "500 normalized characters between their closest spans. The URL "
            "does not select documents."
        ),
        evidence_rule=(
            "Use case-insensitive exact normalized matching, retain the full "
            "FineWeb text, and save the closest normalized name-country distance "
            "plus both original-text sentences."
        ),
        requires_text_context=True,
        requires_url_name=False,
        deduplicate_documents=True,
        max_name_country_distance=500,
    ),
}


def get_retrieval_definition(version: str) -> RetrievalDefinition:
    """Return a frozen definition or reject an unknown version."""
    definition = _DEFINITIONS.get(version)
    if definition is None:
        raise ValueError(
            "retrieval_version must be v1, v2, v3, v4, v5, or v6"
        ) from None
    return definition


def available_retrieval_versions() -> tuple[RetrievalVersion, ...]:
    """Return the version IDs in their stable publication order."""
    return tuple(_DEFINITIONS)
