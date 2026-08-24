"""Small immutable records shared by the V1 pipeline modules."""

from __future__ import annotations

from dataclasses import dataclass

from fineweb_polygons.normalization import normalize_for_search


@dataclass(frozen=True, slots=True)
class PolygonProfile:
    """The minimal named-polygon query used by V1."""

    polygon_id: str
    name: str
    normalized_name: str

    @classmethod
    def create(cls, polygon_id: str, name: str) -> PolygonProfile:
        """Create a profile while retaining its source and search names."""
        return cls(
            polygon_id=polygon_id,
            name=name,
            normalized_name=normalize_for_search(name),
        )


@dataclass(frozen=True, slots=True)
class FineWebDocument:
    """The projected FineWeb fields needed by the matcher."""

    row_index: int
    document_id: str | None
    text: str
    url: str


@dataclass(frozen=True, slots=True)
class MatchEvidence:
    """Auditable evidence for one polygon/document match."""

    polygon_id: str
    polygon_name: str
    fineweb_row_index: int
    fineweb_document_id: str | None
    url: str
    matched_fields: tuple[str, ...]
    context_fields: tuple[str, ...]
    matched_name: str
    context_phrase: str
    text_excerpt: str
    url_excerpt: str

    def to_record(self) -> dict[str, object]:
        """Return a JSON-compatible evidence record."""
        return {
            "polygon_id": self.polygon_id,
            "polygon_name": self.polygon_name,
            "fineweb_row_index": self.fineweb_row_index,
            "fineweb_document_id": self.fineweb_document_id,
            "url": self.url,
            "matched_fields": list(self.matched_fields),
            "context_fields": list(self.context_fields),
            "matched_name": self.matched_name,
            "context_phrase": self.context_phrase,
            "text_excerpt": self.text_excerpt,
            "url_excerpt": self.url_excerpt,
        }
