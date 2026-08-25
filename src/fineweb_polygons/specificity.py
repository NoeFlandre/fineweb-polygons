"""Frequency-based filtering for V5 polygon names."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor
from pathlib import Path

import pyarrow.parquet as pq

from fineweb_polygons.matching import _MultiPatternMatcher
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.normalization import normalize_for_search

FINEWEB_DOCUMENT_FREQUENCY_RATIO = 0.001


@dataclass(frozen=True, slots=True)
class NameFrequency:
    """Observed frequency for one normalized polygon name."""

    normalized_name: str
    osm_occurrences: int
    fineweb_document_frequency: int

    def to_record(self) -> dict[str, object]:
        """Return a stable JSON-compatible record."""
        return {
            "normalized_name": self.normalized_name,
            "osm_occurrences": self.osm_occurrences,
            "fineweb_document_frequency": self.fineweb_document_frequency,
        }


@dataclass(frozen=True, slots=True)
class SpecificityResult:
    """Filtered profiles and the frequency evidence used to filter them."""

    profiles: tuple[PolygonProfile, ...]
    frequencies: tuple[NameFrequency, ...]
    documents_scanned: int

    @property
    def removed_count(self) -> int:
        """Return the number of candidate profiles rejected by V5."""
        return len(self.frequencies) - len(self.profiles)


def count_fineweb_document_frequencies(
    shard_path: Path,
    profiles: Sequence[PolygonProfile],
    *,
    batch_size: int = 8192,
) -> tuple[dict[str, int], int]:
    """Count how many FineWeb documents contain each candidate name in text."""
    names = tuple(sorted({profile.normalized_name for profile in profiles}))
    parquet_file = pq.ParquetFile(shard_path)
    documents_scanned = parquet_file.metadata.num_rows
    if not names:
        return {}, documents_scanned

    matcher = _MultiPatternMatcher(names)
    frequencies = dict.fromkeys(names, 0)
    return _count_batches(
        parquet_file,
        matcher,
        frequencies,
        batch_size=batch_size,
        documents_scanned=documents_scanned,
    )


def _count_batches(
    parquet_file: pq.ParquetFile,
    matcher: _MultiPatternMatcher,
    frequencies: dict[str, int],
    *,
    batch_size: int,
    documents_scanned: int,
) -> tuple[dict[str, int], int]:
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=["text"],
        use_threads=True,
    ):
        for raw_text in batch.column("text").to_pylist():
            for name in matcher.find(_as_text(raw_text), decode_url=False):
                frequencies[name] += 1
    return frequencies, documents_scanned


def filter_specific_profiles(
    profiles: Sequence[PolygonProfile],
    frequencies: Mapping[str, NameFrequency],
    *,
    country_name: str,
    fineweb_document_frequency_threshold: int,
) -> SpecificityResult:
    """Keep names unique in OSM and uncommon in the FineWeb shard.

    The configured country name is context only, never a polygon candidate.
    """
    _validate_threshold(fineweb_document_frequency_threshold)
    normalized_country = normalize_for_search(country_name, decode_url=False)
    selected: list[PolygonProfile] = []
    ordered_frequencies: list[NameFrequency] = []
    for profile in profiles:
        frequency = frequencies[profile.normalized_name]
        ordered_frequencies.append(frequency)
        if _is_specific_name(
            profile,
            frequency,
            normalized_country=normalized_country,
            fineweb_document_frequency_threshold=fineweb_document_frequency_threshold,
        ):
            selected.append(profile)
    return SpecificityResult(
        profiles=tuple(selected),
        frequencies=tuple(ordered_frequencies),
        documents_scanned=0,
    )


def _validate_threshold(threshold: int) -> None:
    if threshold < 0:
        raise ValueError("fineweb_document_frequency_threshold must be non-negative")


def _is_specific_name(
    profile: PolygonProfile,
    frequency: NameFrequency,
    *,
    normalized_country: str,
    fineweb_document_frequency_threshold: int,
) -> bool:
    return (
        profile.normalized_name != normalized_country
        and frequency.osm_occurrences == 1
        and frequency.fineweb_document_frequency <= fineweb_document_frequency_threshold
    )


def frequency_threshold(document_count: int) -> int:
    """Return the inclusive 0.1% document-frequency cutoff."""
    if document_count < 0:
        raise ValueError("document_count must be non-negative")
    return floor(document_count * FINEWEB_DOCUMENT_FREQUENCY_RATIO)


def _as_text(value: object) -> str:
    return "" if value is None else str(value)
