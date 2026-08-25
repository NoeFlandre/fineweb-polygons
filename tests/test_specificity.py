from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fineweb_polygons.specificity as specificity_module
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.specificity import (
    NameFrequency,
    _as_text,
    _count_batches,
    _MultiPatternMatcher,
    _validate_threshold,
    count_fineweb_document_frequencies,
    filter_specific_profiles,
    frequency_threshold,
)


def test_frequency_filter_drops_repeated_common_and_country_names() -> None:
    profiles = (
        PolygonProfile.create("way/1", "Rare Place"),
        PolygonProfile.create("way/2", "Shared Name"),
        PolygonProfile.create("way/3", "Common Name"),
        PolygonProfile.create("relation/4", "Liechtenstein"),
    )
    frequencies = {
        "rare place": NameFrequency("rare place", 1, 1),
        "shared name": NameFrequency("shared name", 2, 1),
        "common name": NameFrequency("common name", 1, 3),
        "liechtenstein": NameFrequency("liechtenstein", 10, 999),
    }

    result = filter_specific_profiles(
        profiles,
        frequencies,
        country_name="Liechtenstein",
        fineweb_document_frequency_threshold=2,
    )

    assert [profile.name for profile in result.profiles] == [
        "Rare Place",
    ]
    assert result.removed_count == 3
    assert result.frequencies == (
        frequencies["rare place"],
        frequencies["shared name"],
        frequencies["common name"],
        frequencies["liechtenstein"],
    )
    assert result.documents_scanned == 0


def test_frequency_filter_keeps_the_inclusive_cutoff_and_casefolds_country() -> None:
    profile = PolygonProfile.create("way/1", "At Limit")
    country = PolygonProfile.create("relation/2", "LIECHTENSTEIN")

    result = filter_specific_profiles(
        (profile, country),
        {
            "at limit": NameFrequency("at limit", 1, 2),
            "liechtenstein": NameFrequency("liechtenstein", 1, 0),
        },
        country_name="Liechtenstein",
        fineweb_document_frequency_threshold=2,
    )

    assert result.profiles == (profile,)


def test_frequency_filter_normalizes_country_without_url_decoding(monkeypatch) -> None:
    calls: list[tuple[object, bool]] = []

    def fake_normalize(value: object, *, decode_url: bool = True) -> str:
        calls.append((value, decode_url))
        return "liechtenstein"

    monkeypatch.setattr(specificity_module, "normalize_for_search", fake_normalize)

    result = filter_specific_profiles(
        (PolygonProfile.create("relation/1", "Liechtenstein"),),
        {"liechtenstein": NameFrequency("liechtenstein", 1, 0)},
        country_name="Liechtenstein",
        fineweb_document_frequency_threshold=0,
    )

    assert result.profiles == ()
    assert calls == [("Liechtenstein", False)]


def test_frequency_counter_uses_its_stable_default_batch_size(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class FakeMetadata:
        num_rows = 3

    class FakeParquetFile:
        metadata = FakeMetadata()

    def fake_parquet_file(path: Path) -> FakeParquetFile:
        calls["path"] = path
        return FakeParquetFile()

    def fake_count_batches(*args, **kwargs):
        del args
        calls["batch_size"] = kwargs["batch_size"]
        return {"rare place": 0}, 3

    monkeypatch.setattr(specificity_module.pq, "ParquetFile", fake_parquet_file)
    monkeypatch.setattr(specificity_module, "_count_batches", fake_count_batches)
    profile = PolygonProfile.create("way/1", "Rare Place")

    assert count_fineweb_document_frequencies(Path("shard.parquet"), (profile,)) == (
        {"rare place": 0},
        3,
    )
    assert calls == {"path": Path("shard.parquet"), "batch_size": 8192}


def test_specificity_rejects_a_negative_frequency_threshold() -> None:
    with pytest.raises(
        ValueError,
        match=r"\Afineweb_document_frequency_threshold must be non-negative\Z",
    ):
        _validate_threshold(-1)


def test_frequency_threshold_validates_and_rounds_down() -> None:
    assert frequency_threshold(0) == 0
    assert frequency_threshold(1001) == 1
    with pytest.raises(ValueError, match=r"\Adocument_count must be non-negative\Z"):
        frequency_threshold(-1)


def test_as_text_handles_null_and_non_string_values() -> None:
    assert _as_text(None) == ""
    assert _as_text(42) == "42"


def test_count_batches_uses_text_only_threaded_batches_and_text_matching() -> None:
    calls: list[dict[str, object]] = []
    matcher_calls: list[tuple[str, bool]] = []

    class FakeColumn:
        def to_pylist(self) -> list[str]:
            return ["Rare Place"]

    class FakeBatch:
        def column(self, name: str) -> FakeColumn:
            assert name == "text"
            return FakeColumn()

    class FakeParquetFile:
        def iter_batches(self, **kwargs: object) -> list[FakeBatch]:
            calls.append(kwargs)
            return [FakeBatch()]

    class FakeMatcher:
        def find(self, value: str, *, decode_url: bool) -> frozenset[str]:
            matcher_calls.append((value, decode_url))
            return frozenset({"rare place"})

    frequencies, documents_scanned = _count_batches(
        cast(pq.ParquetFile, FakeParquetFile()),
        cast(_MultiPatternMatcher, FakeMatcher()),
        {"rare place": 0},
        batch_size=7,
        documents_scanned=1,
    )

    assert calls == [{"batch_size": 7, "columns": ["text"], "use_threads": True}]
    assert matcher_calls == [("Rare Place", False)]
    assert frequencies == {"rare place": 1}
    assert documents_scanned == 1


def test_fineweb_frequency_counts_documents_once_per_name(tmp_path: Path) -> None:
    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [
                    "Rare Place is here.",
                    "Rare Place and Another Place are here.",
                    "Rare Place appears again.",
                ],
                "url": ["", "", ""],
            }
        ),
        shard,
    )
    profiles = (
        PolygonProfile.create("way/1", "Rare Place"),
        PolygonProfile.create("way/2", "Another Place"),
    )

    frequencies, documents_scanned = count_fineweb_document_frequencies(shard, profiles)

    assert frequencies == {"another place": 1, "rare place": 3}
    assert documents_scanned == 3
