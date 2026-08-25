from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

import fineweb_polygons.topic_vocabulary as topic_module
from fineweb_polygons.topic_vocabulary import (
    TopicVocabulary,
    VocabularyMatch,
    load_vocabulary,
)


def _assert_value_error(call: Callable[[], object], message: str) -> None:
    with pytest.raises(ValueError) as error:
        call()
    assert str(error.value) == message


def _mapping() -> dict[str, object]:
    return {
        "version": "v8-topic-vocabulary-v1",
        "language": "en",
        "matching": {
            "field": "text",
            "case_sensitive": False,
            "unicode_normalization": "NFKC",
            "word_boundaries": True,
            "match_url": False,
        },
        "categories": {
            "land_use": ["land-use"],
            "soil_surface": ["soil"],
            "vegetation_ecosystem": ["forest"],
        },
    }


def test_load_vocabulary_preserves_categories_and_term_count(tmp_path: Path) -> None:
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps(_mapping()), encoding="utf-8")

    vocabulary = load_vocabulary(path)

    assert vocabulary.version == "v8-topic-vocabulary-v1"
    assert vocabulary.term_count == 3
    assert vocabulary.categories == {
        "land_use": ("land-use",),
        "soil_surface": ("soil",),
        "vegetation_ecosystem": ("forest",),
    }


def test_matches_are_case_insensitive_normalized_and_categorized() -> None:
    vocabulary = TopicVocabulary.from_mapping(_mapping())

    assert vocabulary.match_text(
        "\uff26\uff2f\uff32\uff25\uff33\uff34 soil and LAND USE"
    ) == (
        VocabularyMatch(category="vegetation_ecosystem", term="forest"),
        VocabularyMatch(category="soil_surface", term="soil"),
        VocabularyMatch(category="land_use", term="land-use"),
    )


def test_matching_uses_exact_word_boundaries() -> None:
    vocabulary = TopicVocabulary.from_mapping(_mapping())

    assert vocabulary.match_text("Forestry and soiled land-useful areas") == ()


def test_matching_deduplicates_repeated_terms() -> None:
    vocabulary = TopicVocabulary.from_mapping(_mapping())

    assert vocabulary.match_text("forest forest soil") == (
        VocabularyMatch(category="vegetation_ecosystem", term="forest"),
        VocabularyMatch(category="soil_surface", term="soil"),
    )


def test_rejects_duplicate_normalized_terms() -> None:
    mapping = _mapping()
    mapping["categories"] = {
        "land_use": ["forest"],
        "vegetation_ecosystem": ["FOREST"],
    }

    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "duplicate vocabulary term: FOREST",
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("field", "url", "field must be text"),
        ("case_sensitive", True, "case_sensitive must be false"),
        ("unicode_normalization", "NFD", "unicode_normalization must be NFKC"),
        ("word_boundaries", False, "word_boundaries must be true"),
        ("match_url", True, "match_url must be false"),
    ],
)
def test_rejects_each_unsupported_matching_contract_value(
    key: str, value: object, message: str
) -> None:
    mapping = _mapping()
    matching = dict(cast(Mapping[str, object], mapping["matching"]))
    matching[key] = value
    mapping["matching"] = matching

    _assert_value_error(lambda: TopicVocabulary.from_mapping(mapping), message)


@pytest.mark.parametrize("key", ["version", "language"])
def test_rejects_missing_required_vocabulary_strings(key: str) -> None:
    mapping = _mapping()
    del mapping[key]

    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        f"{key} must be a non-empty string",
    )


def test_rejects_invalid_vocabulary_shapes() -> None:
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping([]),
        "vocabulary must be a JSON object",
    )

    mapping = _mapping()
    mapping["matching"] = None
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "matching must be a JSON object",
    )

    mapping = _mapping()
    mapping["categories"] = {}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "categories must be a non-empty JSON object",
    )

    mapping = _mapping()
    mapping["categories"] = {None: ["term"]}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "category names must be non-empty strings",
    )

    mapping = _mapping()
    mapping["categories"] = {"category": "term"}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "category 'category' must contain a list",
    )

    mapping = _mapping()
    mapping["categories"] = {"category": 3}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "category 'category' must contain a list",
    )

    mapping = _mapping()
    mapping["categories"] = {"category": []}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "category 'category' must not be empty",
    )

    mapping = _mapping()
    mapping["categories"] = {"category": [""]}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "vocabulary terms must be non-empty strings",
    )

    mapping = _mapping()
    mapping["categories"] = {"category": ["!!!"]}
    _assert_value_error(
        lambda: TopicVocabulary.from_mapping(mapping),
        "vocabulary terms must contain searchable characters",
    )


def test_load_vocabulary_wraps_missing_and_invalid_json(tmp_path: Path) -> None:
    _assert_value_error(
        lambda: load_vocabulary(tmp_path / "missing.json"),
        f"could not read vocabulary: {tmp_path / 'missing.json'}",
    )

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not json", encoding="utf-8")
    _assert_value_error(
        lambda: load_vocabulary(invalid),
        f"could not read vocabulary: {invalid}",
    )


def test_load_vocabulary_requires_utf8_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_read_text(self: Path, *, encoding: str | None = None) -> str:
        captured["encoding"] = encoding
        return json.dumps(_mapping())

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    load_vocabulary(tmp_path / "vocabulary.json")

    assert captured == {"encoding": "utf-8"}


def test_validate_term_disables_url_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[object] = []

    def fake_normalize(value: str, *, decode_url: bool = True) -> str:
        captured.append(decode_url)
        return value

    monkeypatch.setattr(topic_module, "normalize_for_search", fake_normalize)

    assert topic_module._validate_term("%20") == "%20"
    assert captured == [False]


def test_match_text_requires_a_string() -> None:
    vocabulary = TopicVocabulary.from_mapping(_mapping())

    with pytest.raises(TypeError, match="text must be a string"):
        vocabulary.match_text(cast(str, None))
