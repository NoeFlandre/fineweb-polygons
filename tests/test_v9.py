from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fineweb_polygons.artifact_io as artifact_io
import fineweb_polygons.v9 as v9_module
from fineweb_polygons.v9 import V9RunConfig, run_v9


def _vocabulary_mapping() -> dict[str, object]:
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
            "land_use": ["park"],
            "soil_surface": ["coastal"],
            "vegetation_ecosystem": ["forest"],
        },
    }


def _write_vocabulary(path: Path) -> None:
    path.write_text(
        json.dumps(_vocabulary_mapping(), ensure_ascii=False), encoding="utf-8"
    )


def _write_v8_input(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "polygon_id": "way/1",
            "polygon_name": "Fontvieille",
            "context_phrase": "Monaco",
            "context_fields": ["text"],
            "country_name_sentence": "Fontvieille is in Monaco.",
            "matched_fields": ["text"],
            "matched_name": "Fontvieille",
            "polygon_name_sentence": "Fontvieille is in Monaco.",
            "text": (
                "Fontvieille is in Monaco. "
                "The Coastal park has a forest. "
                "A concert follows."
            ),
            "url": "https://example.test/fontvieille",
            "sentences": [
                "Fontvieille is in Monaco. ",
                "The Coastal park has a forest. ",
                "A concert follows.",
            ],
        },
        {
            "polygon_id": "way/2",
            "polygon_name": "Larvotto",
            "context_phrase": "Monaco",
            "text": ("Larvotto is in Monaco. One. Two. Three. A forest is far away."),
            "url": "https://example.test/larvotto",
            "sentences": [
                "Larvotto is in Monaco. ",
                "One. ",
                "Two. ",
                "Three. ",
                "A forest is far away.",
            ],
        },
        {
            "polygon_id": "way/3",
            "polygon_name": "Monaco-Ville",
            "context_phrase": "Monaco",
            "text": "Monaco-Ville is in Monaco. Parking is nearby.",
            "url": "https://example.test/monaco-ville",
            "sentences": [
                "Monaco-Ville is in Monaco. ",
                "Parking is nearby.",
            ],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _config(tmp_path: Path) -> V9RunConfig:
    return V9RunConfig(
        input_path=tmp_path / "v8.jsonl",
        output_path=tmp_path / "artifacts" / "v9.jsonl",
        manifest_path=tmp_path / "runs" / "v9" / "manifest.json",
        vocabulary_path=tmp_path / "vocabulary.json",
    )


def test_run_v9_keeps_local_topic_sentences_and_selected_v8_fields(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_rows = _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)

    summary = run_v9(config)

    output_rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    relevant = output_rows[0]["sentences_with_topic_term"]
    relevant_metadata = output_rows[0]["relevant_sentence_metadata"]
    removed_v9_fields = {
        "context_fields",
        "context_phrase",
        "country_name_sentence",
        "matched_fields",
        "matched_name",
        "polygon_name_sentence",
    }

    assert summary.rows_processed == 3
    assert summary.rows_kept == 1
    assert summary.rows_filtered == 2
    assert summary.sentences_processed == 10
    assert summary.relevant_sentences_written == 1
    assert summary.category_sentences == {
        "land_use": 1,
        "soil_surface": 1,
        "vegetation_ecosystem": 1,
    }
    assert output_rows[0]["text"] == source_rows[0]["text"]
    assert output_rows[0]["sentences"] == source_rows[0]["sentences"]
    assert removed_v9_fields.isdisjoint(output_rows[0])
    assert "relevant_sentences" not in output_rows[0]
    assert output_rows[0]["topic_sentence_count"] == 1
    assert output_rows[0]["topic_terms"] == ["coastal", "park", "forest"]
    assert output_rows[0]["topic_categories"] == [
        "soil_surface",
        "land_use",
        "vegetation_ecosystem",
    ]
    assert relevant == ["The Coastal park has a forest. "]
    assert relevant_metadata == [
        {
            "country_sentence_distance": 1,
            "place_sentence_distance": 1,
            "polygon_sentence_distance": 1,
            "sentence_index": 1,
            "topic_categories": [
                "soil_surface",
                "land_use",
                "vegetation_ecosystem",
            ],
            "topic_category_count": 3,
            "topic_term_count": 3,
            "topic_terms": ["coastal", "park", "forest"],
        }
    ]
    assert manifest["version"] == "v9"
    assert manifest["source_version"] == "v8"
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == 4
    assert manifest["context_window"] == 2
    assert manifest["matching"] == {
        "place_anchor": "polygon_name in exact normalized sentence text",
        "topic_anchor": "V8 vocabulary in exact normalized sentence text",
        "country_anchor": "context_phrase in exact normalized sentence text",
    }
    assert manifest["source"] == {
        "path": str(config.input_path.resolve()),
        "sha256": v9_module._sha256_file(config.input_path),
    }
    assert manifest["result"]["path"] == str(config.output_path.resolve())
    assert manifest["vocabulary"]["path"] == str(config.vocabulary_path.resolve())
    assert manifest["vocabulary"]["sha256"] == v9_module._sha256_file(
        config.vocabulary_path
    )
    assert (
        manifest["vocabulary"]["definition"]
        == v9_module.load_vocabulary(config.vocabulary_path).to_record()
    )
    assert manifest["sentences_processed"] == 10
    assert manifest["relevant_sentences_written"] == 1
    assert manifest["rows_processed"] == 3
    assert manifest["rows_kept"] == 1
    assert manifest["rows_filtered"] == 2
    assert manifest["category_sentences"] == summary.category_sentences
    assert manifest["result"]["sha256"] == summary.result_sha256


def test_run_v9_filters_topic_sentences_outside_the_local_polygon_window(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)

    summary = run_v9(config)

    assert summary.rows_kept == 1
    assert summary.rows_filtered == 2
    assert "Larvotto" not in config.output_path.read_text(encoding="utf-8")


def test_run_v9_uses_case_insensitive_whole_word_topic_matching(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    rows = _write_v8_input(config.input_path)
    rows[0]["text"] = "Fontvieille is in Monaco. COASTAL PARK."
    rows[0]["sentences"] = ["Fontvieille is in Monaco. ", "COASTAL PARK."]
    config.input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_vocabulary(config.vocabulary_path)

    summary = run_v9(config)

    assert summary.rows_kept == 1
    output = config.output_path.read_text(encoding="utf-8")
    assert '"topic_terms": ["coastal", "park"]' in output


def test_run_v9_reuses_a_matching_manifest_without_loading_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    first = run_v9(config)

    def fail_to_load(_: Path):
        raise AssertionError("vocabulary should not load for a reusable run")

    monkeypatch.setattr(v9_module, "load_vocabulary", fail_to_load)

    second = run_v9(config)

    assert second == first


def test_run_v9_rebuilds_when_the_result_file_is_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    run_v9(config)
    config.output_path.unlink()

    summary = run_v9(config)

    assert summary.rows_kept == 1
    assert config.output_path.is_file()


def test_run_v9_rejects_duplicate_paths_and_invalid_context_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "same.jsonl"
    with pytest.raises(ValueError, match="paths must be different"):
        V9RunConfig(
            input_path=path,
            output_path=path,
            manifest_path=tmp_path / "manifest.json",
            vocabulary_path=tmp_path / "vocabulary.json",
        )
    with pytest.raises(ValueError, match="context_window must not be negative"):
        V9RunConfig(
            input_path=tmp_path / "input.jsonl",
            output_path=tmp_path / "output.jsonl",
            manifest_path=tmp_path / "manifest.json",
            vocabulary_path=tmp_path / "vocabulary.json",
            context_window=-1,
        )


def test_run_v9_reports_missing_vocabulary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)

    with pytest.raises(FileNotFoundError, match=str(config.vocabulary_path.resolve())):
        run_v9(config)


def test_run_v9_reports_missing_input(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_vocabulary(config.vocabulary_path)

    with pytest.raises(FileNotFoundError, match=str(config.input_path.resolve())):
        run_v9(config)


def test_run_v9_rejects_invalid_v8_rows_before_publishing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.input_path.write_text(
        json.dumps(
            {
                "polygon_name": "Fontvieille",
                "context_phrase": "Monaco",
                "text": "Fontvieille is in Monaco.",
                "sentences": ["Changed text."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_vocabulary(config.vocabulary_path)

    with pytest.raises(ValueError, match="must reconstruct the input text"):
        run_v9(config)

    assert not config.output_path.exists()
    assert not config.manifest_path.exists()
    assert not tuple(
        config.output_path.parent.glob(f".{config.output_path.name}.*.tmp")
    )


def test_run_v9_rejects_missing_required_fields(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.input_path.write_text(
        json.dumps(
            {
                "context_phrase": "Monaco",
                "text": "Fontvieille is in Monaco.",
                "sentences": ["Fontvieille is in Monaco."],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_vocabulary(config.vocabulary_path)

    with pytest.raises(ValueError, match="polygon_name"):
        run_v9(config)


def test_v9_private_helpers_enforce_distance_and_anchor_contracts() -> None:
    assert v9_module._within_context_window(None, 2) is False
    assert v9_module._within_context_window(0, 2) is True
    assert v9_module._within_context_window(2, 2) is True
    assert v9_module._within_context_window(3, 2) is False
    assert v9_module._nearest_distance(5, ()) is None
    assert v9_module._nearest_distance(5, (2, 8)) == 3
    assert v9_module._place_distance(2, None) == 2
    assert v9_module._place_distance(2, 5) == 2

    matcher = v9_module._exact_matcher("Old Town")
    assert v9_module._anchor_indices(
        ["old town", "Old%20Town", "new place"], matcher
    ) == (0,)
    with pytest.raises(ValueError) as matcher_error:
        v9_module._exact_matcher("!!!")
    assert str(matcher_error.value) == (
        "place evidence must contain searchable characters"
    )

    with pytest.raises(ValueError) as polygon_error:
        v9_module._place_anchor_indices(
            {"polygon_name": "Missing", "context_phrase": "Monaco"},
            ["Monaco is here."],
        )
    assert str(polygon_error.value) == (
        "polygon_name does not occur in the V8 sentence list"
    )
    with pytest.raises(ValueError) as country_error:
        v9_module._place_anchor_indices(
            {"polygon_name": "Monaco", "context_phrase": "Missing"},
            ["Monaco is here."],
        )
    assert str(country_error.value) == (
        "context_phrase does not occur in the V8 sentence list"
    )


def test_v9_skips_topic_matching_outside_the_polygon_context_window() -> None:
    calls: list[str] = []

    class Vocabulary:
        def match_text(self, sentence: str):
            calls.append(sentence)
            return (SimpleNamespace(category="land_use", term="park"),)

    assert (
        v9_module._sentence_evidence(
            4,
            "A park.",
            polygon_indices=(0,),
            country_indices=(),
            vocabulary=cast(v9_module.TopicVocabulary, Vocabulary()),
            context_window=1,
        )
        is None
    )
    assert calls == []


def test_v9_place_matching_disables_url_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalize_calls: list[object] = []

    def fake_normalize(value: object, *, decode_url: bool = True) -> str:
        normalize_calls.append(decode_url)
        return "old town"

    monkeypatch.setattr(v9_module, "normalize_for_search", fake_normalize)

    v9_module._exact_matcher("Old Town")

    assert normalize_calls == [False]


def test_v9_anchor_matching_disables_url_decoding() -> None:
    decode_calls: list[object] = []

    class FakeMatcher:
        def find(self, value: str, *, decode_url: bool = True) -> frozenset[str]:
            decode_calls.append(decode_url)
            return frozenset({value})

    assert v9_module._anchor_indices(
        ["one", "two"],
        cast(v9_module._MultiPatternMatcher, FakeMatcher()),
    ) == (0, 1)
    assert decode_calls == [False, False]


def test_v9_private_helpers_preserve_evidence_and_count_contracts() -> None:
    matches = (
        SimpleNamespace(category="land_use", term="park"),
        SimpleNamespace(category="land_use", term="garden"),
    )
    evidence = v9_module._sentence_evidence_record(
        4,
        "A park and garden.",
        matches=matches,
        polygon_distance=4,
        country_distance=3,
    )

    assert evidence == {
        "country_sentence_distance": 3,
        "place_sentence_distance": 3,
        "polygon_sentence_distance": 4,
        "sentence": "A park and garden.",
        "sentence_index": 4,
        "topic_categories": ["land_use"],
        "topic_category_count": 1,
        "topic_term_count": 2,
        "topic_terms": ["park", "garden"],
    }
    with pytest.raises(ValueError) as distance_error:
        v9_module._sentence_evidence_record(
            0,
            "park",
            matches=matches,
            polygon_distance=None,
            country_distance=0,
        )
    assert str(distance_error.value) == "polygon evidence distance must be available"
    no_country = v9_module._sentence_evidence_record(
        0,
        "park",
        matches=matches,
        polygon_distance=2,
        country_distance=None,
    )
    assert no_country["country_sentence_distance"] is None
    assert no_country["place_sentence_distance"] == 2

    counts = Counter()
    v9_module._add_category_sentences(
        counts,
        [{"topic_categories": ["land_use", "land_use", "soil_surface"]}],
    )
    assert counts == {"land_use": 2, "soil_surface": 1}
    assert v9_module._string_values(["land_use"]) == ("land_use",)
    with pytest.raises(TypeError) as values_error:
        v9_module._string_values(("land_use",))
    assert str(values_error.value) == "V9 evidence values must be lists of strings"
    assert v9_module._is_string_list(["one", "two"]) is True
    assert v9_module._is_string_list(("one", "two")) is False
    assert v9_module._is_string_list(["one", 2]) is False


def test_run_v9_rebuilds_when_a_manifest_has_a_stale_vocabulary_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    run_v9(config)

    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    manifest["vocabulary"]["sha256"] = v9_module._sha256_file(config.vocabulary_path)
    config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def fail_to_load(_: Path):
        raise AssertionError("a stale vocabulary hash must rebuild")

    monkeypatch.setattr(v9_module, "load_vocabulary", fail_to_load)

    summary = run_v9(config)
    assert summary.rows_kept == 1


def test_run_v9_reports_input_line_numbers_and_writes_nested_output(
    tmp_path: Path,
) -> None:
    config = V9RunConfig(
        input_path=tmp_path / "input.jsonl",
        output_path=tmp_path / "nested" / "artifacts" / "v9.jsonl",
        manifest_path=tmp_path / "nested" / "runs" / "manifest.json",
        vocabulary_path=tmp_path / "vocabulary.json",
    )
    _write_vocabulary(config.vocabulary_path)
    valid = _write_v8_input(config.input_path)[0]
    config.input_path.write_text(json.dumps(valid) + "\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="V8 JSONL line 2 is empty"):
        run_v9(config)
    assert not config.output_path.exists()

    config.input_path.write_text(json.dumps(valid) + "\n", encoding="utf-8")
    summary = run_v9(config)
    assert summary.rows_kept == 1
    assert config.output_path.is_file()


def test_v9_manifest_matching_rejects_each_source_or_setting_change(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    run_v9(config)
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    kwargs = {
        "config": config,
        "input_path": config.input_path.resolve(),
        "output_path": config.output_path.resolve(),
        "vocabulary_path": config.vocabulary_path.resolve(),
        "source_sha256": v9_module._sha256_file(config.input_path),
        "vocabulary_sha256": v9_module._sha256_file(config.vocabulary_path),
    }

    assert v9_module._manifest_matches(manifest, **kwargs) is True
    mutations = (
        ("schema_version", 3),
        ("version", "v8"),
        ("source_version", "v7"),
        ("status", "incomplete"),
        ("context_window", 99),
    )
    for key, value in mutations:
        changed = deepcopy(manifest)
        changed[key] = value
        assert v9_module._manifest_matches(changed, **kwargs) is False

    nested_mutations = (
        ("source", "path", "different-input"),
        ("source", "sha256", "different-source"),
        ("result", "path", "different-output"),
        ("vocabulary", "path", "different-vocabulary"),
        ("vocabulary", "sha256", "different-vocabulary-hash"),
    )
    for section, key, value in nested_mutations:
        changed = deepcopy(manifest)
        changed[section][key] = value
        assert v9_module._manifest_matches(changed, **kwargs) is False
    for section in ("source", "result", "vocabulary"):
        changed = deepcopy(manifest)
        changed[section] = None
        assert v9_module._manifest_matches(changed, **kwargs) is False


def test_v9_manifest_readers_validate_counts_categories_and_json(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("not json", encoding="utf-8")
    assert v9_module._read_manifest(manifest_path) is None
    manifest_path.write_text("[]", encoding="utf-8")
    assert v9_module._read_manifest(manifest_path) is None
    manifest_path.write_text('{"status": "complete"}', encoding="utf-8")
    assert v9_module._read_manifest(manifest_path) == {"status": "complete"}

    assert v9_module._manifest_categories({"category_sentences": {"land_use": 2}}) == {
        "land_use": 2
    }
    assert v9_module._manifest_categories({"category_sentences": []}) is None
    assert (
        v9_module._manifest_categories({"category_sentences": {"land_use": "2"}})
        is None
    )
    assert v9_module._manifest_categories({"category_sentences": {2: 2}}) is None


def test_v9_decode_sentences_reports_invalid_sentence_lists() -> None:
    with pytest.raises(ValueError) as error:
        v9_module._decode_sentences({"sentences": None}, "Fontvieille is in Monaco.", 7)

    assert str(error.value) == (
        "V8 JSONL line 7 must contain a list of string sentences"
    )


def test_v9_decode_input_line_preserves_line_numbers_in_all_errors() -> None:
    with pytest.raises(ValueError) as object_error:
        v9_module._decode_input_line("[]", 7)
    assert str(object_error.value) == "V8 JSONL line 7 must be an object"

    with pytest.raises(ValueError) as text_error:
        v9_module._decode_input_line('{"text": 1, "sentences": []}', 8)
    assert str(text_error.value) == ("V8 JSONL line 8 must contain a string text field")

    with pytest.raises(ValueError) as sentences_error:
        v9_module._decode_input_line(
            '{"text": "hello", "sentences": null}',
            9,
        )
    assert str(sentences_error.value) == (
        "V8 JSONL line 9 must contain a list of string sentences"
    )


def test_v9_writes_kept_rows_with_sorted_json_keys() -> None:
    output = StringIO()

    v9_module._write_kept_row(output, {"z": 1, "a": 2})

    assert output.getvalue() == '{"a": 2, "z": 1}\n'


def test_v9_writes_kept_rows_with_unicode_without_ascii_escaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_dumps(value: object, **kwargs: object) -> str:
        calls.append(kwargs)
        return "payload"

    monkeypatch.setattr(v9_module.json, "dumps", fake_dumps)
    output = StringIO()

    v9_module._write_kept_row(output, {"text": "é"})

    assert output.getvalue() == "payload\n"
    assert calls == [{"ensure_ascii": False, "sort_keys": True}]


def test_v9_atomic_json_output_is_stable_and_human_readable(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    v9_module._atomic_json_write(path, {"z": "é", "a": {"b": 1}})

    assert path.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "b": 1\n  },\n  "z": "é"\n}\n'
    )


def test_v9_output_declares_explicit_unix_newlines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_v8_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    newlines: list[object] = []

    def recording_output(path: Path, **kwargs: Any) -> Any:
        newlines.append(kwargs["newline"])
        return artifact_io.atomic_text_output(path, **kwargs)

    monkeypatch.setattr(v9_module, "_atomic_text_output", recording_output)

    run_v9(config)

    assert newlines == ["\n"]


def test_v9_temporary_paths_are_hidden_sibling_tmp_files(tmp_path: Path) -> None:
    target = tmp_path / "artifacts" / "result.jsonl"
    target.parent.mkdir()

    temporary = artifact_io.temporary_path(target)

    assert temporary.parent == target.parent
    assert temporary.name.startswith(f".{target.name}.")
    assert temporary.name.endswith(".tmp")
    assert not temporary.exists()
