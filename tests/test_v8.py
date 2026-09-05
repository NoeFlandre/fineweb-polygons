from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import fineweb_polygons.artifact_io as artifact_io
import fineweb_polygons.v8 as v8_module
from fineweb_polygons.topic_vocabulary import load_vocabulary
from fineweb_polygons.v8 import V8RunConfig, run_v8


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
            "land_use": ["land-use"],
            "vegetation_ecosystem": ["forest"],
        },
    }


def _write_vocabulary(path: Path) -> None:
    path.write_text(
        json.dumps(_vocabulary_mapping(), ensure_ascii=False), encoding="utf-8"
    )


def _write_v7_input(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "polygon_id": "way/1",
            "polygon_name": "Fontvieille",
            "text": "Fontvieille has a forest near the coast.",
            "url": "https://example.test/fontvieille",
            "sentences": ["Fontvieille has a forest near the coast."],
        },
        {
            "polygon_id": "way/2",
            "polygon_name": "Larvotto",
            "text": "Larvotto hosted a concert last summer.",
            "url": "https://example.test/forest-larvotto",
            "sentences": ["Larvotto hosted a concert last summer."],
        },
        {
            "polygon_id": "way/3",
            "polygon_name": "Monaco-Ville",
            "text": "Monaco-Ville uses a different vocabulary.",
            "url": "https://example.test/monaco-ville",
            "sentences": ["Monaco-Ville uses a different vocabulary."],
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _config(tmp_path: Path) -> V8RunConfig:
    return V8RunConfig(
        input_path=tmp_path / "v7.jsonl",
        output_path=tmp_path / "artifacts" / "v8.jsonl",
        manifest_path=tmp_path / "runs" / "v8" / "manifest.json",
        vocabulary_path=tmp_path / "vocabulary.json",
    )


def test_run_v8_keeps_matching_documents_and_preserves_v7_rows(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_rows = _write_v7_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)

    summary = run_v8(config)

    output_rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))

    assert summary.rows_processed == 3
    assert summary.rows_kept == 1
    assert summary.rows_filtered == 2
    assert output_rows == [source_rows[0]]
    assert manifest["version"] == "v8"
    assert manifest["source_version"] == "v7"
    assert manifest["status"] == "complete"
    assert manifest["category_documents"] == {"vegetation_ecosystem": 1}
    assert manifest["vocabulary"]["definition"]["term_count"] == 2
    assert manifest["result"]["sha256"] == summary.result_sha256
    assert manifest["source"]["sha256"] == v8_module._sha256_file(config.input_path)
    assert manifest["vocabulary"]["sha256"] == v8_module._sha256_file(
        config.vocabulary_path
    )


def test_run_v8_allows_an_existing_manifest_parent(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v7_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    config.manifest_path.parent.mkdir(parents=True)

    summary = run_v8(config)

    assert summary.rows_kept == 1


def test_run_v8_reuses_a_matching_manifest_without_loading_vocabulary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_v7_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    first = run_v8(config)

    def fail_to_load(_: Path):
        raise AssertionError("vocabulary should not load for a reusable run")

    monkeypatch.setattr(v8_module, "load_vocabulary", fail_to_load)

    second = run_v8(config)

    assert second == first


def test_run_v8_rebuilds_when_the_result_file_is_missing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v7_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)
    run_v8(config)
    config.output_path.unlink()

    summary = run_v8(config)

    assert summary.rows_kept == 1
    assert config.output_path.is_file()


def test_run_v8_rejects_duplicate_paths(tmp_path: Path) -> None:
    path = tmp_path / "same.jsonl"
    with pytest.raises(ValueError, match="paths must be different"):
        V8RunConfig(
            input_path=path,
            output_path=path,
            manifest_path=tmp_path / "manifest.json",
            vocabulary_path=tmp_path / "vocabulary.json",
        )


def test_run_v8_reports_missing_vocabulary(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v7_input(config.input_path)

    with pytest.raises(FileNotFoundError):
        run_v8(config)


def test_resolve_paths_reports_exact_duplicate_and_missing_paths(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate_config = cast(
        V8RunConfig,
        SimpleNamespace(
            input_path=duplicate,
            output_path=duplicate,
            manifest_path=tmp_path / "manifest.json",
            vocabulary_path=tmp_path / "vocabulary.json",
        ),
    )
    with pytest.raises(ValueError) as duplicate_error:
        v8_module._resolve_paths(duplicate_config)
    assert str(duplicate_error.value) == "V8 paths must be different"

    vocabulary = tmp_path / "vocabulary.json"
    _write_vocabulary(vocabulary)
    missing_input = tmp_path / "missing-input.jsonl"
    missing_input_config = cast(
        V8RunConfig,
        SimpleNamespace(
            input_path=missing_input,
            output_path=tmp_path / "output.jsonl",
            manifest_path=tmp_path / "manifest.json",
            vocabulary_path=vocabulary,
        ),
    )
    with pytest.raises(FileNotFoundError) as input_error:
        v8_module._resolve_paths(missing_input_config)
    assert input_error.value.args == (missing_input.resolve(),)

    input_path = tmp_path / "input.jsonl"
    input_path.write_text('{"text": "forest"}\n', encoding="utf-8")
    missing_vocabulary = tmp_path / "missing-vocabulary.json"
    missing_vocabulary_config = cast(
        V8RunConfig,
        SimpleNamespace(
            input_path=input_path,
            output_path=tmp_path / "output.jsonl",
            manifest_path=tmp_path / "manifest.json",
            vocabulary_path=missing_vocabulary,
        ),
    )
    with pytest.raises(FileNotFoundError) as vocabulary_error:
        v8_module._resolve_paths(missing_vocabulary_config)
    assert vocabulary_error.value.args == (missing_vocabulary.resolve(),)


def test_v8_text_row_decoder_preserves_text_and_error_contracts() -> None:
    row = {"text": "Héllo"}

    assert v8_module._decode_text_row(row, 3) == (row, "Héllo")

    with pytest.raises(ValueError) as error:
        v8_module._decode_text_row({"text": 3}, 4)
    assert str(error.value) == ("V7 JSONL line 4 must contain a string text field")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("", "is empty"),
        ("[]", "must be an object"),
        ('{"text": 3}', "must contain a string text field"),
    ],
)
def test_v8_read_rows_preserves_invalid_row_line_numbers(
    tmp_path: Path, line: str, message: str
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text": "ok"}\n' + line + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        list(v8_module._read_rows(path))
    assert str(error.value) == f"V7 JSONL line 2 {message}"


def test_category_counts_include_each_matching_category_once(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = _write_v7_input(config.input_path)
    rows[0]["text"] = "A forest with land-use planning."
    config.input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_vocabulary(config.vocabulary_path)

    summary = run_v8(config, vocabulary=load_vocabulary(config.vocabulary_path))
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))

    assert summary.category_documents == {
        "land_use": 1,
        "vegetation_ecosystem": 1,
    }
    assert manifest["category_documents"] == summary.category_documents


def test_manifest_count_helpers_reject_invalid_shapes() -> None:
    assert v8_module._manifest_counts({}) is None
    assert (
        v8_module._manifest_counts(
            {
                "rows_processed": 1,
                "rows_kept": 1,
                "rows_filtered": 0,
                "category_documents": None,
            }
        )
        is None
    )
    assert (
        v8_module._manifest_category_documents(
            {"category_documents": {"land_use": "one"}}
        )
        is None
    )
    assert (
        v8_module._manifest_category_documents({"category_documents": {1: 1}}) is None
    )


def test_manifest_count_helpers_accept_valid_values() -> None:
    manifest = {
        "rows_processed": 4,
        "rows_kept": 2,
        "rows_filtered": 2,
        "category_documents": {"land_use": 1},
    }

    assert v8_module._manifest_row_counts(manifest) == (4, 2, 2)
    assert v8_module._manifest_category_documents(manifest) == {"land_use": 1}
    assert v8_module._manifest_counts(manifest) == (4, 2, 2, {"land_use": 1})

    for key in ("rows_processed", "rows_kept", "rows_filtered"):
        invalid: dict[str, object] = dict(manifest)
        invalid[key] = "not an integer"
        assert v8_module._manifest_row_counts(invalid) is None


def test_write_output_creates_nested_parent_and_reuses_existing_parent(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.jsonl"
    vocabulary_path = tmp_path / "vocabulary.json"
    _write_v7_input(input_path)
    _write_vocabulary(vocabulary_path)
    vocabulary = load_vocabulary(vocabulary_path)

    nested_output = tmp_path / "new" / "nested" / "v8.jsonl"
    v8_module._write_output(
        input_path=input_path,
        output_path=nested_output,
        vocabulary=vocabulary,
    )
    assert nested_output.is_file()

    existing_output = nested_output.parent / "second.jsonl"
    v8_module._write_output(
        input_path=input_path,
        output_path=existing_output,
        vocabulary=vocabulary,
    )
    assert existing_output.is_file()


def test_write_output_uses_utf8_newline_and_counts_multiple_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    rows = _write_v7_input(config.input_path)
    rows[1]["text"] = "Larvotto has a forest."
    config.input_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_vocabulary(config.vocabulary_path)
    real_open = Path.open
    captured: list[dict[str, object]] = []

    def recording_open(self: Path, *args: Any, **kwargs: Any):
        if args and args[0] == "w":
            captured.append(dict(kwargs))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    summary = run_v8(config)

    assert summary.rows_kept == 2
    assert summary.category_documents == {"vegetation_ecosystem": 2}
    assert captured == [{"encoding": "utf-8", "newline": "\n"}]


def test_matching_categories_preserves_first_match_order() -> None:
    class Category(str):
        def __hash__(self) -> int:
            return 1 if self == "zulu" else 0

    class FakeVocabulary:
        def match_text(self, text: str) -> tuple[SimpleNamespace, ...]:
            del text
            return (
                SimpleNamespace(category=Category("zulu")),
                SimpleNamespace(category=Category("alpha")),
                SimpleNamespace(category=Category("zulu")),
            )

    assert v8_module._matching_categories(
        cast(v8_module.TopicVocabulary, FakeVocabulary()), "text"
    ) == (
        "zulu",
        "alpha",
    )


def test_read_rows_uses_utf8_and_reports_original_line_number(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"text": "forest"}\n\n', encoding="utf-8")
    real_open = Path.open
    captured: list[object] = []

    def recording_open(self: Path, *args: Any, **kwargs: Any):
        if not args:
            captured.append(kwargs.get("encoding"))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    with pytest.raises(ValueError) as error:
        list(v8_module._read_rows(path))

    assert str(error.value) == "V7 JSONL line 2 is empty"
    assert captured == ["utf-8"]


def test_manifest_matches_rejects_non_mapping_sections(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    vocabulary_path = tmp_path / "vocabulary.json"
    cases = [
        {"source": None, "result": {}, "vocabulary": {}},
        {"source": {}, "result": None, "vocabulary": {}},
        {"source": {}, "result": {}, "vocabulary": None},
    ]

    for manifest in cases:
        assert (
            v8_module._manifest_matches(
                manifest,
                input_path,
                output_path,
                vocabulary_path,
                "source",
                "vocabulary",
            )
            is False
        )


def test_atomic_writes_clean_up_a_missing_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_after_removing_temporary(temporary: Path, output: Path) -> None:
        temporary.unlink()
        raise OSError("replace failed")

    monkeypatch.setattr(artifact_io.os, "replace", fail_after_removing_temporary)

    with pytest.raises(OSError, match="replace failed"):
        v8_module._atomic_json_write(tmp_path / "manifest.json", {"status": "complete"})


def test_write_output_cleans_up_a_missing_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_v7_input(config.input_path)
    _write_vocabulary(config.vocabulary_path)

    def fail_after_removing_temporary(temporary: Path, output: Path) -> None:
        temporary.unlink()
        raise OSError("replace failed")

    monkeypatch.setattr(artifact_io.os, "replace", fail_after_removing_temporary)

    with pytest.raises(OSError, match="replace failed"):
        v8_module._write_output(
            input_path=config.input_path,
            output_path=config.output_path,
            vocabulary=load_vocabulary(config.vocabulary_path),
        )
