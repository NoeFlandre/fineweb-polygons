from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

import fineweb_polygons.v8 as v8_module
import fineweb_polygons.v9 as v9_module
from fineweb_polygons.v7 import V7RunConfig, run_v7
from fineweb_polygons.v8 import V8RunConfig, run_v8
from fineweb_polygons.v9 import V9RunConfig, run_v9
from fineweb_polygons.v10 import V10RunConfig, run_v10

_V6_ROWS = (
    {
        "context_fields": ["text"],
        "context_phrase": "Monaco",
        "country_name_sentence": "Fontvieille is in Monaco.",
        "fineweb_document_id": "doc-0",
        "fineweb_row_index": 0,
        "matched_fields": ["text", "url"],
        "matched_name": "Fontvieille",
        "name_country_distance": 8,
        "polygon_id": "way/1",
        "polygon_name": "Fontvieille",
        "polygon_name_sentence": "Fontvieille is in Monaco.",
        "text": (
            "Fontvieille is in Monaco. "
            "A coastal park grows here. "
            "Festival events happen."
        ),
        "url": "https://example.test/fontvieille",
    },
    {
        "context_fields": ["text"],
        "context_phrase": "Monaco",
        "country_name_sentence": "Larvotto is in Monaco.",
        "fineweb_document_id": "doc-1",
        "fineweb_row_index": 1,
        "matched_fields": ["text", "url"],
        "matched_name": "Larvotto",
        "name_country_distance": 8,
        "polygon_id": "way/2",
        "polygon_name": "Larvotto",
        "polygon_name_sentence": "Larvotto is in Monaco.",
        "text": "Larvotto is in Monaco. A concert happens here.",
        "url": "https://example.test/larvotto",
    },
    {
        "context_fields": ["text"],
        "context_phrase": "Monaco",
        "country_name_sentence": "Monaco-Ville is in Monaco.",
        "fineweb_document_id": "doc-2",
        "fineweb_row_index": 2,
        "matched_fields": ["text", "url"],
        "matched_name": "Monaco-Ville",
        "name_country_distance": 8,
        "polygon_id": "way/3",
        "polygon_name": "Monaco-Ville",
        "polygon_name_sentence": "Monaco-Ville is in Monaco.",
        "text": "Monaco-Ville is in Monaco. One. Two. A forest grows far away.",
        "url": "https://example.test/monaco-ville",
    },
)
_TEXTS = tuple(row["text"] for row in _V6_ROWS)
_SEGMENTS = {
    _TEXTS[0]: (
        "Fontvieille is in Monaco. ",
        "A coastal park grows here. ",
        "Festival events happen.",
    ),
    _TEXTS[1]: ("Larvotto is in Monaco. ", "A concert happens here."),
    _TEXTS[2]: (
        "Monaco-Ville is in Monaco. ",
        "One. ",
        "Two. ",
        "A forest grows far away.",
    ),
}


class _GoldenSegmenter:
    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        return tuple(_SEGMENTS[text] for text in texts)


class _GoldenClassifier:
    def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
        assert tuple(sentences) == (
            "A coastal park grows here. ",
            "Festival events happen.",
        )
        return ("yes", "no")


class _UnreachableSegmenter:
    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        raise AssertionError("a reusable V7 run must not segment again")


class _UnreachableClassifier:
    def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
        raise AssertionError("a reusable V10 run must not classify again")


@dataclass(frozen=True, slots=True)
class _GoldenPaths:
    v6: Path
    vocabulary: Path
    model: Path
    artifacts: Path
    runs: Path


def _write_v6_input(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in _V6_ROWS),
        encoding="utf-8",
    )


def _write_golden_vocabulary(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "version": "golden-vocabulary-v1",
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
                    "human_activity": ["festival"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _make_paths(tmp_path: Path) -> _GoldenPaths:
    v6_path = tmp_path / "v6.jsonl"
    _write_v6_input(v6_path)
    vocabulary_path = tmp_path / "golden-vocabulary.json"
    _write_golden_vocabulary(vocabulary_path)
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    return _GoldenPaths(
        v6=v6_path,
        vocabulary=vocabulary_path,
        model=model_path,
        artifacts=tmp_path / "artifacts",
        runs=tmp_path / "runs",
    )


def _make_configs(
    paths: _GoldenPaths, *, v7_batch_size: int = 32
) -> tuple[V7RunConfig, V8RunConfig, V9RunConfig, V10RunConfig]:
    v7 = V7RunConfig(
        input_path=paths.v6,
        output_path=paths.artifacts / "golden-v7.jsonl",
        manifest_path=paths.runs / "golden-v7" / "manifest.json",
        batch_size=v7_batch_size,
    )
    v8 = V8RunConfig(
        input_path=v7.output_path,
        output_path=paths.artifacts / "golden-v8.jsonl",
        manifest_path=paths.runs / "golden-v8" / "manifest.json",
        vocabulary_path=paths.vocabulary,
    )
    v9 = V9RunConfig(
        input_path=v8.output_path,
        output_path=paths.artifacts / "golden-v9.jsonl",
        manifest_path=paths.runs / "golden-v9" / "manifest.json",
        vocabulary_path=paths.vocabulary,
    )
    v10 = V10RunConfig(
        input_path=v9.output_path,
        output_path=paths.artifacts / "golden-v10.jsonl",
        manifest_path=paths.runs / "golden-v10" / "manifest.json",
        model_path=paths.model,
        checkpoint_path=paths.runs / "golden-v10" / "checkpoint.jsonl",
    )
    return v7, v8, v9, v10


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_golden_pipeline_produces_the_expected_v10_rows(tmp_path: Path) -> None:
    paths = _make_paths(tmp_path)
    v7_config, v8_config, v9_config, v10_config = _make_configs(paths, v7_batch_size=2)

    v7 = run_v7(
        v7_config,
        segmenter=_GoldenSegmenter(),
    )
    assert v7.rows_processed == 3
    assert v7.sentences_written == 9
    assert _read_rows(v7.output_path)[0]["sentences"] == list(_SEGMENTS[_TEXTS[0]])

    v8 = run_v8(v8_config)
    assert v8.rows_processed == 3
    assert v8.rows_kept == 2
    assert v8.category_documents == {
        "human_activity": 1,
        "land_use": 1,
        "soil_surface": 1,
        "vegetation_ecosystem": 1,
    }
    assert [row["fineweb_document_id"] for row in _read_rows(v8.output_path)] == [
        "doc-0",
        "doc-2",
    ]

    v9 = run_v9(v9_config)
    assert v9.rows_processed == 2
    assert v9.rows_kept == 1
    assert v9.sentences_processed == 7
    assert v9.relevant_sentences_written == 2
    assert v9.category_sentences == {
        "human_activity": 1,
        "land_use": 1,
        "soil_surface": 1,
    }

    v10 = run_v10(
        v10_config,
        classifier=_GoldenClassifier(),
    )
    final_rows = _read_rows(v10.output_path)
    assert v10.rows_processed == 1
    assert v10.rows_kept == 1
    assert v10.candidate_sentences_processed == 2
    assert v10.yes_sentences_written == 1
    assert v10.no_sentences == 1
    assert final_rows == [
        {
            "fineweb_document_id": "doc-0",
            "fineweb_row_index": 0,
            "name_country_distance": 8,
            "polygon_id": "way/1",
            "polygon_name": "Fontvieille",
            "relevant_sentence_metadata": [
                {
                    "country_sentence_distance": 1,
                    "place_sentence_distance": 1,
                    "polygon_sentence_distance": 1,
                    "sentence_index": 1,
                    "topic_categories": ["soil_surface", "land_use"],
                    "topic_category_count": 2,
                    "topic_term_count": 2,
                    "topic_terms": ["coastal", "park"],
                }
            ],
            "sentences_with_topic_term": ["A coastal park grows here. "],
            "topic_categories": ["soil_surface", "land_use"],
            "topic_sentence_count": 1,
            "topic_terms": ["coastal", "park"],
            "url": "https://example.test/fontvieille",
        }
    ]

    stage_files = (
        ("v7", paths.v6, v7.output_path, v7.manifest_path),
        ("v8", v7.output_path, v8.output_path, v8.manifest_path),
        ("v9", v8.output_path, v9.output_path, v9.manifest_path),
        ("v10", v9.output_path, v10.output_path, v10.manifest_path),
    )
    for version, input_path, output_path, manifest_path in stage_files:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["version"] == version
        assert manifest["status"] == "complete"
        assert manifest["source"]["sha256"] == _sha256(input_path)
        assert manifest["result"]["sha256"] == _sha256(output_path)


def test_golden_pipeline_reuses_completed_stages_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_paths(tmp_path)
    v7_config, v8_config, v9_config, v10_config = _make_configs(paths)
    v7 = run_v7(v7_config, segmenter=_GoldenSegmenter())
    v8 = run_v8(v8_config)
    v9 = run_v9(v9_config)
    v10 = run_v10(v10_config, classifier=_GoldenClassifier())

    stage_paths = (
        *(
            path
            for stage in (v7, v8, v9, v10)
            for path in (stage.output_path, stage.manifest_path)
        ),
        v10.checkpoint_path,
    )
    original_bytes = {path: path.read_bytes() for path in stage_paths}

    def fail_to_load(_: Path):
        raise AssertionError("a reusable stage must not load its vocabulary")

    monkeypatch.setattr(v8_module, "load_vocabulary", fail_to_load)
    monkeypatch.setattr(v9_module, "load_vocabulary", fail_to_load)

    assert run_v7(v7_config, segmenter=_UnreachableSegmenter()) == v7
    assert run_v8(v8_config) == v8
    assert run_v9(v9_config) == v9
    assert run_v10(v10_config, classifier=_UnreachableClassifier()) == v10
    assert {path: path.read_bytes() for path in stage_paths} == original_bytes
