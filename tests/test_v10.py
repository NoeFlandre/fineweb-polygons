from __future__ import annotations

import json
from collections.abc import Sequence
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest

import fineweb_polygons.v10 as v10_module
import fineweb_polygons.v10_inference as inference_module
from fineweb_polygons.v10 import V10RunConfig, run_v10
from fineweb_polygons.v10_inference import (
    LfmSentenceClassifier,
    MlxSentenceClassifier,
    format_prompt,
    parse_label,
    render_chat_prompt,
    render_mlx_prompt,
)

PROMPT = (
    "Classify whether the TARGET SENTENCE contains information about the target "
    "place that could help characterize its land use, land cover, or geographic "
    "environment from remote sensing, either directly or through observable "
    "proxies.\n"
    "Return exactly one token: yes or no.\n"
    "Answer **yes** for information about vegetation, agriculture, forests, "
    "water, soil or surface, terrain, buildings, settlements, infrastructure, "
    "transport networks, mining, managed land, or other human or natural "
    "features with a spatial or remotely detectable signature.\n"
    "Answer **no** for information only about history, administration, people, "
    "events, demographics, economy, navigation, or activities with no meaningful "
    "land-use, land-cover, or remotely detectable implication.\n"
    "Output only the lowercase token yes or no.\n"
    "TARGET SENTENCE: Target {place}"
)


def _write_v9_input(path: Path) -> None:
    rows = [
        {
            "fineweb_document_id": "doc-1",
            "fineweb_row_index": 1,
            "name_country_distance": 12,
            "polygon_id": "way/1",
            "polygon_name": "Fontvieille",
            "relevant_sentence_metadata": [
                {
                    "sentence_index": 0,
                    "topic_categories": ["land_use"],
                    "topic_terms": ["park"],
                },
                {"sentence_index": 1, "topic_terms": ["event"]},
            ],
            "sentences": [
                "Fontvieille has a park. ",
                "A concert happens there.",
            ],
            "sentences_with_topic_term": [
                "Fontvieille has a park. ",
                "A concert happens there.",
            ],
            "text": "Fontvieille has a park. A concert happens there.",
            "topic_categories": ["land_use", "events"],
            "topic_sentence_count": 2,
            "topic_terms": ["park", "event"],
            "url": "https://example.test/fontvieille",
        },
        {
            "fineweb_document_id": "doc-2",
            "fineweb_row_index": 2,
            "name_country_distance": 10,
            "polygon_id": "way/2",
            "polygon_name": "Larvotto",
            "relevant_sentence_metadata": [
                {"sentence_index": 0, "topic_terms": ["history"]},
            ],
            "sentences": ["Larvotto has a history."],
            "sentences_with_topic_term": ["Larvotto has a history."],
            "text": "Larvotto has a history.",
            "topic_categories": ["history"],
            "topic_sentence_count": 1,
            "topic_terms": ["history"],
            "url": "https://example.test/larvotto",
        },
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _config(tmp_path: Path, *, batch_size: int = 8) -> V10RunConfig:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    return V10RunConfig(
        input_path=tmp_path / "v9.jsonl",
        output_path=tmp_path / "artifacts" / "v10.jsonl",
        manifest_path=tmp_path / "runs" / "v10" / "manifest.json",
        model_path=model_path,
        checkpoint_path=tmp_path / "runs" / "v10" / "checkpoint.jsonl",
        batch_size=batch_size,
    )


class _FakeClassifier:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.calls: list[tuple[str, ...]] = []

    def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
        batch = tuple(sentences)
        self.calls.append(batch)
        result = tuple(self.labels[: len(batch)])
        self.labels = self.labels[len(batch) :]
        return result


def test_v10_prompt_is_exact_and_inserts_the_target_sentence() -> None:
    assert format_prompt("Target {place}") == PROMPT


def test_v10_renders_the_exact_prompt_through_the_model_chat_template() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert kwargs == {"add_generation_prompt": True, "tokenize": False}
            return "rendered prompt"

    assert render_chat_prompt(FakeTokenizer(), "prompt") == "rendered prompt</think>"


def test_v10_renders_an_mlx_chat_prompt_as_token_ids() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert kwargs == {"add_generation_prompt": True, "return_dict": False}
            return [1, 2, 3]

        def encode(self, text):
            assert text == "</think>"
            return [4]

    assert render_mlx_prompt(FakeTokenizer(), "prompt") == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "raw_output, expected", [("<think>reason</think>yes", "yes"), (" no ", "no")]
)
def test_v10_accepts_the_final_label_after_optional_model_reasoning(
    raw_output: str, expected: str
) -> None:
    assert parse_label(raw_output) == expected


def test_run_v10_publishes_only_yes_sentences_and_aligned_metadata(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v9_input(config.input_path)
    classifier = _FakeClassifier(["yes", "no", "no"])

    summary = run_v10(config, classifier=classifier)

    rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert summary.rows_processed == 2
    assert summary.rows_kept == 1
    assert summary.rows_filtered == 1
    assert summary.candidate_sentences_processed == 3
    assert summary.yes_sentences_written == 1
    assert summary.no_sentences == 2
    assert rows[0]["sentences_with_topic_term"] == ["Fontvieille has a park. "]
    assert rows[0]["relevant_sentence_metadata"] == [
        {
            "sentence_index": 0,
            "topic_categories": ["land_use"],
            "topic_terms": ["park"],
        }
    ]
    assert rows[0]["topic_categories"] == ["land_use"]
    assert rows[0]["topic_sentence_count"] == 1
    assert rows[0]["topic_terms"] == ["park"]
    assert "text" not in rows[0]
    assert "sentences" not in rows[0]
    assert manifest["version"] == "v10"
    assert manifest["source_version"] == "v9"
    assert manifest["status"] == "complete"
    assert manifest["candidate_sentences_processed"] == 3
    assert manifest["yes_sentences_written"] == 1
    assert manifest["no_sentences"] == 2
    assert manifest["source"]["path"] == str(config.input_path.resolve())
    assert manifest["source"]["sha256"] == v10_module._sha256_file(config.input_path)
    assert manifest["checkpoint"]["path"] == str(
        config.effective_checkpoint_path.resolve()
    )
    assert manifest["checkpoint"]["sha256"] == v10_module._sha256_file(
        config.effective_checkpoint_path
    )
    assert manifest["classification"]["discarded_label"] == "no"
    assert manifest["classification"]["label_contract"] == ("exact lowercase yes or no")
    assert manifest["classification"]["prompt_template"] == (
        inference_module.V10_PROMPT_TEMPLATE
    )
    assert manifest["classification"]["prompt_sha256"] == v10_module.PROMPT_SHA256
    assert manifest["result"]["sha256"] == summary.result_sha256

    checkpoint_header = json.loads(
        config.effective_checkpoint_path.read_text(encoding="utf-8").splitlines()[0]
    )
    assert checkpoint_header["source_version"] == "v9"
    assert checkpoint_header["source"]["path"] == str(config.input_path.resolve())
    assert checkpoint_header["source"]["sha256"] == manifest["source"]["sha256"]
    assert checkpoint_header["classification"]["assistant_prefill"] == (
        inference_module.REASONING_CLOSE_TAG
    )


def test_run_v10_rejects_any_classifier_output_other_than_yes_or_no(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v9_input(config.input_path)

    with pytest.raises(ValueError, match="yes or no"):
        run_v10(config, classifier=_FakeClassifier(["maybe"]))

    assert not config.output_path.exists()
    assert not config.manifest_path.exists()


def test_run_v10_resumes_completed_checkpoint_batches(tmp_path: Path) -> None:
    config = _config(tmp_path, batch_size=2)
    _write_v9_input(config.input_path)

    class FailingClassifier(_FakeClassifier):
        def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
            if self.calls:
                raise RuntimeError("stop after one checkpoint")
            return super().classify(sentences)

    with pytest.raises(RuntimeError, match="stop after one checkpoint"):
        run_v10(config, classifier=FailingClassifier(["yes", "no"]))

    resumed = _FakeClassifier(["no"])
    summary = run_v10(config, classifier=resumed)

    assert resumed.calls == [("Larvotto has a history.",)]
    assert summary.rows_kept == 1
    assert config.checkpoint_path is not None
    checkpoint_lines = config.checkpoint_path.read_text(encoding="utf-8").splitlines()
    assert len(checkpoint_lines) == 3


def test_run_v10_reuses_a_matching_manifest_without_classifying_again(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v9_input(config.input_path)
    first = run_v10(config, classifier=_FakeClassifier(["yes", "no", "no"]))

    class UnexpectedClassifier:
        def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
            raise AssertionError("classifier should not be loaded for a reusable run")

    second = run_v10(config, classifier=UnexpectedClassifier())

    assert second == first


def test_v10_config_exposes_explicit_runtime_and_validates_settings(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    configured = V10RunConfig(
        input_path=config.input_path,
        output_path=config.output_path,
        manifest_path=config.manifest_path,
        model_path=config.model_path,
        checkpoint_path=config.checkpoint_path,
        runtime_model_path=runtime,
    )

    assert configured.effective_checkpoint_path == config.checkpoint_path
    assert configured.effective_runtime_model_path == runtime
    without_checkpoint = V10RunConfig(
        input_path=config.input_path,
        output_path=config.output_path,
        manifest_path=config.manifest_path,
        model_path=config.model_path,
    )
    assert without_checkpoint.effective_checkpoint_path == (
        config.manifest_path.with_name("classifications.jsonl")
    )
    with pytest.raises(ValueError, match="batch_size must be positive"):
        V10RunConfig(
            input_path=config.input_path,
            output_path=config.output_path,
            manifest_path=config.manifest_path,
            model_path=config.model_path,
            batch_size=0,
        )
    with pytest.raises(ValueError, match="max_new_tokens must be positive"):
        V10RunConfig(
            input_path=config.input_path,
            output_path=config.output_path,
            manifest_path=config.manifest_path,
            model_path=config.model_path,
            max_new_tokens=0,
        )

    assert (
        V10RunConfig(
            input_path=config.input_path,
            output_path=config.output_path,
            manifest_path=config.manifest_path,
            model_path=config.model_path,
            batch_size=1,
            max_new_tokens=1,
        ).batch_size
        == 1
    )


def test_v10_path_resolution_reports_missing_inputs_and_duplicate_runtime(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"v9\.jsonl"):
        v10_module._resolve_paths(config)

    config.input_path.write_text("{}\n", encoding="utf-8")
    (config.model_path / "config.json").unlink()
    config.model_path.rmdir()
    with pytest.raises(FileNotFoundError, match="model"):
        v10_module._resolve_paths(config)

    config.model_path.mkdir()
    runtime = tmp_path / "missing-runtime"
    explicit = SimpleNamespace(
        input_path=config.input_path,
        output_path=config.output_path,
        manifest_path=config.manifest_path,
        model_path=config.model_path,
        runtime_model_path=runtime,
        effective_checkpoint_path=config.effective_checkpoint_path,
        effective_runtime_model_path=config.model_path,
    )
    with pytest.raises(ValueError, match="paths must be different"):
        v10_module._resolve_paths(cast(V10RunConfig, explicit))

    explicit.effective_runtime_model_path = runtime
    with pytest.raises(FileNotFoundError, match="missing-runtime"):
        v10_module._resolve_paths(cast(V10RunConfig, explicit))


def test_v10_resolved_paths_are_absolute_and_ordered(tmp_path: Path) -> None:
    config = _config(tmp_path)

    assert v10_module._resolved_paths(config) == (
        config.input_path.resolve(),
        config.output_path.resolve(),
        config.manifest_path.resolve(),
        config.model_path.resolve(),
        config.effective_checkpoint_path.resolve(),
        config.effective_runtime_model_path.resolve(),
    )


def test_v10_distinct_path_validation_checks_the_checkpoint_and_exact_message(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / name for name in ("input", "output", "manifest", "model"))
    duplicate_paths = (
        paths[0],
        paths[1],
        paths[2],
        paths[3],
        paths[0],
        tmp_path / "runtime",
    )

    with pytest.raises(ValueError) as error:
        v10_module._validate_distinct_paths(duplicate_paths, runtime_is_explicit=False)

    assert str(error.value) == "V10 paths must be different"


def test_v10_builds_the_native_or_mlx_classifier_from_runtime_configuration(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    calls = []

    class Native:
        def __init__(self, path, *, max_new_tokens):
            calls.append(("native", path, max_new_tokens))

    class Mlx:
        def __init__(self, path, *, max_new_tokens):
            calls.append(("mlx", path, max_new_tokens))

    monkeypatch.setattr(v10_module, "LfmSentenceClassifier", Native)
    monkeypatch.setattr(v10_module, "MlxSentenceClassifier", Mlx)
    v10_module._build_classifier(config, config.model_path)
    explicit = V10RunConfig(
        input_path=config.input_path,
        output_path=config.output_path,
        manifest_path=config.manifest_path,
        model_path=config.model_path,
        runtime_model_path=runtime,
    )
    v10_module._build_classifier(explicit, runtime)

    assert calls == [
        ("native", config.model_path, 512),
        ("mlx", runtime, 512),
    ]


def test_run_v10_builds_a_classifier_when_none_is_injected(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    _write_v9_input(config.input_path)
    calls = []

    def build_classifier(received_config, runtime_path):
        calls.append((received_config, runtime_path))
        return _FakeClassifier(["no", "no", "no"])

    monkeypatch.setattr(v10_module, "_build_classifier", build_classifier)

    run_v10(config)

    assert calls == [(config, config.model_path.resolve())]


def test_v10_model_record_contains_file_and_snapshot_fingerprints(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = model_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    (model_path / ".incomplete-config").write_text("ignore", encoding="utf-8")

    record = v10_module._model_record(model_path)

    assert record["snapshot_id"] == "model"
    assert record["path"] == str(model_path)
    assert "sha256" in record
    assert isinstance(record["sha256"], str)
    assert len(record["sha256"]) == 64
    assert record["files"] == [
        {
            "path": "config.json",
            "sha256": v10_module._sha256_file(config_path),
        }
    ]


def test_v10_checkpoint_header_contains_the_reproducibility_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    header = v10_module._checkpoint_header(
        config=config,
        input_path=config.input_path,
        source_sha256="source-sha",
        model_record={"model": "source"},
        runtime_model_record={"model": "runtime"},
    )

    assert header == {
        "record_type": "header",
        "schema_version": v10_module.V10_SCHEMA_VERSION,
        "version": v10_module.V10_VERSION,
        "source_version": v10_module.V10_SOURCE_VERSION,
        "source": {
            "path": str(config.input_path),
            "sha256": "source-sha",
        },
        "model": {"model": "source"},
        "runtime_model": {"model": "runtime"},
        "classification": {
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "prompt_sha256": v10_module.PROMPT_SHA256,
            "assistant_prefill": inference_module.REASONING_CLOSE_TAG,
        },
    }


def test_v10_checkpoint_parser_skips_blank_and_truncated_final_lines() -> None:
    records = v10_module._checkpoint_records(
        [
            "",
            json.dumps({"labels": ["yes"], "row_number": 1}),
            '{"row_number": 2',
        ]
    )

    assert records == {1: ("yes",)}


def test_v10_open_checkpoint_writes_sorted_utf8_headers_and_reads_utf8(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    expected_header = {"z": "é", "a": 1}
    write_encoding = []
    open_encoding = []
    original_write_text = Path.write_text
    original_open = Path.open

    def write_text(path, data, *args, **kwargs):
        write_encoding.append(kwargs.get("encoding"))
        return original_write_text(path, data, *args, **kwargs)

    def open_path(path, *args, **kwargs):
        open_encoding.append(kwargs.get("encoding"))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "open", open_path)

    assert v10_module._open_checkpoint(checkpoint, expected_header) == {}
    assert v10_module._open_checkpoint(checkpoint, expected_header) == {}
    assert write_encoding == ["utf-8"]
    assert open_encoding == ["utf-8", "utf-8"]
    assert checkpoint.read_text(encoding="utf-8") == (
        json.dumps(expected_header, ensure_ascii=False, sort_keys=True) + "\n"
    )


@pytest.mark.parametrize(
    "line, message",
    [
        ("[]", "must be an object"),
        (json.dumps({"labels": [], "row_number": "1"}), "is invalid"),
        (json.dumps({"labels": ["maybe"], "row_number": 1}), "yes or no"),
    ],
)
def test_v10_checkpoint_parser_rejects_invalid_records(line: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        v10_module._checkpoint_records([line])


def test_v10_checkpoint_parser_rejects_non_final_bad_and_duplicate_records() -> None:
    with pytest.raises(ValueError, match="line 2 is invalid"):
        v10_module._checkpoint_records(['{"row_number": 1', "{}"])
    duplicate = json.dumps({"labels": ["no"], "row_number": 1})
    with pytest.raises(ValueError, match="repeats row 1"):
        v10_module._checkpoint_records([duplicate, duplicate])

    with pytest.raises(ValueError) as error:
        v10_module._parse_checkpoint_record("[]", 2, 3)
    assert str(error.value) == "V10 checkpoint line 2 must be an object"


def test_v10_checkpoint_open_rejects_a_mismatched_header(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError) as error:
        v10_module._open_checkpoint(checkpoint, {"expected": True})
    assert str(error.value) == "V10 checkpoint does not match the current run"


def test_v10_decode_and_metadata_validation_reject_bad_input() -> None:
    cases = [
        ("", "is empty"),
        ("[]", "must be an object"),
        (json.dumps({"sentences_with_topic_term": "sentence"}), "candidate sentences"),
        (
            json.dumps(
                {
                    "sentences_with_topic_term": ["sentence"],
                    "relevant_sentence_metadata": [],
                }
            ),
            "metadata must align",
        ),
    ]
    for line, message in cases:
        with pytest.raises(ValueError, match=message):
            v10_module._decode_input_line(line, 4)

    with pytest.raises(ValueError, match="topic_terms"):
        v10_module._metadata_values([{"topic_terms": "park"}], "topic_terms")
    assert v10_module._metadata_values([{"topic_terms": ["park"]}], "missing") == ()

    with pytest.raises(ValueError) as error:
        v10_module._decode_input_line("[]", 4)
    assert str(error.value) == "V9 JSONL line 4 must be an object"
    with pytest.raises(ValueError) as error:
        v10_module._decode_input_line(
            json.dumps({"sentences_with_topic_term": "sentence"}), 4
        )
    assert str(error.value) == (
        "V9 JSONL line 4 must contain a list of candidate sentences"
    )
    assert v10_module._is_mapping_list({}) is False

    with pytest.raises(ValueError) as error:
        v10_module._validate_labels(["maybe"])
    assert str(error.value) == "Classifier labels must be exactly yes or no"


def test_v10_reusable_summary_rejects_bad_counts_and_result_hash(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v9_input(config.input_path)
    run_v10(config, classifier=_FakeClassifier(["yes", "no", "no"]))
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    output_text = config.output_path.read_text(encoding="utf-8")
    config.output_path.unlink()
    assert (
        v10_module._reusable_manifest(
            config.manifest_path,
            config=config,
            input_path=config.input_path,
            output_path=config.output_path,
            checkpoint_path=config.effective_checkpoint_path,
            source_sha256=v10_module._sha256_file(config.input_path),
            model_record=v10_module._model_record(config.model_path),
            runtime_model_record=v10_module._model_record(config.model_path),
        )
        is None
    )
    config.output_path.write_text(output_text, encoding="utf-8")

    manifest["rows_processed"] = "two"
    config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        v10_module._load_reusable_summary(
            config=config,
            input_path=config.input_path,
            output_path=config.output_path,
            manifest_path=config.manifest_path,
            checkpoint_path=config.effective_checkpoint_path,
            source_sha256=v10_module._sha256_file(config.input_path),
            model_record=v10_module._model_record(config.model_path),
            runtime_model_record=v10_module._model_record(config.model_path),
        )
        is None
    )

    manifest["rows_processed"] = 2
    manifest["result"]["sha256"] = "wrong"
    config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert (
        v10_module._load_reusable_summary(
            config=config,
            input_path=config.input_path,
            output_path=config.output_path,
            manifest_path=config.manifest_path,
            checkpoint_path=config.effective_checkpoint_path,
            source_sha256=v10_module._sha256_file(config.input_path),
            model_record=v10_module._model_record(config.model_path),
            runtime_model_record=v10_module._model_record(config.model_path),
        )
        is None
    )


def test_v10_manifest_helpers_reject_malformed_records(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model_record = v10_module._model_record(config.model_path)
    assert not v10_module._manifest_matches(
        {},
        config=config,
        input_path=config.input_path,
        output_path=config.output_path,
        checkpoint_path=config.effective_checkpoint_path,
        source_sha256="source",
        model_record=model_record,
        runtime_model_record=model_record,
    )
    assert v10_module._summary_counts({}) is None
    assert v10_module._summary_counts(
        {
            "rows_processed": 1,
            "rows_kept": 2,
            "rows_filtered": 3,
            "candidate_sentences_processed": 4,
            "yes_sentences_written": 5,
            "no_sentences": 6,
        }
    ) == (1, 2, 3, 4, 5, 6)
    records = v10_module._manifest_records({"source": True}, {"result": True})
    assert records == ({"source": True}, {"result": True})
    with pytest.raises(ValueError) as error:
        v10_module._nested_string({}, "result", "sha256")
    assert str(error.value) == "V10 manifest has an invalid result record"


def test_v10_chat_rendering_supports_fallback_and_string_mlx_templates() -> None:
    assert render_chat_prompt(object(), "prompt") == "prompt"

    class StringTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "rendered"

        def encode(self, text):
            if text == "</think>":
                return [3]
            assert text == "rendered</think>"
            return [7, 8]

    assert render_mlx_prompt(StringTokenizer(), "prompt") == [7, 8]


@pytest.mark.parametrize(
    "tokenizer",
    [
        type(
            "BadListTokenizer",
            (),
            {
                "apply_chat_template": lambda self, *a, **k: ["x"],
                "encode": lambda self, text: [1],
            },
        ),
        type(
            "BadStringTokenizer",
            (),
            {
                "apply_chat_template": lambda self, *a, **k: "x",
                "encode": lambda self, text: ["x"],
            },
        ),
    ],
)
def test_v10_mlx_rendering_rejects_non_integer_tokens(tokenizer) -> None:
    with pytest.raises(TypeError, match="token IDs"):
        render_mlx_prompt(tokenizer(), "prompt")


@pytest.mark.parametrize("raw_output", ["YES", "<think>reason</think>maybe"])
def test_v10_label_parser_rejects_non_contract_answers(raw_output: str) -> None:
    with pytest.raises(ValueError, match="exactly yes or no"):
        parse_label(raw_output)


def test_v10_label_parser_uses_the_last_reasoning_close_tag() -> None:
    assert parse_label("<think>first</think><think>second</think>yes") == "yes"


@pytest.mark.parametrize("label", ["yes", "no"])
def test_v10_label_parser_accepts_exact_contract_labels(label: str) -> None:
    assert parse_label(label) == label


class _FakeTorch:
    class backends:
        class mps:
            available = True

            @classmethod
            def is_available(cls):
                return cls.available

    @staticmethod
    def inference_mode():
        class Context:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

        return Context()


class _FakeEncoded(dict):
    def to(self, device):
        self["device"] = device
        return self


class _FakeTensor:
    shape = (2, 4)

    def __getitem__(self, item):
        return item


class _FakeNativeTokenizer:
    pad_token_id = None
    eos_token = "eos"
    decoded_outputs: ClassVar[list[str]] = ["yes", "<think>reason</think>no"]
    pretrained_args: ClassVar[list[tuple[object, dict[str, object]]]] = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.pretrained_args.append((path, dict(kwargs)))
        return cls()

    def apply_chat_template(self, messages, **kwargs):
        return "rendered"

    def __call__(self, prompts, **kwargs):
        return _FakeEncoded(input_ids=_FakeTensor())

    def batch_decode(self, generated, **kwargs):
        return self.decoded_outputs


class _FakeNativeModel:
    pretrained_args: ClassVar[list[tuple[object, dict[str, object]]]] = []

    def __init__(self) -> None:
        self.device = None
        self.generate_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        instance = cls()
        cls.pretrained_args.append((path, dict(kwargs)))
        return instance

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.evaluated = True

    def generate(self, **kwargs):
        self.generate_kwargs = dict(kwargs)
        return _FakeTensor()


def test_v10_native_classifier_uses_chat_template_and_greedy_generation(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(
        inference_module,
        "_load_transformers",
        lambda: (_FakeTorch, _FakeNativeTokenizer, _FakeNativeModel),
    )

    _FakeNativeTokenizer.pretrained_args = []
    _FakeNativeModel.pretrained_args = []
    classifier = LfmSentenceClassifier(model_path, max_new_tokens=1)
    assert classifier.classify(["first", "second"]) == ("yes", "no")
    assert classifier.classify([]) == ()
    assert _FakeNativeTokenizer.pretrained_args == [
        (str(model_path), {"local_files_only": True})
    ]
    assert _FakeNativeModel.pretrained_args == [
        (
            str(model_path),
            {"local_files_only": True, "low_cpu_mem_usage": True},
        )
    ]
    assert classifier._model.device == "mps"
    assert classifier._model.generate_kwargs["max_new_tokens"] == 1
    assert classifier._model.generate_kwargs["do_sample"] is False


def test_v10_native_classifier_rejects_wrong_label_count(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(
        inference_module,
        "_load_transformers",
        lambda: (_FakeTorch, _FakeNativeTokenizer, _FakeNativeModel),
    )
    classifier = LfmSentenceClassifier(model_path)
    classifier._tokenizer.decoded_outputs = ["yes"]

    with pytest.raises(RuntimeError, match="wrong number"):
        classifier.classify(["first", "second"])


def test_v10_mlx_classifier_batches_and_validates_labels(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]

        def encode(self, text):
            return [2]

    class Result:
        texts = ("yes", "no")

    load_calls = []

    def load(path):
        load_calls.append(path)
        return "model", Tokenizer()

    def batch_generate(model, tokenizer, prompts, **kwargs):
        calls.append((model, tokenizer, prompts, kwargs))
        return Result()

    monkeypatch.setattr(inference_module, "_load_mlx", lambda: (load, batch_generate))
    classifier = MlxSentenceClassifier(model_path, max_new_tokens=11)

    assert classifier.classify(["first", "second"]) == ("yes", "no")
    assert classifier.classify([]) == ()
    assert load_calls == [str(model_path)]
    assert calls[0][3] == {"max_tokens": 11, "verbose": False}


def test_v10_mlx_classifier_rejects_wrong_label_count(
    tmp_path: Path, monkeypatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1]

        def encode(self, text):
            return [2]

    class Result:
        texts = ("yes",)

    monkeypatch.setattr(
        inference_module,
        "_load_mlx",
        lambda: (lambda path: ("model", Tokenizer()), lambda *args, **kwargs: Result()),
    )
    classifier = MlxSentenceClassifier(model_path)

    with pytest.raises(RuntimeError, match="wrong number"):
        classifier.classify(["first", "second"])


@pytest.mark.parametrize(
    "factory, message",
    [
        (LfmSentenceClassifier, "max_new_tokens must be positive"),
        (MlxSentenceClassifier, "max_new_tokens must be positive"),
    ],
)
def test_v10_model_wrappers_validate_model_path_and_token_limit(
    tmp_path: Path, factory, message: str
) -> None:
    with pytest.raises(FileNotFoundError):
        factory(tmp_path / "missing")
    model_path = tmp_path / factory.__name__
    model_path.mkdir()
    with pytest.raises(ValueError, match=message):
        factory(model_path, max_new_tokens=0)


def test_v10_lazy_runtime_loaders_report_missing_optional_dependencies(
    monkeypatch,
) -> None:
    def missing_import(name):
        raise ImportError(name)

    monkeypatch.setattr(inference_module.importlib, "import_module", missing_import)
    with pytest.raises(RuntimeError, match="torch and transformers"):
        inference_module._load_transformers()
    with pytest.raises(RuntimeError) as error:
        inference_module._load_mlx()
    assert str(error.value) == "V10 MLX inference requires the mlx-lm package"


def test_v10_lazy_runtime_loaders_return_runtime_entrypoints(monkeypatch) -> None:
    class Transformers:
        AutoTokenizer = "tokenizer"
        AutoModelForCausalLM = "model"

    modules = {"torch": "torch", "transformers": Transformers}
    monkeypatch.setattr(
        inference_module.importlib, "import_module", modules.__getitem__
    )
    assert inference_module._load_transformers() == ("torch", "tokenizer", "model")

    class Mlx:
        load = "load"

    class Generation:
        batch_generate = "batch"

    modules.update({"mlx_lm": Mlx, "mlx_lm.generate": Generation})
    assert inference_module._load_mlx() == ("load", "batch")


def test_v10_runtime_helpers_select_devices_and_configure_padding() -> None:
    _FakeTorch.backends.mps.available = True
    assert inference_module._select_device(_FakeTorch) == "mps"
    _FakeTorch.backends.mps.available = False
    assert inference_module._select_device(_FakeTorch) == "cpu"

    class Tokenizer:
        pad_token_id = 1
        pad_token = None
        eos_token = "eos"
        padding_side: str

    tokenizer = Tokenizer()
    inference_module._configure_tokenizer(tokenizer)
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token is None


def test_v10_read_rows_uses_one_based_line_numbers_and_utf8(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "v9.jsonl"
    _write_v9_input(input_path)
    encodings = []
    original_open = Path.open

    def open_path(path, *args, **kwargs):
        encodings.append(kwargs.get("encoding"))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_path)

    rows = list(v10_module._read_rows(input_path))

    assert [row.line_number for row in rows] == [1, 2]
    assert encodings == ["utf-8"]


def test_v10_write_output_preserves_batch_boundaries_and_newline_contract(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "v9.jsonl"
    rows = [
        {
            "sentences_with_topic_term": [f"sentence-{index}"],
            "relevant_sentence_metadata": [{"topic_categories": [], "topic_terms": []}],
        }
        for index in range(4)
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output_path = tmp_path / "output.jsonl"
    checkpoint_path = tmp_path / "checkpoint.jsonl"
    checkpoint_path.touch()
    captured = {}

    @contextmanager
    def fake_atomic_text_output(path, **kwargs):
        captured.update(kwargs)
        yield StringIO()

    monkeypatch.setattr(v10_module, "_atomic_text_output", fake_atomic_text_output)
    classifier = _FakeClassifier(["no"] * 4)
    stats = v10_module._OutputStats()

    v10_module._write_output(
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        checkpoint={},
        classifier=classifier,
        batch_size=2,
        stats=stats,
    )

    assert captured == {"newline": "\n"}
    assert classifier.calls == [
        ("sentence-0", "sentence-1"),
        ("sentence-2", "sentence-3"),
    ]


def test_v10_saves_aligned_checkpoint_classifications() -> None:
    candidates = [
        v10_module._CandidateRow(
            index, {}, tuple(f"s{index}-{n}" for n in range(size)), ()
        )
        for index, size in ((1, 1), (2, 2), (3, 1))
    ]
    pending = [v10_module._PendingRow(candidate, None) for candidate in candidates]
    checkpoint = {}
    checkpoint_output = StringIO()

    v10_module._save_classifications(
        pending,
        ["yes", "no", "yes", "no"],
        checkpoint=checkpoint,
        checkpoint_output=checkpoint_output,
    )

    assert [item.labels for item in pending] == [
        ("yes",),
        ("no", "yes"),
        ("no",),
    ]
    assert checkpoint == {1: ("yes",), 2: ("no", "yes"), 3: ("no",)}


def test_v10_classified_row_counts_a_yes_row_as_kept() -> None:
    candidate = v10_module._CandidateRow(
        1,
        {},
        ("a sentence",),
        ({"topic_categories": [], "topic_terms": []},),
    )
    stats = v10_module._OutputStats()

    v10_module._write_classified_row(StringIO(), candidate, ["yes"], stats)

    assert stats.rows_kept == 1
