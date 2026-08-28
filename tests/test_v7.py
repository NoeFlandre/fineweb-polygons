from __future__ import annotations

import io
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import fineweb_polygons.artifact_io as artifact_io
import fineweb_polygons.v7 as v7_module
from fineweb_polygons.v7 import V7RunConfig, run_v7


class FakeSegmenter:
    def __init__(self, results: list[tuple[str, ...]]) -> None:
        self.results = results
        self.calls: list[tuple[str, ...]] = []

    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        self.calls.append(texts)
        start = sum(len(batch) for batch in self.calls[:-1])
        return tuple(self.results[start : start + len(texts)])


@pytest.fixture(autouse=True)
def _stub_native_onnxruntime_for_unit_tests(monkeypatch) -> None:
    """Keep V7 unit tests away from native ONNX Runtime initialization."""
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ("UnsupportedExecutionProvider",)
        ),
    )


def _write_v6_input(path: Path) -> list[dict[str, str | int]]:
    rows = [
        {
            "polygon_id": "way/1",
            "polygon_name": "Fontvieille",
            "text": "Fontvieille is in Monaco. It is near the port.",
            "url": "https://example.test/fontvieille",
            "name_country_distance": 25,
        },
        {
            "polygon_id": "relation/2",
            "polygon_name": "Larvotto",
            "text": "Larvotto is a district in Monaco.",
            "url": "https://example.test/larvotto",
            "name_country_distance": 31,
        },
    ]
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )
    return rows


def _config(tmp_path: Path) -> V7RunConfig:
    return V7RunConfig(
        input_path=tmp_path / "v6.jsonl",
        output_path=tmp_path / "artifacts" / "v7.jsonl",
        manifest_path=tmp_path / "runs" / "v7" / "manifest.json",
    )


def test_run_v7_adds_ordered_sentence_list_and_preserves_v6_fields(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    source_rows = _write_v6_input(config.input_path)
    segmenter = FakeSegmenter(
        [
            ("Fontvieille is in Monaco. ", "It is near the port."),
            ("Larvotto is a district in Monaco.",),
        ]
    )

    summary = run_v7(config, segmenter=segmenter)
    output_rows = [
        json.loads(line)
        for line in config.output_path.read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))

    assert summary.rows_processed == 2
    assert summary.sentences_written == 3
    assert [row["polygon_id"] for row in output_rows] == ["way/1", "relation/2"]
    assert output_rows[0]["sentences"] == [
        "Fontvieille is in Monaco. ",
        "It is near the port.",
    ]
    assert output_rows[0]["text"] == source_rows[0]["text"]
    assert output_rows[1]["sentences"] == ["Larvotto is a district in Monaco."]
    assert set(output_rows[0]) == set(source_rows[0]) | {"sentences"}
    assert manifest["version"] == "v7"
    assert manifest["source_version"] == "v6"
    assert manifest["status"] == "complete"
    assert manifest["model"] == {"id": "sat-3l-sm", "backend": "onnxruntime"}
    assert manifest["rows_processed"] == 2
    assert manifest["sentences_written"] == 3
    assert manifest["source"]["sha256"]
    assert manifest["result"]["sha256"] == summary.result_sha256


def test_run_v7_builds_the_default_segmenter_from_run_settings(
    tmp_path: Path, monkeypatch
) -> None:
    config = replace(_config(tmp_path), batch_size=1)
    _write_v6_input(config.input_path)
    captured: dict[str, object] = {}

    class StubSegmentationConfig:
        def __init__(self, **kwargs: object) -> None:
            self.values = kwargs

        def to_record(self) -> dict[str, object]:
            return dict(self.values)

    class StubSegmenter:
        def __init__(self, *, config: StubSegmentationConfig) -> None:
            captured["config"] = config
            captured["segmenter"] = self
            self.calls: list[tuple[str, ...]] = []

        def configuration_record(self) -> dict[str, object]:
            return {"constructed_by": "default-segmenter"}

        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            self.calls.append(texts)
            return tuple((text,) for text in texts)

    monkeypatch.setattr(v7_module, "SentenceSegmentationConfig", StubSegmentationConfig)
    monkeypatch.setattr(v7_module, "SaTSentenceSegmenter", StubSegmenter)

    run_v7(config)

    segmenter_config = captured["config"]
    assert isinstance(segmenter_config, StubSegmentationConfig)
    assert segmenter_config.values == {
        "model_id": "sat-3l-sm",
        "stride": 64,
        "block_size": 512,
        "batch_size": 1,
    }
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
    assert manifest["segmentation"]["constructed_by"] == "default-segmenter"
    segmenter = captured["segmenter"]
    assert isinstance(segmenter, StubSegmenter)
    assert segmenter.calls == [
        ("Fontvieille is in Monaco. It is near the port.",),
        ("Larvotto is a district in Monaco.",),
    ]


def test_run_v7_is_deterministic_and_skips_a_valid_completed_run(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v6_input(config.input_path)
    first_segmenter = FakeSegmenter(
        [
            ("Fontvieille is in Monaco. ", "It is near the port."),
            ("Larvotto is a district in Monaco.",),
        ]
    )

    first = run_v7(config, segmenter=first_segmenter)
    output_bytes = config.output_path.read_bytes()
    manifest_bytes = config.manifest_path.read_bytes()

    class ShouldNotRun:
        def split_many(self, texts: tuple[str, ...]):
            raise AssertionError("a valid completed V7 run should be reusable")

    second = run_v7(config, segmenter=ShouldNotRun())

    assert first == second
    assert config.output_path.read_bytes() == output_bytes
    assert config.manifest_path.read_bytes() == manifest_bytes


def test_run_v7_fails_before_publishing_when_sentences_do_not_align(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_v6_input(config.input_path)
    segmenter = FakeSegmenter(
        [
            ("Changed text.",),
            ("Larvotto is a district in Monaco.",),
        ]
    )

    with pytest.raises(ValueError, match="do not reconstruct the input text"):
        run_v7(config, segmenter=segmenter)

    assert not config.output_path.exists()
    assert not config.manifest_path.exists()


def test_run_v7_rejects_rows_without_full_text(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.input_path.write_text('{"polygon_id": "way/1"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1 must contain a string text field"):
        run_v7(config, segmenter=FakeSegmenter([]))

    assert not config.output_path.exists()
    assert not config.manifest_path.exists()


def test_v7_config_rejects_invalid_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires model_id"):
        replace(_config(tmp_path), model_id="other")
    with pytest.raises(ValueError) as error:
        replace(_config(tmp_path), backend="")
    assert str(error.value) == "backend must not be empty"
    with pytest.raises(ValueError, match="stride must be positive"):
        replace(_config(tmp_path), stride=0)
    with pytest.raises(ValueError, match="block_size must be positive"):
        replace(_config(tmp_path), block_size=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        replace(_config(tmp_path), batch_size=0)


def test_run_v7_rejects_duplicate_paths_and_missing_input(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v6_input(config.input_path)
    duplicate = V7RunConfig(
        input_path=config.input_path,
        output_path=config.input_path,
        manifest_path=config.manifest_path,
    )
    with pytest.raises(ValueError) as error:
        run_v7(duplicate, segmenter=FakeSegmenter([]))
    assert str(error.value) == "V7 input, output, and manifest paths must be different"

    missing = _config(tmp_path / "missing")
    with pytest.raises(FileNotFoundError) as error:
        run_v7(missing, segmenter=FakeSegmenter([]))
    assert error.value.args == (missing.input_path.resolve(),)


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("\n", "V6 JSONL line 1 is empty"),
        ("3\n", "V6 JSONL line 1 must be an object"),
    ],
)
def test_run_v7_rejects_invalid_jsonl_rows(
    tmp_path: Path, line: str, message: str
) -> None:
    config = _config(tmp_path)
    config.input_path.write_text(line, encoding="utf-8")

    with pytest.raises(ValueError) as error:
        run_v7(config, segmenter=FakeSegmenter([]))
    assert str(error.value) == message

    assert not config.output_path.exists()
    assert not config.manifest_path.exists()


def test_run_v7_rejects_a_sentence_batch_count_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v6_input(config.input_path)

    with pytest.raises(ValueError) as error:
        run_v7(config, segmenter=FakeSegmenter([]))
    assert str(error.value) == "sentence splitter returned a different number of rows"

    assert not config.output_path.exists()
    assert not config.manifest_path.exists()


@pytest.mark.parametrize(
    "manifest_value",
    [
        "not an object",
        {"result": {"path": "wrong"}},
        {"source": [], "result": {}},
        {"rows_processed": "two"},
    ],
)
def test_run_v7_rebuilds_when_completed_manifest_is_not_reusable(
    tmp_path: Path, manifest_value: object
) -> None:
    config = _config(tmp_path)
    _write_v6_input(config.input_path)
    rows = [
        ("Fontvieille is in Monaco. ", "It is near the port."),
        ("Larvotto is a district in Monaco.",),
    ]
    first_segmenter = FakeSegmenter(rows)
    run_v7(config, segmenter=first_segmenter)
    if isinstance(manifest_value, str):
        config.manifest_path.write_text(manifest_value, encoding="utf-8")
    else:
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        manifest.update(manifest_value)
        config.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second_segmenter = FakeSegmenter(rows)
    run_v7(config, segmenter=second_segmenter)

    assert second_segmenter.calls


def test_run_v7_rebuilds_when_output_hash_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_v6_input(config.input_path)
    rows = [
        ("Fontvieille is in Monaco. ", "It is near the port."),
        ("Larvotto is a district in Monaco.",),
    ]
    run_v7(config, segmenter=FakeSegmenter(rows))
    config.output_path.write_text("tampered\n", encoding="utf-8")

    segmenter = FakeSegmenter(rows)
    run_v7(config, segmenter=segmenter)

    assert segmenter.calls


def test_read_reusable_manifest_requires_both_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path)
    config.manifest_path.parent.mkdir(parents=True)
    config.manifest_path.write_text("{}", encoding="utf-8")

    def should_not_read_manifest(path: Path) -> dict[str, object]:
        raise AssertionError(f"manifest should not be read: {path}")

    monkeypatch.setattr(v7_module, "_read_manifest", should_not_read_manifest)

    assert (
        v7_module._read_reusable_manifest(
            config=config,
            input_path=config.input_path,
            output_path=config.output_path,
            manifest_path=config.manifest_path,
            source_sha256="source-hash",
        )
        is None
    )


def test_write_batch_requires_strict_text_and_sentence_alignment() -> None:
    class FixedSegmenter:
        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            return (("Text.",),)

    with pytest.raises(ValueError, match="zip"):
        v7_module._write_batch(
            io.StringIO(),
            [{"text": "Text."}],
            (),
            FixedSegmenter(),
        )


def test_write_batch_uses_unicode_and_sorted_json(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_dumps(value: object, **kwargs: object) -> str:
        captured["value"] = value
        captured["kwargs"] = kwargs
        return "serialized"

    class FixedSegmenter:
        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            return (("Héllo.",),)

    monkeypatch.setattr(v7_module.json, "dumps", fake_dumps)
    output = io.StringIO()
    rows = [{"z": 1, "text": "Héllo."}]

    assert v7_module._write_batch(output, rows, ("Héllo.",), FixedSegmenter()) == (1, 1)
    assert captured == {
        "value": {"text": "Héllo.", "z": 1, "sentences": ["Héllo."]},
        "kwargs": {"ensure_ascii": False, "sort_keys": True},
    }
    assert output.getvalue() == "serialized\n"


def test_write_output_creates_nested_directories_and_uses_utf8_newlines(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "v6.jsonl"
    input_path.write_text(json.dumps({"text": "Héllo."}) + "\n", encoding="utf-8")
    output_path = tmp_path / "nested" / "deeper" / "v7.jsonl"
    write_calls: list[dict[str, object]] = []
    original_open = Path.open

    def recording_open(path: Path, *args, **kwargs):
        if args and args[0] == "w":
            write_calls.append(dict(kwargs))
        return original_open(path, *args, **kwargs)

    class FixedSegmenter:
        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            return (("Héllo.",),)

    monkeypatch.setattr(Path, "open", recording_open)

    assert v7_module._write_output(
        input_path=input_path,
        output_path=output_path,
        batch_size=1,
        segmenter=FixedSegmenter(),
    ) == (1, 1)
    assert output_path.is_file()
    assert write_calls == [{"encoding": "utf-8", "newline": "\n"}]


def test_read_batches_requests_utf8_input(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "v6.jsonl"
    input_path.write_text('{"text": "Héllo."}\n', encoding="utf-8")
    read_calls: list[dict[str, object]] = []
    original_open = Path.open

    def recording_open(path: Path, *args, **kwargs):
        read_calls.append(dict(kwargs))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    batches = list(v7_module._read_batches(input_path, batch_size=1))

    assert batches[0][1] == ("Héllo.",)
    assert read_calls == [{"encoding": "utf-8"}]


def test_write_output_respects_batch_size_and_accumulates_counts(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "v6.jsonl"
    input_path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in ("One.", "Two.")),
        encoding="utf-8",
    )
    output_path = tmp_path / "v7.jsonl"

    class FixedSegmenter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            self.calls.append(texts)
            return tuple((text,) for text in texts)

    segmenter = FixedSegmenter()

    assert v7_module._write_output(
        input_path=input_path,
        output_path=output_path,
        batch_size=1,
        segmenter=segmenter,
    ) == (2, 2)
    assert segmenter.calls == [("One.",), ("Two.",)]


def test_write_output_cleans_missing_temporary_file_after_failure(
    tmp_path: Path, monkeypatch
) -> None:
    input_path = tmp_path / "v6.jsonl"
    input_path.write_text('{"text": "One."}\n', encoding="utf-8")
    output_path = tmp_path / "v7.jsonl"
    unlink_calls: list[bool] = []
    original_unlink = Path.unlink

    def recording_unlink(path: Path, *, missing_ok: bool = False) -> None:
        unlink_calls.append(missing_ok)
        original_unlink(path, missing_ok=missing_ok)

    class BrokenSegmenter:
        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            raise RuntimeError("split failed")

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    with pytest.raises(RuntimeError, match="split failed"):
        v7_module._write_output(
            input_path=input_path,
            output_path=output_path,
            batch_size=1,
            segmenter=BrokenSegmenter(),
        )

    assert unlink_calls[-1] is True
    assert not output_path.exists()


def test_temporary_path_stays_next_to_target_and_is_removed(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "v7.jsonl"
    target.parent.mkdir()

    temporary = artifact_io.temporary_path(target)

    assert temporary.parent == target.parent
    assert temporary.name.startswith(f".{target.name}.")
    assert temporary.name.endswith(".tmp")
    assert not temporary.exists()


def test_segmentation_record_includes_segmenter_configuration(tmp_path: Path) -> None:
    class ConfiguredSegmenter:
        def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
            return tuple((text,) for text in texts)

        def configuration_record(self) -> dict[str, object]:
            return {
                "batch_size": 4,
                "block_size": 128,
                "model_id": "sat-3l-sm",
                "providers": ["CPUExecutionProvider"],
            }

    record = v7_module._segmentation_record(_config(tmp_path), ConfiguredSegmenter())

    assert record["batch_size"] == 4
    assert record["block_size"] == 128
    assert record["providers"] == ["CPUExecutionProvider"]


def test_read_manifest_requests_utf8(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_read_text(path: Path, **kwargs: object) -> str:
        captured.update(kwargs)
        return "{}"

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    assert v7_module._read_manifest(Path("manifest.json")) == {}
    assert captured == {"encoding": "utf-8"}


def test_matching_segmentation_settings_rejects_non_mapping_records(
    tmp_path: Path,
) -> None:
    assert not v7_module._matching_segmentation_settings(
        {"segmentation": None}, _config(tmp_path)
    )


def test_atomic_json_write_uses_stable_json_and_cleans_temp_file(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    write_kwargs: dict[str, object] = {}

    def fake_dumps(value: object, **kwargs: object) -> str:
        captured["value"] = value
        captured["kwargs"] = kwargs
        return "payload"

    original_write_text = Path.write_text

    def recording_write_text(path: Path, data: str, **kwargs: str | None) -> int:
        write_kwargs.update(kwargs)
        return original_write_text(path, data, **kwargs)

    monkeypatch.setattr(v7_module.json, "dumps", fake_dumps)
    monkeypatch.setattr(Path, "write_text", recording_write_text)
    path = tmp_path / "manifest.json"

    v7_module._atomic_json_write(path, {"z": 1, "é": 2})

    assert captured == {
        "value": {"z": 1, "é": 2},
        "kwargs": {"ensure_ascii": False, "indent": 2, "sort_keys": True},
    }
    assert write_kwargs == {"encoding": "utf-8"}
    assert path.read_text(encoding="utf-8") == "payload\n"


def test_atomic_json_write_removes_temporary_file_on_replace_failure(
    tmp_path: Path, monkeypatch
) -> None:
    unlink_calls: list[bool] = []
    original_unlink = Path.unlink

    def recording_unlink(path: Path, *, missing_ok: bool = False) -> None:
        unlink_calls.append(missing_ok)
        original_unlink(path, missing_ok=missing_ok)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(Path, "unlink", recording_unlink)
    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        v7_module._atomic_json_write(tmp_path / "manifest.json", {"ok": True})

    assert unlink_calls[-1] is True


def test_sha256_file_uses_megabyte_chunks_and_stops_on_empty_read(
    tmp_path: Path,
) -> None:
    expected_chunk_size = 1024 * 1024

    class Source:
        def __init__(self) -> None:
            self.read_sizes: list[int] = []

        def __enter__(self) -> Source:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            self.read_sizes.append(size)
            if size != expected_chunk_size:
                return b""
            if len(self.read_sizes) == 1:
                return b"payload"
            if len(self.read_sizes) == 2:
                return b""
            raise AssertionError("the empty read should stop iteration")

    source = Source()

    class FakePath:
        def open(self, mode: str) -> Source:
            assert mode == "rb"
            return source

    digest = v7_module._sha256_file(cast(Path, FakePath()))

    import hashlib

    assert digest == hashlib.sha256(b"payload").hexdigest()
    assert source.read_sizes == [expected_chunk_size, expected_chunk_size]
