from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path
from typing import Any, cast

import pytest

import fineweb_polygons.artifact_io as artifact_io


def test_temporary_path_is_a_missing_hidden_sibling(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.jsonl"
    target.parent.mkdir()

    temporary = artifact_io.temporary_path(target)

    assert temporary.parent == target.parent
    assert temporary.name.startswith(f".{target.name}.")
    assert temporary.name.endswith(".tmp")
    assert not temporary.exists()


def test_deterministic_temporary_path_is_a_hidden_sibling(tmp_path: Path) -> None:
    target = tmp_path / "result.jsonl"

    temporary = artifact_io.deterministic_temporary_path(target)

    assert temporary == tmp_path / ".result.jsonl.tmp"


def test_atomic_text_output_publishes_complete_utf8_text(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "result.jsonl"

    with artifact_io.atomic_text_output(
        target,
        temporary_factory=artifact_io.deterministic_temporary_path,
    ) as output:
        output.write("évidence\n")

    assert target.read_text(encoding="utf-8") == "évidence\n"
    assert not artifact_io.deterministic_temporary_path(target).exists()


def test_atomic_text_output_cleans_up_when_body_fails(tmp_path: Path) -> None:
    target = tmp_path / "result.jsonl"

    with (
        pytest.raises(RuntimeError, match="body failed"),
        artifact_io.atomic_text_output(
            target,
            temporary_factory=artifact_io.deterministic_temporary_path,
        ) as output,
    ):
        output.write("partial\n")
        raise RuntimeError("body failed")

    assert not target.exists()
    assert not artifact_io.deterministic_temporary_path(target).exists()


def test_atomic_text_output_cleans_up_when_open_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.jsonl"
    temporary = artifact_io.deterministic_temporary_path(target)
    original_open: Any = Path.open

    def fail_open(self: Path, *args: object, **kwargs: object):
        if self == temporary:
            raise RuntimeError("open failed")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)

    with (
        pytest.raises(RuntimeError, match="open failed"),
        artifact_io.atomic_text_output(
            target,
            temporary_factory=artifact_io.deterministic_temporary_path,
        ),
    ):
        pass

    assert not temporary.exists()


def test_atomic_json_write_creates_sorted_unicode_safe_json(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "manifest.json"

    artifact_io.atomic_json_write(target, {"z": "é", "a": {"b": 1}})

    assert target.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "b": 1\n  },\n  "z": "é"\n}\n'
    )


def test_atomic_json_write_removes_temporary_file_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary_paths: list[Path] = []

    def fail_replace(source: Path, target: Path) -> None:
        temporary_paths.append(source)
        raise OSError("replace failed")

    monkeypatch.setattr(artifact_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        artifact_io.atomic_json_write(tmp_path / "manifest.json", {"ok": True})

    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()


def test_read_json_object_returns_only_valid_json_objects(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    invalid = tmp_path / "invalid.json"
    not_object = tmp_path / "list.json"
    valid = tmp_path / "valid.json"
    invalid.write_text("not json", encoding="utf-8")
    not_object.write_text("[]", encoding="utf-8")
    valid.write_text('{"status": "complete"}', encoding="utf-8")

    assert artifact_io.read_json_object(missing) is None
    assert artifact_io.read_json_object(invalid) is None
    assert artifact_io.read_json_object(not_object) is None
    assert artifact_io.read_json_object(valid) == {"status": "complete"}


def test_decode_json_object_line_reports_versioned_errors() -> None:
    with pytest.raises(ValueError) as empty_error:
        artifact_io.decode_json_object_line("\n", 7, version="V7")
    assert str(empty_error.value) == "V7 JSONL line 7 is empty"

    with pytest.raises(ValueError) as object_error:
        artifact_io.decode_json_object_line("[]", 8, version="V7")
    assert str(object_error.value) == "V7 JSONL line 8 must be an object"


def test_iter_json_objects_yields_line_numbers_and_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"b": "é"}\n', encoding="utf-8")
    real_open: Any = Path.open
    encodings: list[object] = []

    def recording_open(self: Path, *args: object, **kwargs: object):
        if self == path:
            encodings.append(kwargs.get("encoding"))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    assert list(artifact_io.iter_json_objects(path, version="V7")) == [
        (1, {"a": 1}),
        (2, {"b": "é"}),
    ]
    assert encodings == ["utf-8"]

    invalid = tmp_path / "invalid-rows.jsonl"
    invalid.write_text("{}\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        list(artifact_io.iter_json_objects(invalid, version="V7"))
    assert str(error.value) == "V7 JSONL line 2 must be an object"


def test_write_json_line_is_sorted_and_unicode_safe() -> None:
    output = StringIO()

    artifact_io.write_json_line(output, {"z": "é", "a": 1})

    assert output.getvalue() == '{"a": 1, "z": "é"}\n'


def test_sha256_file_hashes_file_contents(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(b"payload")

    assert artifact_io.sha256_file(target) == hashlib.sha256(b"payload").hexdigest()


def test_sha256_file_stops_after_an_empty_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_sizes: list[int] = []
    chunks = iter((b"first", b"second", b"", b"must not be read"))

    class Source:
        def __enter__(self) -> Source:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, size: int) -> bytes:
            read_sizes.append(size)
            return next(chunks)

    class FakePath:
        def open(self, mode: str) -> Source:
            assert mode == "rb"
            return Source()

    assert (
        artifact_io.sha256_file(cast(Path, FakePath()))
        == hashlib.sha256(b"firstsecond").hexdigest()
    )
    assert read_sizes == [1024 * 1024, 1024 * 1024, 1024 * 1024]


def test_json_object_reader_returns_none_for_empty_files(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")

    assert artifact_io.read_json_object(path) is None
