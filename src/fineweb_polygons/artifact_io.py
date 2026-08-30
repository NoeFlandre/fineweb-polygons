"""Small, shared I/O boundary for reproducible pipeline artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

SHA256_CHUNK_SIZE = 1024 * 1024

__all__ = [
    "SHA256_CHUNK_SIZE",
    "atomic_json_write",
    "atomic_text_output",
    "decode_json_object_line",
    "deterministic_temporary_path",
    "iter_json_objects",
    "read_json_object",
    "sha256_file",
    "temporary_path",
    "write_json_line",
]


def temporary_path(path: Path) -> Path:
    """Return a missing hidden temporary sibling for ``path``."""
    descriptor, name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()
    return temporary


def deterministic_temporary_path(path: Path) -> Path:
    """Return the historical deterministic temporary sibling for ``path``."""
    return path.with_name(f".{path.name}.tmp")


def atomic_json_write(
    path: Path,
    value: Mapping[str, object],
    *,
    temporary_factory: Callable[[Path], Path] = temporary_path,
) -> None:
    """Write a stable JSON object and publish it with one atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_factory(path)
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@contextmanager
def atomic_text_output(
    path: Path,
    *,
    temporary_factory: Callable[[Path], Path] = temporary_path,
    newline: str | None = None,
) -> Iterator[TextIO]:
    """Yield a UTF-8 output stream and publish it after a successful body."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_factory(path)
    try:
        if newline is None:
            stream = temporary.open("w", encoding="utf-8")
        else:
            stream = temporary.open("w", encoding="utf-8", newline=newline)
        with stream as output:
            yield output
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def read_json_object(path: Path) -> dict[str, Any] | None:
    """Read a UTF-8 JSON object, returning ``None`` for unusable manifests."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return value


def decode_json_object_line(
    line: str, line_number: int, *, version: str
) -> dict[str, Any]:
    """Decode one versioned JSONL line and require an object record."""
    if not line.strip():
        raise ValueError(f"{version} JSONL line {line_number} is empty")
    decoded = json.loads(line)
    if not isinstance(decoded, dict):
        raise ValueError(f"{version} JSONL line {line_number} must be an object")
    return decoded


def iter_json_objects(
    path: Path, *, version: str
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield numbered JSONL object records from a UTF-8 file."""
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            yield (
                line_number,
                decode_json_object_line(line, line_number, version=version),
            )


def write_json_line(output: Any, value: Mapping[str, object]) -> None:
    """Write one deterministic, Unicode-safe JSONL record."""
    output.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    """Hash a file while keeping reads bounded to one megabyte."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(SHA256_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
