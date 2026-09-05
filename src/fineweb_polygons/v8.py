"""V8 post-processing: filter V7 documents with a strong topic vocabulary."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fineweb_polygons.artifact_io import (
    atomic_json_write as _atomic_json_write,
)
from fineweb_polygons.artifact_io import (
    atomic_text_output as _atomic_text_output,
)
from fineweb_polygons.artifact_io import (
    iter_json_objects as _iter_json_objects,
)
from fineweb_polygons.artifact_io import (
    read_json_object as _read_manifest,
)
from fineweb_polygons.artifact_io import (
    sha256_file as _sha256_file,
)
from fineweb_polygons.artifact_io import (
    write_json_line as _write_kept_row,
)
from fineweb_polygons.topic_vocabulary import (
    TopicVocabulary,
    load_vocabulary,
)

V8_VERSION = "v8"
V8_SOURCE_VERSION = "v7"


@dataclass(frozen=True, slots=True)
class V8RunConfig:
    """Inputs and immutable settings for one V8 artifact."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    vocabulary_path: Path

    def __post_init__(self) -> None:
        paths = tuple(
            path.expanduser().resolve()
            for path in (
                self.input_path,
                self.output_path,
                self.manifest_path,
                self.vocabulary_path,
            )
        )
        if len(set(paths)) != 4:
            raise ValueError("V8 paths must be different")


@dataclass(frozen=True, slots=True)
class V8RunSummary:
    """Stable result of a complete or reused V8 run."""

    output_path: Path
    manifest_path: Path
    rows_processed: int
    rows_kept: int
    rows_filtered: int
    category_documents: dict[str, int]
    result_sha256: str


def run_v8(
    config: V8RunConfig,
    *,
    vocabulary: TopicVocabulary | None = None,
) -> V8RunSummary:
    """Filter V7 rows and publish a resumable V8 artifact."""
    input_path, output_path, manifest_path, vocabulary_path = _resolve_paths(config)
    source_sha256 = _sha256_file(input_path)
    vocabulary_sha256 = _sha256_file(vocabulary_path)
    reusable = _load_reusable_summary(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        vocabulary_path=vocabulary_path,
        source_sha256=source_sha256,
        vocabulary_sha256=vocabulary_sha256,
    )
    if reusable is not None:
        return reusable

    active_vocabulary = vocabulary or load_vocabulary(vocabulary_path)
    rows_processed, rows_kept, category_documents = _write_output(
        input_path=input_path,
        output_path=output_path,
        vocabulary=active_vocabulary,
    )
    rows_filtered = rows_processed - rows_kept
    result_sha256 = _sha256_file(output_path)
    manifest = _manifest_record(
        input_path=input_path,
        output_path=output_path,
        vocabulary_path=vocabulary_path,
        source_sha256=source_sha256,
        vocabulary_sha256=vocabulary_sha256,
        result_sha256=result_sha256,
        vocabulary=active_vocabulary,
        rows_processed=rows_processed,
        rows_kept=rows_kept,
        rows_filtered=rows_filtered,
        category_documents=category_documents,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(manifest_path, manifest)
    return V8RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_processed=rows_processed,
        rows_kept=rows_kept,
        rows_filtered=rows_filtered,
        category_documents=category_documents,
        result_sha256=result_sha256,
    )


def _resolve_paths(
    config: V8RunConfig,
) -> tuple[Path, Path, Path, Path]:
    paths = tuple(
        path.expanduser().resolve()
        for path in (
            config.input_path,
            config.output_path,
            config.manifest_path,
            config.vocabulary_path,
        )
    )
    if len(set(paths)) != 4:
        raise ValueError("V8 paths must be different")
    input_path, output_path, manifest_path, vocabulary_path = paths
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not vocabulary_path.is_file():
        raise FileNotFoundError(vocabulary_path)
    return input_path, output_path, manifest_path, vocabulary_path


def _write_output(
    *,
    input_path: Path,
    output_path: Path,
    vocabulary: TopicVocabulary,
) -> tuple[int, int, dict[str, int]]:
    rows_processed = 0
    rows_kept = 0
    category_documents: Counter[str] = Counter()
    with _atomic_text_output(output_path, newline="\n") as output:
        for row, text in _read_rows(input_path):
            rows_processed += 1
            categories = _matching_categories(vocabulary, text)
            if not categories:
                continue
            _write_kept_row(output, row)
            rows_kept += 1
            category_documents.update(categories)
    return rows_processed, rows_kept, dict(sorted(category_documents.items()))


def _matching_categories(vocabulary: TopicVocabulary, text: str) -> tuple[str, ...]:
    matches = vocabulary.match_text(text)
    return tuple(dict.fromkeys(match.category for match in matches))


def _read_rows(input_path: Path) -> Iterator[tuple[dict[str, Any], str]]:
    for line_number, decoded in _iter_json_objects(input_path, version="V7"):
        yield _decode_text_row(decoded, line_number)


def _decode_text_row(
    decoded: dict[str, Any], line_number: int
) -> tuple[dict[str, Any], str]:
    text = decoded.get("text")
    if not isinstance(text, str):
        raise ValueError(
            f"V7 JSONL line {line_number} must contain a string text field"
        )
    return decoded, text


def _manifest_record(
    *,
    input_path: Path,
    output_path: Path,
    vocabulary_path: Path,
    source_sha256: str,
    vocabulary_sha256: str,
    result_sha256: str,
    vocabulary: TopicVocabulary,
    rows_processed: int,
    rows_kept: int,
    rows_filtered: int,
    category_documents: Mapping[str, int],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": V8_VERSION,
        "source_version": V8_SOURCE_VERSION,
        "status": "complete",
        "source": {"path": str(input_path), "sha256": source_sha256},
        "result": {"path": str(output_path), "sha256": result_sha256},
        "vocabulary": {
            "path": str(vocabulary_path),
            "sha256": vocabulary_sha256,
            "definition": vocabulary.to_record(),
        },
        "rows_processed": rows_processed,
        "rows_kept": rows_kept,
        "rows_filtered": rows_filtered,
        "category_documents": dict(sorted(category_documents.items())),
    }


def _load_reusable_summary(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    vocabulary_path: Path,
    source_sha256: str,
    vocabulary_sha256: str,
) -> V8RunSummary | None:
    manifest = _load_reusable_manifest(
        manifest_path=manifest_path,
        input_path=input_path,
        output_path=output_path,
        vocabulary_path=vocabulary_path,
        source_sha256=source_sha256,
        vocabulary_sha256=vocabulary_sha256,
    )
    if manifest is None:
        return None
    counts = _manifest_counts(manifest)
    if counts is None:
        return None
    rows_processed, rows_kept, rows_filtered, category_documents = counts
    result = manifest["result"]
    assert isinstance(result, Mapping)
    result_sha256 = result["sha256"]
    assert isinstance(result_sha256, str)
    return V8RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_processed=rows_processed,
        rows_kept=rows_kept,
        rows_filtered=rows_filtered,
        category_documents=category_documents,
        result_sha256=result_sha256,
    )


def _load_reusable_manifest(
    *,
    manifest_path: Path,
    input_path: Path,
    output_path: Path,
    vocabulary_path: Path,
    source_sha256: str,
    vocabulary_sha256: str,
) -> dict[str, Any] | None:
    if not output_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return None
    if not _manifest_matches(
        manifest,
        input_path,
        output_path,
        vocabulary_path,
        source_sha256,
        vocabulary_sha256,
    ):
        return None
    return _with_valid_result_hash(manifest, output_path)


def _manifest_matches(
    manifest: Mapping[str, object],
    input_path: Path,
    output_path: Path,
    vocabulary_path: Path,
    source_sha256: str,
    vocabulary_sha256: str,
) -> bool:
    source = manifest.get("source")
    result = manifest.get("result")
    vocabulary = manifest.get("vocabulary")
    if not isinstance(source, Mapping):
        return False
    if not isinstance(result, Mapping):
        return False
    if not isinstance(vocabulary, Mapping):
        return False
    return all(
        (
            manifest.get("schema_version") == 1,
            manifest.get("version") == V8_VERSION,
            manifest.get("source_version") == V8_SOURCE_VERSION,
            manifest.get("status") == "complete",
            source.get("path") == str(input_path),
            source.get("sha256") == source_sha256,
            result.get("path") == str(output_path),
            vocabulary.get("path") == str(vocabulary_path),
            vocabulary.get("sha256") == vocabulary_sha256,
        )
    )


def _with_valid_result_hash(
    manifest: dict[str, Any], output_path: Path
) -> dict[str, Any] | None:
    result = manifest.get("result")
    if not isinstance(result, Mapping):
        return None
    result_sha256 = _sha256_file(output_path)
    if result.get("sha256") != result_sha256:
        return None
    return manifest


def _manifest_counts(
    manifest: Mapping[str, object],
) -> tuple[int, int, int, dict[str, int]] | None:
    row_counts = _manifest_row_counts(manifest)
    category_documents = _manifest_category_documents(manifest)
    if row_counts is None or category_documents is None:
        return None
    return (*row_counts, category_documents)


def _manifest_row_counts(
    manifest: Mapping[str, object],
) -> tuple[int, int, int] | None:
    values: list[int] = []
    for key in ("rows_processed", "rows_kept", "rows_filtered"):
        value = _manifest_integer(manifest.get(key))
        if value is None:
            return None
        values.append(value)
    return values[0], values[1], values[2]


def _manifest_integer(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _manifest_category_documents(
    manifest: Mapping[str, object],
) -> dict[str, int] | None:
    value = manifest.get("category_documents")
    if not isinstance(value, Mapping):
        return None
    return _validated_category_documents(value)


def _validated_category_documents(
    value: Mapping[object, object],
) -> dict[str, int] | None:
    documents: dict[str, int] = {}
    for category, count in value.items():
        if not isinstance(category, str) or not isinstance(count, int):
            return None
        documents[category] = count
    return documents
