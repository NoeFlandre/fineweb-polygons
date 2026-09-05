"""V7 post-processing: add exact sentence lists to V6 documents."""

from __future__ import annotations

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
    write_json_line as _write_json_line,
)
from fineweb_polygons.segmentation import (
    SaTSentenceSegmenter,
    SentenceSegmentationConfig,
    SentenceSegmenter,
    validate_segments,
)

V7_VERSION = "v7"
V7_SOURCE_VERSION = "v6"
V7_MODEL_ID = "sat-3l-sm"


@dataclass(frozen=True, slots=True)
class V7RunConfig:
    """Inputs and immutable settings for one V7 artifact."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    model_id: str = V7_MODEL_ID
    backend: str = "onnxruntime"
    stride: int = 64
    block_size: int = 512
    batch_size: int = 32

    def __post_init__(self) -> None:
        _validate_v7_model_id(self.model_id)
        _validate_backend(self.backend)
        _validate_positive("stride", self.stride)
        _validate_positive("block_size", self.block_size)
        _validate_positive("batch_size", self.batch_size)

    def segmentation_record(self) -> dict[str, object]:
        """Return the settings that define sentence boundaries."""
        return SentenceSegmentationConfig(
            model_id=self.model_id,
            stride=self.stride,
            block_size=self.block_size,
            batch_size=self.batch_size,
        ).to_record()


@dataclass(frozen=True, slots=True)
class V7RunSummary:
    """Stable result of a complete or reused V7 run."""

    output_path: Path
    manifest_path: Path
    rows_processed: int
    sentences_written: int
    result_sha256: str


def _validate_v7_model_id(model_id: str) -> None:
    if model_id != V7_MODEL_ID:
        raise ValueError(f"V7 requires model_id {V7_MODEL_ID!r}")


def _validate_backend(backend: str) -> None:
    if not backend.strip():
        raise ValueError("backend must not be empty")


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


def run_v7(
    config: V7RunConfig,
    *,
    segmenter: SentenceSegmenter | None = None,
) -> V7RunSummary:
    """Split every V6 row and publish the result atomically."""
    input_path, output_path, manifest_path = _resolve_paths(config)
    source_sha256 = _sha256_file(input_path)
    reusable = _load_reusable_summary(
        config=config,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        source_sha256=source_sha256,
    )
    if reusable is not None:
        return reusable

    active_segmenter = segmenter or SaTSentenceSegmenter(
        config=SentenceSegmentationConfig(
            model_id=config.model_id,
            stride=config.stride,
            block_size=config.block_size,
            batch_size=config.batch_size,
        )
    )
    segmentation_record = _segmentation_record(config, active_segmenter)
    rows_processed, sentences_written = _write_output(
        input_path=input_path,
        output_path=output_path,
        batch_size=config.batch_size,
        segmenter=active_segmenter,
    )

    result_sha256 = _sha256_file(output_path)
    manifest = _manifest_record(
        config=config,
        input_path=input_path,
        output_path=output_path,
        source_sha256=source_sha256,
        result_sha256=result_sha256,
        segmentation_record=segmentation_record,
        rows_processed=rows_processed,
        sentences_written=sentences_written,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(manifest_path, manifest)
    return V7RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_processed=rows_processed,
        sentences_written=sentences_written,
        result_sha256=result_sha256,
    )


def _resolve_paths(config: V7RunConfig) -> tuple[Path, Path, Path]:
    paths = tuple(
        path.expanduser().resolve()
        for path in (config.input_path, config.output_path, config.manifest_path)
    )
    if len(set(paths)) != 3:
        raise ValueError("V7 input, output, and manifest paths must be different")
    input_path, output_path, manifest_path = paths
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    return input_path, output_path, manifest_path


def _write_output(
    *,
    input_path: Path,
    output_path: Path,
    batch_size: int,
    segmenter: SentenceSegmenter,
) -> tuple[int, int]:
    rows_processed = 0
    sentences_written = 0
    with _atomic_text_output(output_path, newline="\n") as output:
        for rows, texts in _read_batches(input_path, batch_size):
            rows_count, sentence_count = _write_batch(output, rows, texts, segmenter)
            rows_processed += rows_count
            sentences_written += sentence_count
    return rows_processed, sentences_written


def _write_batch(
    output: Any,
    rows: list[dict[str, Any]],
    texts: tuple[str, ...],
    segmenter: SentenceSegmenter,
) -> tuple[int, int]:
    sentence_batches = segmenter.split_many(texts)
    if len(sentence_batches) != len(rows):
        raise ValueError("sentence splitter returned a different number of rows")
    sentences_written = 0
    for row, text, sentences in zip(rows, texts, sentence_batches, strict=True):
        row["sentences"] = list(validate_segments(text, sentences))
        _write_json_line(output, row)
        sentences_written += len(sentences)
    return len(rows), sentences_written


def _manifest_record(
    *,
    config: V7RunConfig,
    input_path: Path,
    output_path: Path,
    source_sha256: str,
    result_sha256: str,
    segmentation_record: Mapping[str, object],
    rows_processed: int,
    sentences_written: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "version": V7_VERSION,
        "source_version": V7_SOURCE_VERSION,
        "status": "complete",
        "source": {"path": str(input_path), "sha256": source_sha256},
        "result": {"path": str(output_path), "sha256": result_sha256},
        "model": {"id": config.model_id, "backend": config.backend},
        "segmentation": dict(segmentation_record),
        "rows_processed": rows_processed,
        "sentences_written": sentences_written,
    }


def _read_batches(
    input_path: Path, batch_size: int
) -> Iterator[tuple[list[dict[str, Any]], tuple[str, ...]]]:
    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for line_number, decoded in _iter_json_objects(input_path, version="V6"):
        text = _text_from_row(decoded, line_number)
        rows.append(decoded)
        texts.append(text)
        if len(rows) == batch_size:
            yield rows, tuple(texts)
            rows = []
            texts = []
    if rows:
        yield rows, tuple(texts)


def _text_from_row(decoded: dict[str, Any], line_number: int) -> str:
    text = decoded.get("text")
    if not isinstance(text, str):
        raise ValueError(
            f"V6 JSONL line {line_number} must contain a string text field"
        )
    return text


def _segmentation_record(
    config: V7RunConfig, segmenter: SentenceSegmenter
) -> dict[str, object]:
    record = config.segmentation_record()
    configuration_record = getattr(segmenter, "configuration_record", None)
    if callable(configuration_record):
        candidate = configuration_record()
        if isinstance(candidate, Mapping):
            record.update(candidate)
    return dict(record)


def _load_reusable_summary(
    *,
    config: V7RunConfig,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_sha256: str,
) -> V7RunSummary | None:
    manifest = _read_reusable_manifest(
        config=config,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        source_sha256=source_sha256,
    )
    if manifest is None:
        return None
    counts = _manifest_counts(manifest)
    if counts is None:
        return None
    rows_processed, sentences_written = counts
    result = manifest["result"]
    assert isinstance(result, Mapping)
    result_sha256 = result["sha256"]
    assert isinstance(result_sha256, str)
    return V7RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_processed=rows_processed,
        sentences_written=sentences_written,
        result_sha256=result_sha256,
    )


def _read_reusable_manifest(
    *,
    config: V7RunConfig,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    source_sha256: str,
) -> dict[str, Any] | None:
    if not output_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _read_manifest(manifest_path)
    if manifest is None or not _manifest_matches(
        manifest, config, input_path, output_path, source_sha256
    ):
        return None
    return _with_valid_result_hash(manifest, output_path)


def _with_valid_result_hash(
    manifest: dict[str, Any], output_path: Path
) -> dict[str, Any] | None:
    result = manifest["result"]
    assert isinstance(result, Mapping)
    result_sha256 = _sha256_file(output_path)
    if result["sha256"] != result_sha256:
        return None
    return manifest


def _manifest_matches(
    manifest: Mapping[str, object],
    config: V7RunConfig,
    input_path: Path,
    output_path: Path,
    source_sha256: str,
) -> bool:
    source = manifest.get("source")
    result = manifest.get("result")
    if not isinstance(source, Mapping) or not isinstance(result, Mapping):
        return False
    return all(
        (
            manifest.get("schema_version") == 1,
            manifest.get("version") == V7_VERSION,
            manifest.get("source_version") == V7_SOURCE_VERSION,
            manifest.get("status") == "complete",
            source.get("path") == str(input_path),
            source.get("sha256") == source_sha256,
            result.get("path") == str(output_path),
            manifest.get("model") == {"id": config.model_id, "backend": config.backend},
            _matching_segmentation_settings(manifest, config),
        )
    )


def _manifest_counts(manifest: Mapping[str, object]) -> tuple[int, int] | None:
    rows_processed = manifest.get("rows_processed")
    sentences_written = manifest.get("sentences_written")
    if not isinstance(rows_processed, int) or not isinstance(sentences_written, int):
        return None
    return rows_processed, sentences_written


def _matching_segmentation_settings(
    manifest: Mapping[str, object], config: V7RunConfig
) -> bool:
    recorded = manifest.get("segmentation")
    if not isinstance(recorded, Mapping):
        return False
    return all(
        recorded.get(key) == value
        for key, value in config.segmentation_record().items()
    )
