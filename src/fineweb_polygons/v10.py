"""V10 post-processing: classify V9 topic sentences with a local LFM model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, TypeGuard, cast

from fineweb_polygons.artifact_io import (
    atomic_json_write as _atomic_json_write,
)
from fineweb_polygons.artifact_io import (
    atomic_text_output as _atomic_text_output,
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
from fineweb_polygons.v10_inference import (
    PROMPT_SHA256,
    REASONING_CLOSE_TAG,
    V10_PROMPT_TEMPLATE,
    LfmSentenceClassifier,
    MlxSentenceClassifier,
)
from fineweb_polygons.v10_models import (
    V10_BATCH_SIZE,
    V10_MAX_NEW_TOKENS,
    V10_SCHEMA_VERSION,
    V10_SOURCE_VERSION,
    V10_VERSION,
    SentenceClassifier,
    V10RunConfig,
    V10RunSummary,
)

__all__ = [
    "PROMPT_SHA256",
    "V10_BATCH_SIZE",
    "V10_MAX_NEW_TOKENS",
    "V10RunConfig",
    "V10RunSummary",
    "run_v10",
]

_V10_REMOVED_OUTPUT_FIELDS = frozenset(
    (
        "relevant_sentence_metadata",
        "sentences",
        "sentences_with_topic_term",
        "text",
        "topic_categories",
        "topic_sentence_count",
        "topic_terms",
    )
)


@dataclass(frozen=True, slots=True)
class _CandidateRow:
    line_number: int
    row: dict[str, Any]
    sentences: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _PendingRow:
    candidate: _CandidateRow
    labels: tuple[str, ...] | None


@dataclass(slots=True)
class _OutputStats:
    rows_processed: int = 0
    rows_kept: int = 0
    candidate_sentences_processed: int = 0
    yes_sentences_written: int = 0
    no_sentences: int = 0

    def record_seen(self, sentence_count: int) -> None:
        self.rows_processed += 1
        self.candidate_sentences_processed += sentence_count

    def record_labels(self, labels: Sequence[str]) -> None:
        self.yes_sentences_written += sum(label == "yes" for label in labels)
        self.no_sentences += sum(label == "no" for label in labels)

    def record_row(self, kept: bool) -> None:
        if kept:
            self.rows_kept += 1


def run_v10(
    config: V10RunConfig,
    *,
    classifier: SentenceClassifier | None = None,
) -> V10RunSummary:
    """Classify V9 candidate sentences and publish only LFM-yes evidence."""
    paths = _resolve_paths(config)
    (
        input_path,
        output_path,
        manifest_path,
        model_path,
        checkpoint_path,
        runtime_model_path,
    ) = paths
    source_sha256 = _sha256_file(input_path)
    model_record = _model_record(model_path)
    runtime_model_record = _model_record(runtime_model_path)
    reusable = _load_reusable_summary(
        config=config,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        source_sha256=source_sha256,
        model_record=model_record,
        runtime_model_record=runtime_model_record,
    )
    if reusable is not None:
        return reusable

    checkpoint_header = _checkpoint_header(
        config=config,
        input_path=input_path,
        source_sha256=source_sha256,
        model_record=model_record,
        runtime_model_record=runtime_model_record,
    )
    checkpoint = _open_checkpoint(checkpoint_path, checkpoint_header)
    active_classifier = classifier or _build_classifier(config, runtime_model_path)
    stats = _OutputStats()
    _write_output(
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        classifier=active_classifier,
        batch_size=config.batch_size,
        stats=stats,
    )
    result_sha256 = _sha256_file(output_path)
    manifest = _manifest_record(
        config=config,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        source_sha256=source_sha256,
        model_record=model_record,
        runtime_model_record=runtime_model_record,
        result_sha256=result_sha256,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        stats=stats,
    )
    _atomic_json_write(manifest_path, manifest)
    return V10RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        rows_processed=stats.rows_processed,
        rows_kept=stats.rows_kept,
        rows_filtered=stats.rows_processed - stats.rows_kept,
        candidate_sentences_processed=stats.candidate_sentences_processed,
        yes_sentences_written=stats.yes_sentences_written,
        no_sentences=stats.no_sentences,
        result_sha256=result_sha256,
    )


def _resolve_paths(
    config: V10RunConfig,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    paths = _resolved_paths(config)
    _validate_distinct_paths(
        paths, runtime_is_explicit=config.runtime_model_path is not None
    )
    _validate_existing_paths(paths)
    return paths


def _resolved_paths(
    config: V10RunConfig,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    resolved = tuple(
        path.expanduser().resolve()
        for path in (
            config.input_path,
            config.output_path,
            config.manifest_path,
            config.model_path,
            config.effective_checkpoint_path,
            config.effective_runtime_model_path,
        )
    )
    return cast(tuple[Path, Path, Path, Path, Path, Path], resolved)


def _validate_distinct_paths(
    paths: tuple[Path, ...],
    *,
    runtime_is_explicit: bool,
) -> None:
    distinct_paths = paths if runtime_is_explicit else paths[:-1]
    if len(set(distinct_paths)) != len(distinct_paths):
        raise ValueError("V10 paths must be different")


def _validate_existing_paths(paths: tuple[Path, ...]) -> None:
    input_path, _, _, model_path, _, runtime_model_path = paths
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not runtime_model_path.is_dir():
        raise FileNotFoundError(runtime_model_path)


def _build_classifier(
    config: V10RunConfig,
    runtime_model_path: Path,
) -> SentenceClassifier:
    if config.runtime_model_path is None:
        return LfmSentenceClassifier(
            runtime_model_path,
            max_new_tokens=config.max_new_tokens,
        )
    return MlxSentenceClassifier(
        runtime_model_path,
        max_new_tokens=config.max_new_tokens,
    )


def _model_record(model_path: Path) -> dict[str, object]:
    files = tuple(
        {
            "path": str(path.relative_to(model_path)),
            "sha256": _sha256_file(path),
        }
        for path in sorted(model_path.rglob("*"))
        if path.is_file() and ".incomplete-" not in path.name
    )
    fingerprint = hashlib.sha256(
        json.dumps(files, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(model_path),
        "snapshot_id": model_path.name,
        "files": list(files),
        "sha256": fingerprint,
    }


def _checkpoint_header(
    *,
    config: V10RunConfig,
    input_path: Path,
    source_sha256: str,
    model_record: Mapping[str, object],
    runtime_model_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "record_type": "header",
        "schema_version": V10_SCHEMA_VERSION,
        "version": V10_VERSION,
        "source_version": V10_SOURCE_VERSION,
        "source": {"path": str(input_path), "sha256": source_sha256},
        "model": dict(model_record),
        "runtime_model": dict(runtime_model_record),
        "classification": {
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "prompt_sha256": PROMPT_SHA256,
            "assistant_prefill": REASONING_CLOSE_TAG,
        },
    }


def _open_checkpoint(
    checkpoint_path: Path,
    expected_header: Mapping[str, object],
) -> dict[int, tuple[str, ...]]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint_path.exists():
        checkpoint_path.write_text(
            json.dumps(expected_header, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {}
    with checkpoint_path.open(encoding="utf-8") as source:
        lines = source.read().splitlines()
    if not lines or json.loads(lines[0]) != expected_header:
        raise ValueError("V10 checkpoint does not match the current run")
    return _checkpoint_records(lines[1:])


def _checkpoint_records(lines: Sequence[str]) -> dict[int, tuple[str, ...]]:
    records: dict[int, tuple[str, ...]] = {}
    for line_number, line in enumerate(lines, start=2):
        if not line.strip():
            continue
        parsed = _parse_checkpoint_record(line, line_number, len(lines) + 1)
        if parsed is None:
            break
        row_number, labels = parsed
        if row_number in records:
            raise ValueError(f"V10 checkpoint repeats row {row_number}")
        records[row_number] = tuple(labels)
    return records


def _parse_checkpoint_record(
    line: str,
    line_number: int,
    final_line_number: int,
) -> tuple[int, list[str]] | None:
    record = _decode_checkpoint_json(line, line_number, final_line_number)
    if record is None:
        return None
    return _checkpoint_fields(record, line_number)


def _decode_checkpoint_json(
    line: str,
    line_number: int,
    final_line_number: int,
) -> object | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        if line_number == final_line_number:
            return None
        raise ValueError(f"V10 checkpoint line {line_number} is invalid") from error
    return record


def _checkpoint_fields(record: object, line_number: int) -> tuple[int, list[str]]:
    if not isinstance(record, dict):
        raise ValueError(f"V10 checkpoint line {line_number} must be an object")
    row_number = record.get("row_number")
    labels = record.get("labels")
    if not isinstance(row_number, int) or not _is_string_list(labels):
        raise ValueError(f"V10 checkpoint line {line_number} is invalid")
    _validate_labels(labels)
    return row_number, labels


@contextmanager
def _append_checkpoint(checkpoint_path: Path) -> Iterator[TextIO]:
    with checkpoint_path.open("a", encoding="utf-8") as output:
        yield output


def _write_output(
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint: dict[int, tuple[str, ...]],
    classifier: SentenceClassifier,
    batch_size: int,
    stats: _OutputStats,
) -> None:
    with (
        _atomic_text_output(output_path, newline="\n") as output,
        _append_checkpoint(checkpoint_path) as checkpoint_output,
    ):
        pending: list[_PendingRow] = []
        pending_size = 0
        for candidate in _read_rows(input_path):
            stats.record_seen(len(candidate.sentences))
            labels = checkpoint.get(candidate.line_number)
            pending.append(_PendingRow(candidate, labels))
            if labels is None:
                pending_size += len(candidate.sentences)
            if pending_size >= batch_size:
                _classify_pending(
                    pending,
                    classifier=classifier,
                    output=output,
                    checkpoint_output=checkpoint_output,
                    checkpoint=checkpoint,
                    stats=stats,
                )
                pending = []
                pending_size = 0
        if pending:
            _classify_pending(
                pending,
                classifier=classifier,
                output=output,
                checkpoint_output=checkpoint_output,
                checkpoint=checkpoint,
                stats=stats,
            )


def _classify_pending(
    pending: Sequence[_PendingRow],
    *,
    classifier: SentenceClassifier,
    output: TextIO,
    checkpoint_output: TextIO,
    checkpoint: dict[int, tuple[str, ...]],
    stats: _OutputStats,
) -> None:
    unknown = tuple(item for item in pending if item.labels is None)
    labels = _classify_sentences(unknown, classifier)
    _save_classifications(
        unknown,
        labels,
        checkpoint=checkpoint,
        checkpoint_output=checkpoint_output,
    )
    _write_pending_rows(pending, output=output, stats=stats)


def _classify_sentences(
    pending: Sequence[_PendingRow], classifier: SentenceClassifier
) -> tuple[str, ...]:
    sentences = tuple(
        sentence for item in pending for sentence in item.candidate.sentences
    )
    labels = tuple(classifier.classify(sentences))
    _validate_labels(labels)
    if len(labels) != len(sentences):
        raise ValueError("Classifier returned a label count different from its input")
    return labels


def _save_classifications(
    pending: Sequence[_PendingRow],
    labels: Sequence[str],
    *,
    checkpoint: dict[int, tuple[str, ...]],
    checkpoint_output: TextIO,
) -> None:
    offset = 0
    for item in pending:
        row_labels = tuple(labels[offset : offset + len(item.candidate.sentences)])
        offset += len(item.candidate.sentences)
        item.labels = row_labels
        checkpoint[item.candidate.line_number] = row_labels
        _write_json_line(
            checkpoint_output,
            {"labels": list(row_labels), "row_number": item.candidate.line_number},
        )
        checkpoint_output.flush()


def _write_pending_rows(
    pending: Sequence[_PendingRow], *, output: TextIO, stats: _OutputStats
) -> None:
    for item in pending:
        if item.labels is None:
            raise RuntimeError("V10 pending row has no classification")
        _write_classified_row(output, item.candidate, item.labels, stats)


def _write_classified_row(
    output: TextIO,
    candidate: _CandidateRow,
    labels: Sequence[str],
    stats: _OutputStats,
) -> None:
    stats.record_labels(labels)
    selected = _selected_row(candidate, labels)
    stats.record_row(selected is not None)
    if selected is not None:
        _write_json_line(output, selected)


def _selected_row(
    candidate: _CandidateRow,
    labels: Sequence[str],
) -> dict[str, object] | None:
    yes_indices = _yes_indices(labels)
    if not yes_indices:
        return None
    output = {
        key: value
        for key, value in candidate.row.items()
        if key not in _V10_REMOVED_OUTPUT_FIELDS
    }
    metadata = [candidate.metadata[index] for index in yes_indices]
    output.update(_selected_evidence(candidate, yes_indices, metadata))
    return output


def _yes_indices(labels: Sequence[str]) -> tuple[int, ...]:
    return tuple(index for index, label in enumerate(labels) if label == "yes")


def _selected_evidence(
    candidate: _CandidateRow,
    yes_indices: Sequence[int],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    return {
        "relevant_sentence_metadata": list(metadata),
        "sentences_with_topic_term": [
            candidate.sentences[index] for index in yes_indices
        ],
        "topic_categories": list(_metadata_values(metadata, "topic_categories")),
        "topic_sentence_count": len(yes_indices),
        "topic_terms": list(_metadata_values(metadata, "topic_terms")),
    }


def _metadata_values(
    metadata: Sequence[Mapping[str, Any]], key: str
) -> tuple[str, ...]:
    values: list[str] = []
    for item in metadata:
        value = item.get(key, [])
        if not _is_string_list(value):
            raise ValueError(f"V9 metadata field {key} must be a list of strings")
        values.extend(value)
    return tuple(dict.fromkeys(values))


def _read_rows(input_path: Path) -> Iterator[_CandidateRow]:
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            yield _decode_input_line(line, line_number)


def _decode_input_line(line: str, line_number: int) -> _CandidateRow:
    if not line.strip():
        raise ValueError(f"V9 JSONL line {line_number} is empty")
    decoded = _decode_object(line, line_number)
    sentences, metadata = _candidate_fields(decoded, line_number)
    return _CandidateRow(
        line_number=line_number,
        row=decoded,
        sentences=tuple(sentences),
        metadata=tuple(dict(item) for item in metadata),
    )


def _decode_object(line: str, line_number: int) -> dict[str, Any]:
    decoded = json.loads(line)
    if not isinstance(decoded, dict):
        raise ValueError(f"V9 JSONL line {line_number} must be an object")
    return decoded


def _candidate_fields(
    decoded: Mapping[str, Any], line_number: int
) -> tuple[list[str], list[Mapping[str, Any]]]:
    sentences = decoded.get("sentences_with_topic_term")
    metadata = decoded.get("relevant_sentence_metadata")
    if not _is_string_list(sentences):
        raise ValueError(
            f"V9 JSONL line {line_number} must contain a list of candidate sentences"
        )
    if not _is_mapping_list(metadata) or len(metadata) != len(sentences):
        raise ValueError(
            f"V9 JSONL line {line_number} metadata must align with candidate sentences"
        )
    return sentences, metadata


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _is_mapping_list(value: object) -> TypeGuard[list[Mapping[str, Any]]]:
    return isinstance(value, list) and all(isinstance(item, Mapping) for item in value)


def _validate_labels(labels: Sequence[str]) -> None:
    if any(label not in {"yes", "no"} for label in labels):
        raise ValueError("Classifier labels must be exactly yes or no")


def _manifest_record(
    *,
    config: V10RunConfig,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    source_sha256: str,
    model_record: Mapping[str, object],
    runtime_model_record: Mapping[str, object],
    result_sha256: str,
    checkpoint_sha256: str,
    stats: _OutputStats,
) -> dict[str, object]:
    return {
        "schema_version": V10_SCHEMA_VERSION,
        "version": V10_VERSION,
        "source_version": V10_SOURCE_VERSION,
        "status": "complete",
        "source": {"path": str(input_path), "sha256": source_sha256},
        "result": {"path": str(output_path), "sha256": result_sha256},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "classification": {
            "candidate_field": "sentences_with_topic_term",
            "output_field": "sentences_with_topic_term",
            "kept_label": "yes",
            "discarded_label": "no",
            "label_contract": "exact lowercase yes or no",
            "prompt_template": V10_PROMPT_TEMPLATE,
            "prompt_sha256": PROMPT_SHA256,
            "assistant_prefill": REASONING_CLOSE_TAG,
            "batch_size": config.batch_size,
            "max_new_tokens": config.max_new_tokens,
            "model": dict(model_record),
            "runtime_model": dict(runtime_model_record),
        },
        "rows_processed": stats.rows_processed,
        "rows_kept": stats.rows_kept,
        "rows_filtered": stats.rows_processed - stats.rows_kept,
        "candidate_sentences_processed": stats.candidate_sentences_processed,
        "yes_sentences_written": stats.yes_sentences_written,
        "no_sentences": stats.no_sentences,
    }


def _load_reusable_summary(
    *,
    config: V10RunConfig,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    source_sha256: str,
    model_record: Mapping[str, object],
    runtime_model_record: Mapping[str, object],
) -> V10RunSummary | None:
    reusable = _reusable_manifest(
        manifest_path,
        config=config,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        source_sha256=source_sha256,
        model_record=model_record,
        runtime_model_record=runtime_model_record,
    )
    if reusable is None:
        return None
    manifest, result_sha256 = reusable
    counts = _summary_counts(manifest)
    if counts is None:
        return None
    return V10RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        rows_processed=counts[0],
        rows_kept=counts[1],
        rows_filtered=counts[2],
        candidate_sentences_processed=counts[3],
        yes_sentences_written=counts[4],
        no_sentences=counts[5],
        result_sha256=result_sha256,
    )


def _reusable_manifest(
    manifest_path: Path,
    *,
    output_path: Path,
    config: V10RunConfig,
    input_path: Path,
    checkpoint_path: Path,
    source_sha256: str,
    model_record: Mapping[str, object],
    runtime_model_record: Mapping[str, object],
) -> tuple[Mapping[str, object], str] | None:
    if not output_path.is_file() or not manifest_path.is_file():
        return None
    manifest = _read_manifest(manifest_path)
    if manifest is None:
        return None
    return _matching_reusable_result(
        manifest,
        config=config,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        source_sha256=source_sha256,
        model_record=model_record,
        runtime_model_record=runtime_model_record,
    )


def _matching_reusable_result(
    manifest: Mapping[str, object],
    *,
    output_path: Path,
    config: V10RunConfig,
    input_path: Path,
    checkpoint_path: Path,
    source_sha256: str,
    model_record: Mapping[str, object],
    runtime_model_record: Mapping[str, object],
) -> tuple[Mapping[str, object], str] | None:
    if not _manifest_matches(
        manifest,
        config=config,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        source_sha256=source_sha256,
        model_record=model_record,
        runtime_model_record=runtime_model_record,
    ):
        return None
    result_sha256 = _nested_string(manifest, "result", "sha256")
    if _sha256_file(output_path) != result_sha256:
        return None
    return manifest, result_sha256


def _manifest_matches(
    manifest: Mapping[str, object],
    *,
    config: V10RunConfig,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    source_sha256: str,
    model_record: Mapping[str, object],
    runtime_model_record: Mapping[str, object],
) -> bool:
    source = manifest.get("source")
    result = manifest.get("result")
    checkpoint = manifest.get("checkpoint")
    classification = manifest.get("classification")
    records = _manifest_records(source, result, checkpoint, classification)
    if records is None:
        return False
    source, result, checkpoint, classification = records
    return all(
        (
            manifest.get("schema_version") == V10_SCHEMA_VERSION,
            manifest.get("version") == V10_VERSION,
            manifest.get("source_version") == V10_SOURCE_VERSION,
            manifest.get("status") == "complete",
            source.get("path") == str(input_path),
            source.get("sha256") == source_sha256,
            result.get("path") == str(output_path),
            checkpoint.get("path") == str(checkpoint_path),
            classification.get("candidate_field") == "sentences_with_topic_term",
            classification.get("output_field") == "sentences_with_topic_term",
            classification.get("prompt_sha256") == PROMPT_SHA256,
            classification.get("assistant_prefill") == REASONING_CLOSE_TAG,
            classification.get("batch_size") == config.batch_size,
            classification.get("max_new_tokens") == config.max_new_tokens,
            classification.get("model") == model_record,
            classification.get("runtime_model") == runtime_model_record,
        )
    )


def _manifest_records(
    *records: object,
) -> tuple[Mapping[str, object], ...] | None:
    if not all(isinstance(record, Mapping) for record in records):
        return None
    return cast(tuple[Mapping[str, object], ...], records)


def _summary_counts(
    manifest: Mapping[str, object],
) -> tuple[int, int, int, int, int, int] | None:
    keys = (
        "rows_processed",
        "rows_kept",
        "rows_filtered",
        "candidate_sentences_processed",
        "yes_sentences_written",
        "no_sentences",
    )
    values = tuple(manifest.get(key) for key in keys)
    if not all(isinstance(value, int) for value in values):
        return None
    return cast(tuple[int, int, int, int, int, int], values)


def _nested_string(manifest: Mapping[str, object], key: str, nested_key: str) -> str:
    nested = manifest.get(key)
    if not isinstance(nested, Mapping) or not isinstance(nested.get(nested_key), str):
        raise ValueError("V10 manifest has an invalid result record")
    return nested[nested_key]
