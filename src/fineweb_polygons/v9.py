"""V9 post-processing: keep topic-relevant sentences near polygon evidence."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeGuard, cast

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
    write_json_line as _write_kept_row,
)
from fineweb_polygons.matching import _MultiPatternMatcher
from fineweb_polygons.normalization import normalize_for_search
from fineweb_polygons.segmentation import validate_segments
from fineweb_polygons.topic_vocabulary import TopicVocabulary, load_vocabulary
from fineweb_polygons.v9_models import (
    V9_CONTEXT_WINDOW,
    V9RunConfig,
    V9RunSummary,
    _add_category_sentences,
    _OutputStats,
    _string_values,
)

__all__ = [
    "V9_CONTEXT_WINDOW",
    "V9RunConfig",
    "V9RunSummary",
    "_OutputStats",
    "_add_category_sentences",
    "_string_values",
]

V9_VERSION = "v9"
V9_SOURCE_VERSION = "v8"
V9_SCHEMA_VERSION = 3

_V9_REMOVED_OUTPUT_FIELDS = frozenset(
    (
        "context_fields",
        "context_phrase",
        "country_name_sentence",
        "matched_fields",
        "matched_name",
        "polygon_name_sentence",
    )
)


def run_v9(
    config: V9RunConfig,
    *,
    vocabulary: TopicVocabulary | None = None,
) -> V9RunSummary:
    """Filter V8 rows to local topic sentences and publish them atomically."""
    input_path, output_path, manifest_path, vocabulary_path = _resolve_paths(config)
    source_sha256 = _sha256_file(input_path)
    vocabulary_sha256 = _sha256_file(vocabulary_path)
    reusable = _load_reusable_summary(
        config=config,
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
    (
        rows_processed,
        rows_kept,
        sentences_processed,
        relevant_sentences_written,
        category_sentences,
    ) = _write_output(
        input_path=input_path,
        output_path=output_path,
        vocabulary=active_vocabulary,
        context_window=config.context_window,
    )
    rows_filtered = rows_processed - rows_kept
    result_sha256 = _sha256_file(output_path)
    manifest = _manifest_record(
        config=config,
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
        sentences_processed=sentences_processed,
        relevant_sentences_written=relevant_sentences_written,
        category_sentences=category_sentences,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_write(manifest_path, manifest)
    return V9RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_processed=rows_processed,
        rows_kept=rows_kept,
        rows_filtered=rows_filtered,
        sentences_processed=sentences_processed,
        relevant_sentences_written=relevant_sentences_written,
        category_sentences=category_sentences,
        result_sha256=result_sha256,
    )


def _resolve_paths(
    config: V9RunConfig,
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
    context_window: int,
) -> tuple[int, int, int, int, dict[str, int]]:
    stats = _OutputStats()
    with _atomic_text_output(output_path, newline="\n") as output:
        for row, sentences in _read_rows(input_path):
            stats.record_seen(len(sentences))
            output_row = _enriched_row(
                row,
                sentences,
                vocabulary=vocabulary,
                context_window=context_window,
            )
            if output_row is None:
                continue
            _write_kept_row(output, output_row)
            # pragma: no mutate start
            relevant = cast(
                Sequence[Mapping[str, object]],
                output_row["relevant_sentence_metadata"],
            )
            # pragma: no mutate end
            stats.record_kept(relevant)
    return stats.as_tuple()


def _enriched_row(
    row: Mapping[str, Any],
    sentences: tuple[str, ...],
    *,
    vocabulary: TopicVocabulary,
    context_window: int,
) -> dict[str, object] | None:
    relevant = _relevant_sentences(
        row,
        sentences,
        vocabulary=vocabulary,
        context_window=context_window,
    )
    if not relevant:
        return None
    output_row = {
        key: value for key, value in row.items() if key not in _V9_REMOVED_OUTPUT_FIELDS
    }
    output_row.update(_sentence_columns(relevant))
    output_row["topic_sentence_count"] = len(relevant)
    output_row["topic_terms"] = list(_row_terms(relevant))
    output_row["topic_categories"] = list(_row_categories(relevant))
    return output_row


def _sentence_columns(
    relevant: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "relevant_sentences": [evidence["sentence"] for evidence in relevant],
        "relevant_sentence_metadata": [
            {key: value for key, value in evidence.items() if key != "sentence"}
            for evidence in relevant
        ],
    }


def _row_terms(relevant: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return _ordered_unique(
        term
        for evidence in relevant
        for term in _string_values(evidence["topic_terms"])
    )


def _row_categories(relevant: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return _ordered_unique(
        category
        for evidence in relevant
        for category in _string_values(evidence["topic_categories"])
    )


def _relevant_sentences(
    row: Mapping[str, Any],
    sentences: tuple[str, ...],
    *,
    vocabulary: TopicVocabulary,
    context_window: int,
) -> tuple[dict[str, object], ...]:
    polygon_indices, country_indices = _place_anchor_indices(row, sentences)
    return tuple(
        evidence
        for sentence_index, sentence in enumerate(sentences)
        if (
            evidence := _sentence_evidence(
                sentence_index,
                sentence,
                polygon_indices=polygon_indices,
                country_indices=country_indices,
                vocabulary=vocabulary,
                context_window=context_window,
            )
        )
        is not None
    )


def _place_anchor_indices(
    row: Mapping[str, Any], sentences: Sequence[str]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    polygon_name = _required_row_string(row, "polygon_name")
    context_phrase = _required_row_string(row, "context_phrase")
    polygon_indices = _anchor_indices(sentences, _exact_matcher(polygon_name))
    country_indices = _anchor_indices(sentences, _exact_matcher(context_phrase))
    if not polygon_indices:
        raise ValueError("polygon_name does not occur in the V8 sentence list")
    if not country_indices:
        raise ValueError("context_phrase does not occur in the V8 sentence list")
    return polygon_indices, country_indices


def _sentence_evidence(
    sentence_index: int,
    sentence: str,
    *,
    polygon_indices: Sequence[int],
    country_indices: Sequence[int],
    vocabulary: TopicVocabulary,
    context_window: int,
) -> dict[str, object] | None:
    matches = vocabulary.match_text(sentence)
    if not matches:
        return None
    polygon_distance = _nearest_distance(sentence_index, polygon_indices)
    if not _within_context_window(polygon_distance, context_window):
        return None
    country_distance = _nearest_distance(sentence_index, country_indices)
    return _sentence_evidence_record(
        sentence_index,
        sentence,
        matches=matches,
        polygon_distance=polygon_distance,
        country_distance=country_distance,
    )


def _within_context_window(distance: int | None, context_window: int) -> bool:
    return distance is not None and distance <= context_window


def _sentence_evidence_record(
    sentence_index: int,
    sentence: str,
    *,
    matches: Sequence[Any],
    polygon_distance: int | None,
    country_distance: int | None,
) -> dict[str, object]:
    if polygon_distance is None:
        raise ValueError("polygon evidence distance must be available")
    categories = _ordered_unique(match.category for match in matches)
    terms = _ordered_unique(match.term for match in matches)
    place_distance = _place_distance(polygon_distance, country_distance)
    return {
        "country_sentence_distance": country_distance,
        "place_sentence_distance": place_distance,
        "polygon_sentence_distance": polygon_distance,
        "sentence": sentence,
        "sentence_index": sentence_index,
        "topic_categories": list(categories),
        "topic_category_count": len(categories),
        "topic_term_count": len(terms),
        "topic_terms": list(terms),
    }


def _place_distance(polygon_distance: int, country_distance: int | None) -> int:
    if country_distance is None:
        return polygon_distance
    return min(polygon_distance, country_distance)


def _required_row_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V8 row must contain a non-empty string {key}")
    return value


def _exact_matcher(value: str) -> _MultiPatternMatcher:
    normalized = normalize_for_search(value, decode_url=False)
    if not normalized:
        raise ValueError("place evidence must contain searchable characters")
    return _MultiPatternMatcher((normalized,))


def _anchor_indices(
    sentences: Sequence[str], matcher: _MultiPatternMatcher
) -> tuple[int, ...]:
    return tuple(
        index
        for index, sentence in enumerate(sentences)
        if matcher.find(sentence, decode_url=False)
    )


def _nearest_distance(index: int, anchors: Sequence[int]) -> int | None:
    if not anchors:
        return None
    return min(abs(index - anchor) for anchor in anchors)


def _ordered_unique(values: Iterator[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _read_rows(
    input_path: Path,
) -> Iterator[tuple[dict[str, Any], tuple[str, ...]]]:
    with _open_text_input(input_path) as source:
        for line_number, line in enumerate(source, start=1):
            yield _decode_input_line(line, line_number)


def _decode_input_line(
    line: str, line_number: int
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not line.strip():
        raise ValueError(f"V8 JSONL line {line_number} is empty")
    decoded = _decode_object(line, line_number)
    text = _decode_text(decoded, line_number)
    sentences = _decode_sentences(decoded, text, line_number)
    return decoded, sentences


def _decode_object(line: str, line_number: int) -> dict[str, Any]:
    decoded = json.loads(line)
    if not isinstance(decoded, dict):
        raise ValueError(f"V8 JSONL line {line_number} must be an object")
    return decoded


def _decode_text(decoded: Mapping[str, Any], line_number: int) -> str:
    text = decoded.get("text")
    if not isinstance(text, str):
        raise ValueError(
            f"V8 JSONL line {line_number} must contain a string text field"
        )
    return text


def _decode_sentences(
    decoded: Mapping[str, Any], text: str, line_number: int
) -> tuple[str, ...]:
    raw_sentences = decoded.get("sentences")
    if not _is_string_list(raw_sentences):
        raise ValueError(
            f"V8 JSONL line {line_number} must contain a list of string sentences"
        )
    try:
        sentences = validate_segments(text, raw_sentences)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"V8 JSONL line {line_number} sentences must reconstruct the input text"
        ) from error
    return sentences


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _manifest_record(
    *,
    config: V9RunConfig,
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
    sentences_processed: int,
    relevant_sentences_written: int,
    category_sentences: Mapping[str, int],
) -> dict[str, object]:
    return {
        "schema_version": V9_SCHEMA_VERSION,
        "version": V9_VERSION,
        "source_version": V9_SOURCE_VERSION,
        "status": "complete",
        "context_window": config.context_window,
        "matching": {
            "place_anchor": "polygon_name in exact normalized sentence text",
            "topic_anchor": "V8 vocabulary in exact normalized sentence text",
            "country_anchor": "context_phrase in exact normalized sentence text",
        },
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
        "sentences_processed": sentences_processed,
        "relevant_sentences_written": relevant_sentences_written,
        "category_sentences": dict(sorted(category_sentences.items())),
    }


def _load_reusable_summary(
    *,
    config: V9RunConfig,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    vocabulary_path: Path,
    source_sha256: str,
    vocabulary_sha256: str,
) -> V9RunSummary | None:
    manifest = _load_reusable_manifest(
        config=config,
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        vocabulary_path=vocabulary_path,
        source_sha256=source_sha256,
        vocabulary_sha256=vocabulary_sha256,
    )
    if manifest is None:
        return None
    counts = _manifest_counts(manifest)
    if counts is None:
        return None
    (
        rows_processed,
        rows_kept,
        rows_filtered,
        sentences_processed,
        relevant_sentences_written,
        category_sentences,
    ) = counts
    result = manifest["result"]
    assert isinstance(result, Mapping)
    result_sha256 = result["sha256"]
    assert isinstance(result_sha256, str)
    return V9RunSummary(
        output_path=output_path,
        manifest_path=manifest_path,
        rows_processed=rows_processed,
        rows_kept=rows_kept,
        rows_filtered=rows_filtered,
        sentences_processed=sentences_processed,
        relevant_sentences_written=relevant_sentences_written,
        category_sentences=category_sentences,
        result_sha256=result_sha256,
    )


def _load_reusable_manifest(
    *,
    config: V9RunConfig,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
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
        config=config,
        input_path=input_path,
        output_path=output_path,
        vocabulary_path=vocabulary_path,
        source_sha256=source_sha256,
        vocabulary_sha256=vocabulary_sha256,
    ):
        return None
    return _with_valid_result_hash(manifest, output_path)


def _manifest_matches(
    manifest: Mapping[str, object],
    *,
    config: V9RunConfig,
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
            manifest.get("schema_version") == V9_SCHEMA_VERSION,
            manifest.get("version") == V9_VERSION,
            manifest.get("source_version") == V9_SOURCE_VERSION,
            manifest.get("status") == "complete",
            manifest.get("context_window") == config.context_window,
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
) -> tuple[int, int, int, int, int, dict[str, int]] | None:
    integer_values = _manifest_integer_values(manifest)
    if integer_values is None:
        return None
    categories = _manifest_categories(manifest)
    if categories is None:
        return None
    return (*integer_values, categories)


def _manifest_integer_values(
    manifest: Mapping[str, object],
) -> tuple[int, int, int, int, int] | None:
    values: list[int] = []
    for key in (
        "rows_processed",
        "rows_kept",
        "rows_filtered",
        "sentences_processed",
        "relevant_sentences_written",
    ):
        value = manifest.get(key)
        if not isinstance(value, int):
            return None
        values.append(value)
    return values[0], values[1], values[2], values[3], values[4]


def _manifest_categories(
    manifest: Mapping[str, object],
) -> dict[str, int] | None:
    category_sentences = manifest.get("category_sentences")
    if not isinstance(category_sentences, Mapping):
        return None
    categories: dict[str, int] = {}
    for category, count in category_sentences.items():
        if not isinstance(category, str) or not isinstance(count, int):
            return None
        categories[category] = count
    return categories


def _open_text_input(path: Path) -> Any:
    # pragma: no mutate start
    return path.open(encoding="utf-8")
    # pragma: no mutate end
