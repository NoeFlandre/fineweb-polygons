"""Value objects used by the V9 sentence-topic post-processing stage."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

V9_CONTEXT_WINDOW = 2


@dataclass(frozen=True, slots=True)
class V9RunConfig:
    """Inputs and immutable settings for one V9 artifact."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    vocabulary_path: Path
    context_window: int = V9_CONTEXT_WINDOW

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
            raise ValueError("V9 paths must be different")
        if self.context_window < 0:
            raise ValueError("context_window must not be negative")


@dataclass(frozen=True, slots=True)
class V9RunSummary:
    """Stable result of a complete or reused V9 run."""

    output_path: Path
    manifest_path: Path
    rows_processed: int
    rows_kept: int
    rows_filtered: int
    sentences_processed: int
    relevant_sentences_written: int
    category_sentences: dict[str, int]
    result_sha256: str


@dataclass(slots=True)
class _OutputStats:
    rows_processed: int = 0
    rows_kept: int = 0
    sentences_processed: int = 0
    relevant_sentences_written: int = 0
    category_sentences: Counter[str] = field(default_factory=Counter)

    def record_seen(self, sentence_count: int) -> None:
        self.rows_processed += 1
        self.sentences_processed += sentence_count

    def record_kept(self, relevant: Sequence[Mapping[str, object]]) -> None:
        self.rows_kept += 1
        self.relevant_sentences_written += len(relevant)
        _add_category_sentences(self.category_sentences, relevant)

    def as_tuple(self) -> tuple[int, int, int, int, dict[str, int]]:
        return (
            self.rows_processed,
            self.rows_kept,
            self.sentences_processed,
            self.relevant_sentences_written,
            dict(sorted(self.category_sentences.items())),
        )


def _add_category_sentences(
    counts: Counter[str], evidence: Sequence[Mapping[str, object]]
) -> None:
    for item in evidence:
        for category in _string_values(item["topic_categories"]):
            counts[category] += 1


def _string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError("V9 evidence values must be lists of strings")
    return tuple(value)
