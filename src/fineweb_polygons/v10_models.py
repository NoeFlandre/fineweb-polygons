"""Value objects shared by the V10 sentence-classification stage."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

V10_VERSION = "v10"
V10_SOURCE_VERSION = "v9"
V10_SCHEMA_VERSION = 1
V10_BATCH_SIZE = 8
V10_MAX_NEW_TOKENS = 4


class SentenceClassifier(Protocol):
    """Classify a non-empty batch of sentences as exact yes/no labels."""

    def classify(self, sentences: Sequence[str]) -> Sequence[str]:
        """Return one lowercase label for every input sentence."""


@dataclass(frozen=True, slots=True)
class V10RunConfig:
    """Inputs and immutable settings for one V10 artifact."""

    input_path: Path
    output_path: Path
    manifest_path: Path
    model_path: Path
    checkpoint_path: Path | None = None
    runtime_model_path: Path | None = None
    batch_size: int = V10_BATCH_SIZE
    max_new_tokens: int = V10_MAX_NEW_TOKENS

    def __post_init__(self) -> None:
        paths = tuple(
            path.expanduser().resolve()
            for path in (
                self.input_path,
                self.output_path,
                self.manifest_path,
                self.model_path,
                self.checkpoint_path,
                self.runtime_model_path,
            )
            if path is not None
        )
        if len(set(paths)) != len(paths):
            raise ValueError("V10 paths must be different")
        _validate_positive("batch_size", self.batch_size)
        _validate_positive("max_new_tokens", self.max_new_tokens)

    @property
    def effective_checkpoint_path(self) -> Path:
        """Return the checkpoint path used when none was supplied."""
        if self.checkpoint_path is not None:
            return self.checkpoint_path
        return self.manifest_path.with_name("classifications.jsonl")

    @property
    def effective_runtime_model_path(self) -> Path:
        """Return the inference runtime, defaulting to the source model."""
        if self.runtime_model_path is not None:
            return self.runtime_model_path
        return self.model_path


@dataclass(frozen=True, slots=True)
class V10RunSummary:
    """Stable result of a complete or reused V10 run."""

    output_path: Path
    manifest_path: Path
    checkpoint_path: Path
    rows_processed: int
    rows_kept: int
    rows_filtered: int
    candidate_sentences_processed: int
    yes_sentences_written: int
    no_sentences: int
    result_sha256: str


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")
