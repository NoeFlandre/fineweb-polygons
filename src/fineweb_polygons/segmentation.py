"""Exact sentence segmentation for V7."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from fineweb_polygons.foundation import DEFAULT_DATA_ROOT

_COREML_PROVIDER = "CoreMLExecutionProvider"
_CPU_PROVIDER = "CPUExecutionProvider"


class _SaTModel(Protocol):
    def split(
        self, text_or_texts: Sequence[str], /, **kwargs: object
    ) -> Sequence[Sequence[str]]: ...


class SentenceSegmenter(Protocol):
    """The batch interface required by the V7 post-processing run."""

    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class SentenceSegmentationConfig:
    """Stable SaT settings used to create one V7 artifact."""

    model_id: str = "sat-3l-sm"
    stride: int = 64
    block_size: int = 512
    batch_size: int = 32
    split_on_input_newlines: bool = False
    strip_whitespace: bool = False

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def to_record(self) -> dict[str, object]:
        """Return the settings that affect sentence boundaries."""
        return {
            "batch_size": self.batch_size,
            "block_size": self.block_size,
            "model_id": self.model_id,
            "split_on_input_newlines": self.split_on_input_newlines,
            "strip_whitespace": self.strip_whitespace,
            "stride": self.stride,
        }


def validate_segments(text: str, segments: Sequence[str]) -> tuple[str, ...]:
    """Require a segmentation to preserve the source text byte-for-byte."""
    result = tuple(segments)
    if not all(isinstance(segment, str) for segment in result):
        raise TypeError("sentence segments must be strings")
    if "".join(result) != text:
        raise ValueError(
            "sentence segments do not reconstruct the input text; "
            "the splitter must not rewrite characters"
        )
    return result


def select_onnx_providers(available: Sequence[str]) -> tuple[str, ...]:
    """Select the best available Apple provider with a CPU fallback."""
    available_set = set(available)
    selected = tuple(
        provider
        for provider in (_COREML_PROVIDER, _CPU_PROVIDER)
        if provider in available_set
    )
    if not selected:
        raise RuntimeError(
            "ONNX Runtime must provide CPUExecutionProvider or CoreMLExecutionProvider"
        )
    return selected


class SaTSentenceSegmenter:
    """Batch wrapper around ``wtpsplit``'s ``sat-3l-sm`` model."""

    def __init__(
        self,
        *,
        config: SentenceSegmentationConfig | None = None,
        model: _SaTModel | None = None,
        providers: Sequence[str] | None = None,
    ) -> None:
        self.config = config or SentenceSegmentationConfig()
        self.providers = self._resolve_providers(providers)
        self._model = model or self._load_model()

    def split_many(self, texts: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
        """Split a batch while retaining every original character."""
        if not texts:
            return ()
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be strings")
        return _validate_batch(texts, self._split_batch(texts))

    def configuration_record(self) -> dict[str, object]:
        """Return model, provider, and segmentation settings for a manifest."""
        record = self.config.to_record()
        record["providers"] = list(self.providers)
        return record

    def _resolve_providers(self, providers: Sequence[str] | None) -> tuple[str, ...]:
        if providers is not None:
            selected = tuple(providers)
            if not selected:
                raise ValueError("providers must not be empty")
            return selected
        import onnxruntime as ort

        return select_onnx_providers(ort.get_available_providers())

    def _load_model(self) -> _SaTModel:
        from wtpsplit import SaT

        return cast(
            _SaTModel,
            SaT(
                self.config.model_id,
                ort_providers=list(self.providers),
                from_pretrained_kwargs={"cache_dir": str(_model_cache_dir())},
            ),
        )

    def _split_batch(self, texts: tuple[str, ...]) -> tuple[Sequence[str], ...]:
        raw_results = self._model.split(
            texts,
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            split_on_input_newlines=self.config.split_on_input_newlines,
            strip_whitespace=self.config.strip_whitespace,
            stride=self.config.stride,
        )
        return tuple(raw_results)


def _model_cache_dir() -> Path:
    """Return an explicit external cache, never a user-home cache."""
    configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_DATA_ROOT / "cache" / "huggingface"


def _validate_batch(
    texts: tuple[str, ...], raw_results: tuple[Sequence[str], ...]
) -> tuple[tuple[str, ...], ...]:
    if len(raw_results) != len(texts):
        raise ValueError(
            "sentence splitter returned "
            f"{len(raw_results)} sentence lists for {len(texts)} texts"
        )
    return tuple(
        validate_segments(text, raw_segments)
        for text, raw_segments in zip(texts, raw_results, strict=True)
    )
