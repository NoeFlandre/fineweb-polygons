from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import fineweb_polygons.segmentation as segmentation_module
from fineweb_polygons.segmentation import (
    SaTSentenceSegmenter,
    SentenceSegmentationConfig,
    select_onnx_providers,
    validate_segments,
)


class FakeSaT:
    def __init__(self, results: Sequence[Sequence[str]]) -> None:
        self.results = tuple(tuple(result) for result in results)
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def split(self, texts: Sequence[str], **kwargs: object):
        self.calls.append((tuple(texts), kwargs))
        return self.results


@pytest.fixture(autouse=True)
def _stub_native_onnxruntime_for_unit_tests(monkeypatch) -> None:
    """Keep mutation tests away from native ONNX Runtime initialization."""
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: ("UnsupportedExecutionProvider",)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "wtpsplit",
        SimpleNamespace(SaT=lambda *args, **kwargs: FakeSaT(())),
    )


def test_validate_segments_preserves_original_order_and_text() -> None:
    text = "First sentence.\nSecond sentence."
    segments = ("First sentence.", "\nSecond sentence.")

    assert validate_segments(text, segments) == segments
    assert "".join(segments) == text


def test_validate_segments_rejects_output_that_changes_text() -> None:
    with pytest.raises(ValueError) as error:
        validate_segments("Original.", ("Changed.",))
    assert str(error.value) == (
        "sentence segments do not reconstruct the input text; "
        "the splitter must not rewrite characters"
    )


def test_validate_segments_rejects_non_string_segments() -> None:
    segments = cast(Sequence[str], ("Original.", 3))
    with pytest.raises(TypeError) as error:
        validate_segments("Original.", segments)
    assert str(error.value) == "sentence segments must be strings"


def test_segmentation_config_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id must not be empty"):
        SentenceSegmentationConfig(model_id="")


def test_segmentation_config_rejects_non_positive_settings() -> None:
    with pytest.raises(ValueError, match="stride must be positive"):
        SentenceSegmentationConfig(stride=0)
    with pytest.raises(ValueError, match="block_size must be positive"):
        SentenceSegmentationConfig(block_size=0)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        SentenceSegmentationConfig(batch_size=0)


def test_model_cache_dir_always_resolves_to_the_configured_external_cache(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_HUB_CACHE", "/Volumes/Seagate M3/model-cache")
    monkeypatch.setenv("HF_HOME", "/Users/should-not-be-used")
    assert segmentation_module._model_cache_dir() == Path(
        "/Volumes/Seagate M3/model-cache"
    )

    monkeypatch.delenv("HF_HUB_CACHE")
    assert segmentation_module._model_cache_dir() == Path("/Users/should-not-be-used")


def test_model_cache_dir_defaults_to_the_project_volume(monkeypatch) -> None:
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)

    assert segmentation_module._model_cache_dir() == Path(
        "/Volumes/Seagate M3/projects/fineweb-polygons/cache/huggingface"
    )


def test_select_onnx_providers_prefers_coreml_then_cpu() -> None:
    assert select_onnx_providers(
        ("CPUExecutionProvider", "CoreMLExecutionProvider")
    ) == ("CoreMLExecutionProvider", "CPUExecutionProvider")


def test_select_onnx_providers_requires_a_supported_provider() -> None:
    with pytest.raises(RuntimeError) as error:
        select_onnx_providers(("CUDAExecutionProvider",))
    assert str(error.value) == (
        "ONNX Runtime must provide CPUExecutionProvider or CoreMLExecutionProvider"
    )


def test_segmenter_rejects_empty_explicit_provider_list() -> None:
    with pytest.raises(ValueError) as error:
        SaTSentenceSegmenter(model=FakeSaT(()), providers=())
    assert str(error.value) == "providers must not be empty"


def test_segmenter_records_exact_reconstruction_settings_and_batches() -> None:
    model = FakeSaT((("First. ", "Second."),))
    config = SentenceSegmentationConfig(batch_size=7)
    segmenter = SaTSentenceSegmenter(
        config=config,
        model=model,
        providers=("CPUExecutionProvider",),
    )

    assert segmenter.split_many(("First. Second.",)) == (("First. ", "Second."),)
    assert model.calls == [
        (
            ("First. Second.",),
            {
                "batch_size": 7,
                "block_size": 512,
                "split_on_input_newlines": False,
                "strip_whitespace": False,
                "stride": 64,
            },
        )
    ]
    assert segmenter.configuration_record() == {
        "batch_size": 7,
        "block_size": 512,
        "model_id": "sat-3l-sm",
        "providers": ["CPUExecutionProvider"],
        "split_on_input_newlines": False,
        "strip_whitespace": False,
        "stride": 64,
    }


def test_segmenter_returns_no_batches_for_empty_input() -> None:
    model = FakeSaT(())
    segmenter = SaTSentenceSegmenter(model=model, providers=("CPUExecutionProvider",))

    assert segmenter.providers == ("CPUExecutionProvider",)
    assert segmenter._model is model
    assert segmenter.split_many(()) == ()
    assert model.calls == []


def test_segmenter_rejects_non_string_texts() -> None:
    model = FakeSaT(())
    segmenter = SaTSentenceSegmenter(model=model, providers=("CPUExecutionProvider",))

    with pytest.raises(TypeError) as error:
        segmenter.split_many(cast(tuple[str, ...], (3,)))
    assert str(error.value) == "texts must be strings"


def test_segmenter_discovers_available_onnx_providers(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(
            get_available_providers=lambda: (
                "CPUExecutionProvider",
                "CoreMLExecutionProvider",
            )
        ),
    )
    segmenter = SaTSentenceSegmenter(model=FakeSaT(()))

    assert segmenter.providers == (
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    )


def test_segmenter_passes_the_external_cache_to_model_loader(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_sat(model_id: str, **kwargs: object) -> FakeSaT:
        captured["model_id"] = model_id
        captured["kwargs"] = kwargs
        return FakeSaT(())

    monkeypatch.setenv("HF_HUB_CACHE", "/Volumes/Seagate M3/cache")
    monkeypatch.setitem(sys.modules, "wtpsplit", SimpleNamespace(SaT=fake_sat))

    def fake_cast(target: object, value: object) -> object:
        captured["cast_target"] = target
        return value

    monkeypatch.setattr(segmentation_module, "cast", fake_cast)

    SaTSentenceSegmenter(providers=("CPUExecutionProvider",))

    assert captured == {
        "model_id": "sat-3l-sm",
        "kwargs": {
            "ort_providers": ["CPUExecutionProvider"],
            "from_pretrained_kwargs": {
                "cache_dir": "/Volumes/Seagate M3/cache",
            },
        },
        "cast_target": segmentation_module._SaTModel,
    }


def test_segmenter_rejects_a_model_result_count_mismatch() -> None:
    model = FakeSaT((("Only one.",),))
    segmenter = SaTSentenceSegmenter(model=model, providers=("CPUExecutionProvider",))

    with pytest.raises(ValueError, match="returned 1 sentence lists for 2 texts"):
        segmenter.split_many(("Only one.", "Second text."))


def test_validate_batch_requires_strict_result_alignment() -> None:
    class LyingSequence:
        def __len__(self) -> int:
            return 1

        def __iter__(self):
            return iter(())

    with pytest.raises(ValueError, match="zip"):
        segmentation_module._validate_batch(
            ("Text.",), cast(tuple[Sequence[str], ...], LyingSequence())
        )
