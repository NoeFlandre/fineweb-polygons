"""Local LFM inference for the V10 sentence-classification contract."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

V10_PROMPT_TEMPLATE = (
    "Classify whether the TARGET SENTENCE contains information about the target "
    "place that could help characterize its land use, land cover, or geographic "
    "environment from remote sensing, either directly or through observable "
    "proxies.\n"
    "Return exactly one token: yes or no.\n"
    "Answer **yes** for information about vegetation, agriculture, forests, "
    "water, soil or surface, terrain, buildings, settlements, infrastructure, "
    "transport networks, mining, managed land, or other human or natural "
    "features with a spatial or remotely detectable signature.\n"
    "Answer **no** for information only about history, administration, people, "
    "events, demographics, economy, navigation, or activities with no meaningful "
    "land-use, land-cover, or remotely detectable implication.\n"
    "Output only the lowercase token yes or no.\n"
    "TARGET SENTENCE: {}"
)
REASONING_CLOSE_TAG = "</think>"
PROMPT_SHA256 = hashlib.sha256(V10_PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


def format_prompt(target_sentence: str) -> str:
    """Insert one target sentence into the approved prompt."""
    return V10_PROMPT_TEMPLATE.format(target_sentence)


def render_chat_prompt(tokenizer: Any, prompt: str) -> str:
    """Render the prompt with the chat format expected by LFM2.5."""
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if apply_chat_template is None:
        return prompt
    return (
        apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        + REASONING_CLOSE_TAG
    )


def render_mlx_prompt(tokenizer: Any, prompt: str) -> list[int]:
    """Render an LFM prompt as the token IDs expected by ``batch_generate``."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_dict=False,
    )
    close_tag = tokenizer.encode(REASONING_CLOSE_TAG)
    if _is_token_ids(rendered) and _is_token_ids(close_tag):
        return rendered + close_tag
    if isinstance(rendered, str):
        encoded = tokenizer.encode(rendered + REASONING_CLOSE_TAG)
        if _is_token_ids(encoded):
            return encoded
    raise TypeError("MLX chat template did not return token IDs")


def _is_token_ids(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, int) for item in value)


def parse_label(raw_output: str) -> str:
    """Accept only the exact lowercase labels required by the contract."""
    label = raw_output.strip()
    if label not in {"yes", "no"}:
        if "</think>" in label:
            label = label.rsplit("</think>", maxsplit=1)[-1].strip()
        if label not in {"yes", "no"}:
            raise ValueError(
                f"Classifier output must be exactly yes or no: {raw_output!r}"
            )
    return label


class LfmSentenceClassifier:
    """Run the local LiquidAI LFM model once per batch of sentences."""

    def __init__(self, model_path: Path, *, max_new_tokens: int = 2) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        torch, tokenizer_class, model_class = _load_transformers()
        self._torch = torch
        self._device = _select_device(torch)
        self._tokenizer = tokenizer_class.from_pretrained(
            str(model_path), local_files_only=True
        )
        _configure_tokenizer(self._tokenizer)
        self._model = model_class.from_pretrained(
            str(model_path),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self._model.to(self._device)
        self._model.eval()
        self._max_new_tokens = max_new_tokens

    def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
        """Classify a batch and validate every generated answer."""
        if not sentences:
            return ()
        prompts = [
            render_chat_prompt(self._tokenizer, format_prompt(sentence))
            for sentence in sentences
        ]
        encoded = self._encode(prompts)
        generated = self._generate(encoded)
        input_width = encoded["input_ids"].shape[1]
        decoded = self._tokenizer.batch_decode(
            generated[:, input_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        labels = tuple(parse_label(output) for output in decoded)
        if len(labels) != len(sentences):
            raise RuntimeError("The classifier returned the wrong number of labels")
        return labels

    def _encode(self, prompts: Sequence[str]) -> Any:
        return self._tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(self._device)

    def _generate(self, encoded: Any) -> Any:
        with self._torch.inference_mode():
            return self._model.generate(
                **encoded,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
            )


class MlxSentenceClassifier:
    """Run a Seagate-backed MLX LFM runtime with continuous batching."""

    def __init__(self, model_path: Path, *, max_new_tokens: int = 512) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        load, batch_generate = _load_mlx()
        self._model, self._tokenizer = load(str(model_path))
        self._batch_generate = batch_generate
        self._max_new_tokens = max_new_tokens

    def classify(self, sentences: Sequence[str]) -> tuple[str, ...]:
        """Classify a batch and validate every generated answer."""
        if not sentences:
            return ()
        prompts = [
            render_mlx_prompt(self._tokenizer, format_prompt(sentence))
            for sentence in sentences
        ]
        result = self._batch_generate(
            self._model,
            self._tokenizer,
            prompts,
            max_tokens=self._max_new_tokens,
            verbose=False,
        )
        labels = tuple(parse_label(text) for text in result.texts)
        if len(labels) != len(sentences):
            raise RuntimeError("The classifier returned the wrong number of labels")
        return labels


def _load_transformers() -> tuple[Any, Any, Any]:
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError(
            "V10 requires torch and transformers in the model runtime"
        ) from error
    return torch, transformers.AutoTokenizer, transformers.AutoModelForCausalLM


def _load_mlx() -> tuple[Any, Any]:
    try:
        mlx_lm = importlib.import_module("mlx_lm")
        generation = importlib.import_module("mlx_lm.generate")
    except ImportError as error:
        raise RuntimeError("V10 MLX inference requires the mlx-lm package") from error
    return mlx_lm.load, generation.batch_generate


def _select_device(torch: Any) -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _configure_tokenizer(tokenizer: Any) -> None:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
