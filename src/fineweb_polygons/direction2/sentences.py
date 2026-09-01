"""Deterministic sentence windows for lexical candidate evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass

from fineweb_polygons.direction2.models import SentenceWindow

_TERMINAL_PUNCTUATION = re.compile(r"[.!?]+(?=\s|$)")


@dataclass(frozen=True, slots=True)
class SentenceSpan:
    """A sentence boundary in the original document text."""

    start: int
    end: int

    def value(self, text: str) -> str:
        """Return the trimmed original sentence."""
        return text[self.start : self.end].strip()


def split_sentences(text: str) -> tuple[SentenceSpan, ...]:
    """Split on terminal punctuation while retaining original text offsets."""
    spans: list[SentenceSpan] = []
    start = 0
    for punctuation in _TERMINAL_PUNCTUATION.finditer(text):
        end = punctuation.end()
        if text[start:end].strip():
            spans.append(SentenceSpan(start=start, end=end))
        start = end
    if text[start:].strip():
        spans.append(SentenceSpan(start=start, end=len(text)))
    return tuple(spans)


def context_for_match(
    text: str,
    spans: tuple[SentenceSpan, ...],
    *,
    match_start: int,
) -> SentenceWindow:
    """Return the containing sentence plus at most one neighbor on each side."""
    sentence_index = _sentence_index(spans, match_start)
    sentence = spans[sentence_index]
    first = max(0, sentence_index - 1)
    last = min(len(spans) - 1, sentence_index + 1)
    context = text[spans[first].start : spans[last].end].strip()
    return SentenceWindow(sentence=sentence.value(text), context=context)


def _sentence_index(spans: tuple[SentenceSpan, ...], match_start: int) -> int:
    for index, span in enumerate(spans):
        if span.start <= match_start < span.end:
            return index
    raise ValueError("match_start is outside the document sentences")
