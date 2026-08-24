"""Shared normalization rules for V1 exact matching."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote

NORMALIZATION_VERSION = "v1-nfkc-casefold-separators"
_SEPARATOR_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_for_search(value: object, *, decode_url: bool = True) -> str:
    """Normalize a field for case-insensitive exact token matching."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    if decode_url:
        text = unquote(text)
    text = text.casefold()
    return _SEPARATOR_RE.sub(" ", text).strip()


def has_monaco_marker(value: object, *, decode_url: bool = True) -> bool:
    """Cheap, sound prefilter for either accepted Monaco context phrase."""
    if value is None:
        return False
    text = unicodedata.normalize("NFKC", str(value))
    if decode_url:
        text = unquote(text)
    text = text.casefold()
    return "monaco" in text
