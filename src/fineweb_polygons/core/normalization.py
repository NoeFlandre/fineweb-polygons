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


def has_context_marker(
    value: object, context_name: str, *, decode_url: bool = True
) -> bool:
    """Return whether a normalized field contains one exact context name."""
    if value is None:
        return False
    normalized = normalize_for_search(value, decode_url=decode_url)
    context = normalize_for_search(context_name, decode_url=False)
    return bool(context) and f" {context} " in f" {normalized} "


def has_monaco_marker(value: object, *, decode_url: bool = True) -> bool:
    """Backward-compatible Monaco context prefilter for older callers."""
    return has_context_marker(value, "Monaco", decode_url=decode_url)
