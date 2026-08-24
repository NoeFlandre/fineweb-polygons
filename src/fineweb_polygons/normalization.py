"""Shared normalization rules for V1 exact matching."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import unquote

NORMALIZATION_VERSION = "v1-nfkc-casefold-separators"
_SEPARATOR_RE = re.compile(r"[\W_]+", re.UNICODE)


def normalize_for_search(value: object) -> str:
    """Normalize text for case-insensitive exact token matching."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = unquote(text)
    return " ".join(_SEPARATOR_RE.split(text)).strip()
