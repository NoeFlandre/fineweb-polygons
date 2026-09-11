"""Property-based invariants for pure search primitives."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from fineweb_polygons.core.normalization import normalize_for_search
from fineweb_polygons.directions.lexical.matching import (
    AhoCorasickPatternMatcher,
)


@pytest.mark.property
@given(st.text(max_size=256))
def test_search_normalization_is_idempotent(text: str) -> None:
    normalized = normalize_for_search(text, decode_url=False)

    assert normalize_for_search(normalized, decode_url=False) == normalized


@pytest.mark.property
@given(
    st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd")),
            min_size=1,
            max_size=12,
        ),
        max_size=8,
    )
)
def test_pattern_matcher_returns_each_indexed_word_once(
    patterns: list[str],
) -> None:
    matcher = AhoCorasickPatternMatcher.build(patterns)
    expected = tuple(
        sorted({normalize_for_search(pattern, decode_url=False) for pattern in patterns})
    )

    assert matcher.find_unique_patterns(" ".join(patterns)) == expected
