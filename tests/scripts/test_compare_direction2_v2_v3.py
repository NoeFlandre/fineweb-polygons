"""Behavioral tests for the typed Direction 2 comparison boundary."""

from __future__ import annotations

import pytest
from scripts.compare_direction2_v2_v3 import _as_int


@pytest.mark.parametrize(
    ("value", "expected"),
    ((0, 0), (True, 1), (2.9, 2), ("3", 3), (b"4", 4)),
)
def test_as_int_preserves_supported_numeric_conversions(
    value: object, expected: int
) -> None:
    assert _as_int(value) == expected


def test_as_int_rejects_unexpected_parquet_values() -> None:
    with pytest.raises(TypeError, match="expected an integer-like value"):
        _as_int({"value": 1})
