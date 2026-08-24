"""Minimal command-line entry point for the foundation release."""

from collections.abc import Sequence

FOUNDATION_MESSAGE = "fineweb-polygons foundation only; no pipeline executed"


def main(argv: Sequence[str] | None = None) -> int:
    """Report that no data-processing pipeline has been selected yet."""
    del argv
    print(FOUNDATION_MESSAGE)
    return 0
