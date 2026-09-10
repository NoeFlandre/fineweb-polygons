"""Direction 1 - FineWeb polygon retrieval (frozen at V10).

V1-V6 are retrieval rules driven by `runs.execute_run`; V7-V10 are
post-processing stages over an earlier version's published rows. The
`stages` subpackage holds the numbered post-processing chain so the
version numbering stays contained in one place.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "RunSummary": (
        "fineweb_polygons.directions.retrieval.runs",
        "RunSummary",
    ),
    "ScanRunConfig": (
        "fineweb_polygons.directions.retrieval.runs",
        "ScanRunConfig",
    ),
    "execute_run": (
        "fineweb_polygons.directions.retrieval.runs",
        "execute_run",
    ),
    "V7RunConfig": (
        "fineweb_polygons.directions.retrieval.stages.v7",
        "V7RunConfig",
    ),
    "V7RunSummary": (
        "fineweb_polygons.directions.retrieval.stages.v7",
        "V7RunSummary",
    ),
    "run_v7": (
        "fineweb_polygons.directions.retrieval.stages.v7",
        "run_v7",
    ),
    "V8RunConfig": (
        "fineweb_polygons.directions.retrieval.stages.v8",
        "V8RunConfig",
    ),
    "V8RunSummary": (
        "fineweb_polygons.directions.retrieval.stages.v8",
        "V8RunSummary",
    ),
    "run_v8": (
        "fineweb_polygons.directions.retrieval.stages.v8",
        "run_v8",
    ),
    "V9RunConfig": (
        "fineweb_polygons.directions.retrieval.stages.v9",
        "V9RunConfig",
    ),
    "V9RunSummary": (
        "fineweb_polygons.directions.retrieval.stages.v9",
        "V9RunSummary",
    ),
    "run_v9": (
        "fineweb_polygons.directions.retrieval.stages.v9",
        "run_v9",
    ),
    "V10_MAX_NEW_TOKENS": (
        "fineweb_polygons.directions.retrieval.stages.v10",
        "V10_MAX_NEW_TOKENS",
    ),
    "V10RunConfig": (
        "fineweb_polygons.directions.retrieval.stages.v10",
        "V10RunConfig",
    ),
    "V10RunSummary": (
        "fineweb_polygons.directions.retrieval.stages.v10",
        "V10RunSummary",
    ),
    "run_v10": (
        "fineweb_polygons.directions.retrieval.stages.v10",
        "run_v10",
    ),
}

DIRECTION_ID = "direction-1-fineweb-retrieval"

__all__ = [
    "DIRECTION_ID",
    "V10_MAX_NEW_TOKENS",
    "RunSummary",
    "ScanRunConfig",
    "V7RunConfig",
    "V7RunSummary",
    "V8RunConfig",
    "V8RunSummary",
    "V9RunConfig",
    "V9RunSummary",
    "V10RunConfig",
    "V10RunSummary",
    "execute_run",
    "run_v7",
    "run_v8",
    "run_v9",
    "run_v10",
]


def __getattr__(name: str) -> Any:
    """Load one public direction symbol only when it is first used."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
