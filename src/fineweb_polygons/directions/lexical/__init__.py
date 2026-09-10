"""Direction 2 - lexical polygon candidates (active POC).

Modules in this package are shared by every lexical version: OSM area
reading, sentence windows, and the Aho-Corasick matcher. Each version keeps
its own contract, run configuration, pipeline, and dataset card in its own
subpackage, so a new version is a new subpackage rather than a new file
prefix.
"""

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DIRECTION_VERSION": (
        "fineweb_polygons.directions.lexical.v1.models",
        "DIRECTION_VERSION",
    ),
    "HF_CONFIG_NAME": (
        "fineweb_polygons.directions.lexical.v1.models",
        "HF_CONFIG_NAME",
    ),
    "Direction2RunConfig": (
        "fineweb_polygons.directions.lexical.v1.models",
        "Direction2RunConfig",
    ),
    "Direction2RunSummary": (
        "fineweb_polygons.directions.lexical.v1.models",
        "Direction2RunSummary",
    ),
    "run_direction2": (
        "fineweb_polygons.directions.lexical.v1.pipeline",
        "run_direction2",
    ),
    "DIRECTION_V2_VERSION": (
        "fineweb_polygons.directions.lexical.v2.models",
        "DIRECTION_V2_VERSION",
    ),
    "HF_CONFIG_NAME_V2": (
        "fineweb_polygons.directions.lexical.v2.models",
        "HF_CONFIG_NAME_V2",
    ),
    "Direction2V2RunConfig": (
        "fineweb_polygons.directions.lexical.v2.models",
        "Direction2V2RunConfig",
    ),
    "Direction2V2RunSummary": (
        "fineweb_polygons.directions.lexical.v2.models",
        "Direction2V2RunSummary",
    ),
    "run_direction2_v2": (
        "fineweb_polygons.directions.lexical.v2.pipeline",
        "run_direction2_v2",
    ),
}

DIRECTION_ID = "direction-2-lexical-candidates"

__all__ = [
    "DIRECTION_ID",
    "DIRECTION_V2_VERSION",
    "DIRECTION_VERSION",
    "HF_CONFIG_NAME",
    "HF_CONFIG_NAME_V2",
    "Direction2RunConfig",
    "Direction2RunSummary",
    "Direction2V2RunConfig",
    "Direction2V2RunSummary",
    "run_direction2",
    "run_direction2_v2",
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
