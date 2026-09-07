"""Direction 2 - lexical polygon candidates (active POC).

Modules in this package are shared by every lexical version: OSM area
reading, sentence windows, and the Aho-Corasick matcher. Each version keeps
its own contract, run configuration, pipeline, and dataset card in its own
subpackage, so a new version is a new subpackage rather than a new file
prefix.
"""

from fineweb_polygons.directions.lexical.v1.models import (
    DIRECTION_VERSION,
    HF_CONFIG_NAME,
    Direction2RunConfig,
    Direction2RunSummary,
)
from fineweb_polygons.directions.lexical.v1.pipeline import run_direction2
from fineweb_polygons.directions.lexical.v2.models import (
    DIRECTION_V2_VERSION,
    HF_CONFIG_NAME_V2,
    Direction2V2RunConfig,
    Direction2V2RunSummary,
)
from fineweb_polygons.directions.lexical.v2.pipeline import run_direction2_v2

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
