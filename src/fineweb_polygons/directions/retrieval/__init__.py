"""Direction 1 - FineWeb polygon retrieval (frozen at V10).

V1-V6 are retrieval rules driven by `runs.execute_run`; V7-V10 are
post-processing stages over an earlier version's published rows. The
`stages` subpackage holds the numbered post-processing chain so the
version numbering stays contained in one place.
"""

from fineweb_polygons.directions.retrieval.runs import (
    RunSummary,
    ScanRunConfig,
    execute_run,
)
from fineweb_polygons.directions.retrieval.stages.v7 import (
    V7RunConfig,
    V7RunSummary,
    run_v7,
)
from fineweb_polygons.directions.retrieval.stages.v8 import (
    V8RunConfig,
    V8RunSummary,
    run_v8,
)
from fineweb_polygons.directions.retrieval.stages.v9 import (
    V9RunConfig,
    V9RunSummary,
    run_v9,
)
from fineweb_polygons.directions.retrieval.stages.v10 import (
    V10_MAX_NEW_TOKENS,
    V10RunConfig,
    V10RunSummary,
    run_v10,
)

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
