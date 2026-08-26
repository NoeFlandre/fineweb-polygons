"""Tests for the public compatibility boundaries used during organization."""

from fineweb_polygons import runs, v9
from fineweb_polygons.run_models import (
    RunSummary,
    ScanRunConfig,
    _Partition,
    _ProfileRunData,
    _RowGroup,
    _RunCounters,
    _RunLayout,
)
from fineweb_polygons.v9_models import V9RunConfig, V9RunSummary, _OutputStats


def test_runs_facade_reexports_its_value_objects() -> None:
    assert runs.ScanRunConfig is ScanRunConfig
    assert runs.RunSummary is RunSummary
    assert runs._Partition is _Partition
    assert runs._ProfileRunData is _ProfileRunData
    assert runs._RowGroup is _RowGroup
    assert runs._RunCounters is _RunCounters
    assert runs._RunLayout is _RunLayout


def test_v9_facade_reexports_its_value_objects() -> None:
    assert v9.V9RunConfig is V9RunConfig
    assert v9.V9RunSummary is V9RunSummary
    assert v9._OutputStats is _OutputStats
