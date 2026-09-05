"""Tests for the public compatibility boundaries used during organization."""

from fineweb_polygons import (
    runs,
    v9,
    v10,
)
from fineweb_polygons.run_models import (
    RunSummary,
    ScanRunConfig,
)
from fineweb_polygons.v9_models import V9RunConfig, V9RunSummary
from fineweb_polygons.v10_models import V10RunConfig, V10RunSummary


def test_runs_facade_reexports_its_value_objects() -> None:
    assert runs.ScanRunConfig is ScanRunConfig
    assert runs.RunSummary is RunSummary


def test_v9_facade_reexports_its_value_objects() -> None:
    assert v9.V9RunConfig is V9RunConfig
    assert v9.V9RunSummary is V9RunSummary


def test_v10_facade_reexports_its_value_objects() -> None:
    assert v10.V10RunConfig is V10RunConfig
    assert v10.V10RunSummary is V10RunSummary
