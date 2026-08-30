"""Tests for the public compatibility boundaries used during organization."""

from fineweb_polygons import (
    artifact_io,
    deduplication,
    runs,
    scanning,
    v7,
    v8,
    v9,
    v10,
)
from fineweb_polygons.run_models import (
    RunSummary,
    ScanRunConfig,
    _Partition,
    _ProfileRunData,
    _ProfileSelection,
    _RowGroup,
    _RunCounters,
    _RunLayout,
)
from fineweb_polygons.v9_models import V9RunConfig, V9RunSummary, _OutputStats
from fineweb_polygons.v10_models import V10RunConfig, V10RunSummary


def test_runs_facade_reexports_its_value_objects() -> None:
    assert runs.ScanRunConfig is ScanRunConfig
    assert runs.RunSummary is RunSummary
    assert runs._Partition is _Partition
    assert runs._ProfileSelection is _ProfileSelection
    assert runs._ProfileRunData is _ProfileRunData
    assert runs._RowGroup is _RowGroup
    assert runs._RunCounters is _RunCounters
    assert runs._RunLayout is _RunLayout


def test_v9_facade_reexports_its_value_objects() -> None:
    assert v9.V9RunConfig is V9RunConfig
    assert v9.V9RunSummary is V9RunSummary
    assert v9._OutputStats is _OutputStats


def test_v10_facade_reexports_its_value_objects() -> None:
    assert v10.V10RunConfig is V10RunConfig
    assert v10.V10RunSummary is V10RunSummary


def test_pipeline_stages_share_the_artifact_io_boundary() -> None:
    assert v7._atomic_json_write is artifact_io.atomic_json_write
    assert v7._read_manifest is artifact_io.read_json_object
    assert v7._sha256_file is artifact_io.sha256_file
    assert v8._atomic_json_write is artifact_io.atomic_json_write
    assert v8._read_manifest is artifact_io.read_json_object
    assert v8._sha256_file is artifact_io.sha256_file
    assert v9._atomic_json_write is artifact_io.atomic_json_write
    assert v9._read_manifest is artifact_io.read_json_object
    assert v9._sha256_file is artifact_io.sha256_file
    assert v10._atomic_json_write is artifact_io.atomic_json_write
    assert v10._read_manifest is artifact_io.read_json_object
    assert v10._sha256_file is artifact_io.sha256_file
    assert runs._sha256_file is artifact_io.sha256_file
    assert scanning._atomic_text_output is artifact_io.atomic_text_output
    assert scanning._write_json_line is artifact_io.write_json_line
    assert deduplication._atomic_text_output is artifact_io.atomic_text_output


def test_jsonl_stages_share_the_artifact_object_reader() -> None:
    for stage in (v7, v8, v9, v10):
        assert stage._decode_json_object_line is artifact_io.decode_json_object_line
        assert stage._iter_json_objects is artifact_io.iter_json_objects
