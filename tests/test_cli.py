import json
import sys
from pathlib import Path

import pytest

from fineweb_polygons.cli import _build_parser, _summary_record, main
from fineweb_polygons.foundation import DEFAULT_DATA_ROOT
from fineweb_polygons.runs import RunSummary


def test_cli_reports_foundation_only(capsys) -> None:
    assert main([]) == 0

    assert capsys.readouterr().out == (
        "fineweb-polygons foundation only; no pipeline executed\n"
    )


def test_cli_scan_passes_external_paths_to_runner(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "external"
    pbf = data_root / "raw" / "monaco.osm.pbf"
    shard = data_root / "raw" / "shard.parquet"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return RunSummary(
            result_path=data_root / "artifacts" / "matches.jsonl",
            manifest_path=data_root / "runs" / "case" / "manifest.json",
            partitions_completed=1,
            partitions_skipped=0,
            rows_scanned=4,
            matches_written=2,
        )

    assert (
        main(
            [
                "scan",
                "--data-root",
                str(data_root),
                "--pbf",
                str(pbf),
                "--shard",
                str(shard),
                "--run-id",
                "case",
                "--batch-size",
                "16",
            ],
            runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].pbf_path == pbf
    assert captured["config"].shard_path == shard
    assert captured["config"].run_id == "case"
    assert captured["config"].batch_size == 16
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "result_path": str(data_root / "artifacts" / "matches.jsonl"),
        "manifest_path": str(data_root / "runs" / "case" / "manifest.json"),
        "partitions_completed": 1,
        "partitions_skipped": 0,
        "rows_scanned": 4,
        "matches_written": 2,
    }


def test_parser_exposes_exact_scan_defaults_and_help() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["scan", "--shard", "/tmp/shard.parquet"])

    assert parser.prog == "fineweb-polygons"
    assert parsed.command == "scan"
    assert parsed.data_root == DEFAULT_DATA_ROOT
    assert parsed.pbf is None
    assert parsed.run_id == "v1-10bt-000-v2"
    assert parsed.batch_size == 8192
    help_text = parser.format_help()
    assert "scan one FineWeb Parquet shard" in help_text
    assert "XXscan one FineWeb Parquet shardXX" not in help_text


def test_parser_requires_a_command_and_shard() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args([])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["scan"])


def test_main_uses_arguments_after_the_program_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return RunSummary(
            result_path=tmp_path / "result.jsonl",
            manifest_path=tmp_path / "manifest.json",
            partitions_completed=0,
            partitions_skipped=0,
            rows_scanned=0,
            matches_written=0,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        ["fineweb-polygons", "scan", "--shard", str(tmp_path / "shard.parquet")],
    )

    assert main(None, runner=fake_runner) == 0
    assert captured["config"].shard_path == tmp_path / "shard.parquet"


def test_scan_uses_default_pbf_and_reports_runner_errors(
    capsys,
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "external"
    shard = data_root / "raw" / "shard.parquet"
    seen = {}

    def failing_runner(config):
        seen["config"] = config
        raise ValueError("synthetic failure")

    assert (
        main(
            ["scan", "--data-root", str(data_root), "--shard", str(shard)],
            runner=failing_runner,
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: synthetic failure\n"
    assert seen["config"].paths.data_root == data_root.resolve()
    assert seen["config"].pbf_path == (data_root / "raw" / "monaco-latest.osm.pbf")


def test_summary_record_and_scan_output_preserve_all_fields_and_unicode(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    summary = RunSummary(
        result_path=tmp_path / "Café-results.jsonl",
        manifest_path=tmp_path / "Café-manifest.json",
        partitions_completed=3,
        partitions_skipped=4,
        rows_scanned=5,
        matches_written=6,
    )

    assert _summary_record(summary) == {
        "result_path": str(summary.result_path),
        "manifest_path": str(summary.manifest_path),
        "partitions_completed": 3,
        "partitions_skipped": 4,
        "rows_scanned": 5,
        "matches_written": 6,
    }

    def fake_runner(config):
        return summary

    import fineweb_polygons.cli as cli_module

    original_dumps = cli_module.json.dumps
    seen = {}

    def recording_dumps(value, *args, **kwargs):
        seen.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(cli_module.json, "dumps", recording_dumps)

    assert (
        main(
            ["scan", "--data-root", str(tmp_path), "--shard", str(tmp_path / "s")],
            runner=fake_runner,
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Café" in output
    assert "\\u00e9" not in output
    assert output.index('"matches_written"') < output.index('"result_path"')
    assert seen["ensure_ascii"] is False
    assert seen["sort_keys"] is True
