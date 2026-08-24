import argparse
import json
import sys
from pathlib import Path

import pytest

import fineweb_polygons.cli as cli_module
from fineweb_polygons.cli import _build_parser, main
from fineweb_polygons.foundation import DEFAULT_DATA_ROOT
from fineweb_polygons.runs import RunSummary


def test_cli_reports_foundation_only(capsys) -> None:
    assert main([]) == 0

    assert capsys.readouterr().out == (
        "fineweb-polygons foundation only; no pipeline executed\n"
    )


def test_cli_uses_process_argv_when_argv_is_none(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "external"
    shard = data_root / "raw" / "shard.parquet"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return RunSummary(
            result_path=data_root / "artifacts" / "matches.jsonl",
            manifest_path=data_root / "runs" / "case" / "manifest.json",
            partitions_completed=0,
            partitions_skipped=0,
            rows_scanned=0,
            matches_written=0,
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fineweb-polygons",
            "scan",
            "--data-root",
            str(data_root),
            "--shard",
            str(shard),
        ],
    )

    assert cli_module.main(None, runner=fake_runner) == 0
    assert captured["config"].shard_path == shard
    assert json.loads(capsys.readouterr().out)["matches_written"] == 0


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
                "--retrieval-version",
                "v2",
            ],
            runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].pbf_path == pbf
    assert captured["config"].shard_path == shard
    assert captured["config"].paths.data_root == data_root
    assert captured["config"].run_id == "case"
    assert captured["config"].batch_size == 16
    assert captured["config"].retrieval_version == "v2"
    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "manifest_path": str(data_root / "runs" / "case" / "manifest.json"),
        "matches_written": 2,
        "partitions_completed": 1,
        "partitions_skipped": 0,
        "result_path": str(data_root / "artifacts" / "matches.jsonl"),
        "rows_scanned": 4,
    }


def test_cli_parser_has_stable_scan_defaults_and_help() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(["scan", "--shard", "shard.parquet"])

    assert parser.prog == "fineweb-polygons"
    assert parsed.data_root == DEFAULT_DATA_ROOT
    assert parsed.pbf is None
    assert parsed.shard == Path("shard.parquet")
    assert parsed.run_id == "v1-10bt-000-v2"
    assert parsed.batch_size == 8192
    assert parsed.retrieval_version == "v1"
    assert "scan one FineWeb Parquet shard" in parser.format_help()


def test_cli_parser_exposes_exact_scan_contract() -> None:
    parser = _build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    scan_help = next(
        action.help for action in subparsers._choices_actions if action.dest == "scan"
    )
    assert scan_help == "scan one FineWeb Parquet shard"
    assert (
        parser.parse_args(
            ["scan", "--shard", "shard.parquet", "--retrieval-version", "v1"]
        ).retrieval_version
        == "v1"
    )
    assert (
        parser.parse_args(
            ["scan", "--shard", "shard.parquet", "--retrieval-version", "v3"]
        ).retrieval_version
        == "v3"
    )
    assert (
        parser.parse_args(
            ["scan", "--shard", "shard.parquet", "--retrieval-version", "v4"]
        ).retrieval_version
        == "v4"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["scan", "--shard", "shard.parquet", "--retrieval-version", "v5"]
        )


def test_cli_reports_unknown_commands_with_the_command_name(monkeypatch) -> None:
    class FakeParser:
        def parse_args(self, arguments):
            return argparse.Namespace(command="unexpected")

        def error(self, message):
            raise ValueError(message)

    monkeypatch.setattr(cli_module, "_build_parser", lambda: FakeParser())

    with pytest.raises(ValueError, match="unknown command: unexpected"):
        cli_module.main(["unexpected"])


def test_cli_requires_a_subcommand_and_a_shard() -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["scan"])


def test_cli_uses_default_pbf_path_and_reports_all_summary_fields(
    tmp_path: Path, capsys
) -> None:
    data_root = tmp_path / "external"
    shard = data_root / "raw" / "shard.parquet"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return RunSummary(
            result_path=data_root / "artifacts" / "result.jsonl",
            manifest_path=data_root / "runs" / "default" / "manifest.json",
            partitions_completed=3,
            partitions_skipped=4,
            rows_scanned=5,
            matches_written=6,
        )

    assert (
        main(
            [
                "scan",
                "--data-root",
                str(data_root),
                "--shard",
                str(shard),
            ],
            runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].pbf_path == (data_root / "raw" / "monaco-latest.osm.pbf")
    assert json.loads(capsys.readouterr().out) == {
        "manifest_path": str(data_root / "runs" / "default" / "manifest.json"),
        "matches_written": 6,
        "partitions_completed": 3,
        "partitions_skipped": 4,
        "result_path": str(data_root / "artifacts" / "result.jsonl"),
        "rows_scanned": 5,
    }


def test_cli_serializes_summary_with_stable_json_options(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    data_root = tmp_path / "external"
    captured: dict[str, object] = {}

    def fake_runner(config):
        return RunSummary(
            result_path=data_root / "artifacts" / "result.jsonl",
            manifest_path=data_root / "runs" / "case" / "manifest.json",
            partitions_completed=1,
            partitions_skipped=0,
            rows_scanned=1,
            matches_written=1,
        )

    def fake_dumps(value, **kwargs):
        captured["kwargs"] = kwargs
        return "serialized"

    monkeypatch.setattr(cli_module.json, "dumps", fake_dumps)

    assert (
        cli_module.main(
            [
                "scan",
                "--data-root",
                str(data_root),
                "--shard",
                str(data_root / "raw" / "shard.parquet"),
            ],
            runner=fake_runner,
        )
        == 0
    )

    assert captured["kwargs"] == {"ensure_ascii": False, "sort_keys": True}
    assert capsys.readouterr().out == "serialized\n"


@pytest.mark.parametrize("error", [FileNotFoundError("missing"), ValueError("bad")])
def test_cli_reports_runner_errors_as_exit_code_two(
    tmp_path: Path, capsys, error: Exception
) -> None:
    def failing_runner(config):
        raise error

    assert (
        main(
            [
                "scan",
                "--data-root",
                str(tmp_path / "external"),
                "--shard",
                str(tmp_path / "shard.parquet"),
            ],
            runner=failing_runner,
        )
        == 2
    )

    assert capsys.readouterr().err == f"error: {error}\n"
