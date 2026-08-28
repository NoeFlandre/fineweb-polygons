import argparse
import json
import sys
from pathlib import Path

import pytest

import fineweb_polygons.cli as cli_module
from fineweb_polygons.cli import _build_parser, main
from fineweb_polygons.foundation import DEFAULT_DATA_ROOT
from fineweb_polygons.runs import RunSummary
from fineweb_polygons.v7 import V7RunSummary
from fineweb_polygons.v8 import V8RunSummary
from fineweb_polygons.v9 import V9RunSummary
from fineweb_polygons.v10 import V10RunSummary


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
    assert captured["config"].country_name == "Monaco"
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
    assert parsed.country_name == "Monaco"
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
    assert (
        parser.parse_args(
            ["scan", "--shard", "shard.parquet", "--retrieval-version", "v6"]
        ).retrieval_version
        == "v6"
    )
    assert (
        parser.parse_args(
            [
                "scan",
                "--shard",
                "shard.parquet",
                "--retrieval-version",
                "v5",
                "--country-name",
                "Liechtenstein",
            ]
        ).country_name
        == "Liechtenstein"
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
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["scan", "--shard", "shard.parquet", "--retrieval-version", "v7"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["segment-v7", "--output", "v7.jsonl", "--manifest", "manifest.json"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["segment-v7", "--input", "v6.jsonl", "--manifest", "manifest.json"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(["segment-v7", "--input", "v6.jsonl", "--output", "v7.jsonl"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "segment-v7",
                "--input",
                "v6.jsonl",
                "--output",
                "v7.jsonl",
                "--manifest",
                "manifest.json",
                "--model-id",
                "other-model",
            ]
        )


def test_cli_parser_exposes_the_separate_v7_segmentation_command() -> None:
    parser = _build_parser()

    parsed = parser.parse_args(
        [
            "segment-v7",
            "--input",
            "v6.jsonl",
            "--output",
            "v7.jsonl",
            "--manifest",
            "manifest.json",
        ]
    )

    assert parsed.command == "segment-v7"
    assert parsed.data_root == DEFAULT_DATA_ROOT
    assert parsed.input == Path("v6.jsonl")
    assert parsed.output == Path("v7.jsonl")
    assert parsed.manifest == Path("manifest.json")
    assert parsed.batch_size == 32
    assert parsed.model_id == "sat-3l-sm"
    explicit_model = parser.parse_args(
        [
            "segment-v7",
            "--input",
            "v6.jsonl",
            "--output",
            "v7.jsonl",
            "--manifest",
            "manifest.json",
            "--model-id",
            "sat-3l-sm",
        ]
    )
    assert explicit_model.model_id == "sat-3l-sm"
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    segment_help = next(
        action.help
        for action in subparsers._choices_actions
        if action.dest == "segment-v7"
    )
    assert segment_help == "split V6 documents into exact sentence lists"


def test_cli_parser_exposes_the_separate_v8_topic_filter_command() -> None:
    parser = _build_parser()

    parsed = parser.parse_args(
        [
            "filter-v8",
            "--input",
            "v7.jsonl",
            "--output",
            "v8.jsonl",
            "--manifest",
            "manifest.json",
            "--vocabulary",
            "vocabulary.json",
        ]
    )

    assert parsed.command == "filter-v8"
    assert parsed.data_root == DEFAULT_DATA_ROOT
    assert parsed.input == Path("v7.jsonl")
    assert parsed.output == Path("v8.jsonl")
    assert parsed.manifest == Path("manifest.json")
    assert parsed.vocabulary == Path("vocabulary.json")

    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    topic_help = next(
        action.help
        for action in subparsers._choices_actions
        if action.dest == "filter-v8"
    )
    assert topic_help == "filter V7 documents with the approved topic vocabulary"


@pytest.mark.parametrize(
    "missing", ["--input", "--output", "--manifest", "--vocabulary"]
)
def test_cli_v8_requires_all_input_paths(missing: str) -> None:
    parser = _build_parser()
    arguments = {
        "--input": "v7.jsonl",
        "--output": "v8.jsonl",
        "--manifest": "manifest.json",
        "--vocabulary": "vocabulary.json",
    }
    filtered = [
        value
        for option, value in arguments.items()
        if option != missing
        for value in (option, value)
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(["filter-v8", *filtered])


def test_cli_runs_v7_segmentation_and_serializes_its_summary(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    data_root = tmp_path / "external-é"
    captured = {}
    constructor_kwargs = {}

    def fake_config(**kwargs):
        constructor_kwargs.update(kwargs)
        return argparse.Namespace(**kwargs)

    monkeypatch.setattr(cli_module, "V7RunConfig", fake_config)

    def fake_runner(config):
        captured["config"] = config
        return V7RunSummary(
            output_path=data_root / "artifacts" / "v7.jsonl",
            manifest_path=data_root / "runs" / "v7" / "manifest.json",
            rows_processed=2,
            sentences_written=3,
            result_sha256="result-sha",
        )

    assert (
        main(
            [
                "segment-v7",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "artifacts" / "v6.jsonl"),
                "--output",
                str(data_root / "artifacts" / "v7.jsonl"),
                "--manifest",
                str(data_root / "runs" / "v7" / "manifest.json"),
                "--batch-size",
                "8",
            ],
            v7_runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].input_path == data_root / "artifacts" / "v6.jsonl"
    assert captured["config"].output_path == data_root / "artifacts" / "v7.jsonl"
    assert captured["config"].manifest_path == (
        data_root / "runs" / "v7" / "manifest.json"
    )
    assert captured["config"].batch_size == 8
    assert captured["config"].model_id == "sat-3l-sm"
    assert constructor_kwargs["model_id"] == "sat-3l-sm"
    assert (
        capsys.readouterr().out
        == json.dumps(
            {
                "manifest_path": str(data_root / "runs" / "v7" / "manifest.json"),
                "output_path": str(data_root / "artifacts" / "v7.jsonl"),
                "result_sha256": "result-sha",
                "rows_processed": 2,
                "sentences_written": 3,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )


def test_cli_runs_v8_topic_filter_and_serializes_its_summary(
    tmp_path: Path, capsys
) -> None:
    data_root = tmp_path / "external"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return V8RunSummary(
            output_path=data_root / "artifacts" / "v8.jsonl",
            manifest_path=data_root / "runs" / "v8" / "manifest.json",
            rows_processed=3,
            rows_kept=1,
            rows_filtered=2,
            category_documents={"land_use": 1},
            result_sha256="result-sha",
        )

    assert (
        main(
            [
                "filter-v8",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "artifacts" / "v7.jsonl"),
                "--output",
                str(data_root / "artifacts" / "v8.jsonl"),
                "--manifest",
                str(data_root / "runs" / "v8" / "manifest.json"),
                "--vocabulary",
                str(data_root / "vocabulary.json"),
            ],
            v8_runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].input_path == data_root / "artifacts" / "v7.jsonl"
    assert captured["config"].output_path == data_root / "artifacts" / "v8.jsonl"
    assert captured["config"].manifest_path == (
        data_root / "runs" / "v8" / "manifest.json"
    )
    assert captured["config"].vocabulary_path == data_root / "vocabulary.json"
    assert json.loads(capsys.readouterr().out) == {
        "category_documents": {"land_use": 1},
        "manifest_path": str(data_root / "runs" / "v8" / "manifest.json"),
        "output_path": str(data_root / "artifacts" / "v8.jsonl"),
        "result_sha256": "result-sha",
        "rows_filtered": 2,
        "rows_kept": 1,
        "rows_processed": 3,
    }


def test_cli_v8_passes_explicit_json_serialization_flags(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    data_root = tmp_path / "external"
    captured: dict[str, object] = {}

    def fake_dumps(value: object, **kwargs: object) -> str:
        captured["value"] = value
        captured["kwargs"] = kwargs
        return "serialized"

    def fake_runner(config):
        return V8RunSummary(
            output_path=data_root / "v8.jsonl",
            manifest_path=data_root / "manifest.json",
            rows_processed=1,
            rows_kept=1,
            rows_filtered=0,
            category_documents={"land_use": 1},
            result_sha256="sha",
        )

    monkeypatch.setattr(cli_module.json, "dumps", fake_dumps)

    assert (
        main(
            [
                "filter-v8",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v7.jsonl"),
                "--output",
                str(data_root / "v8.jsonl"),
                "--manifest",
                str(data_root / "manifest.json"),
                "--vocabulary",
                str(data_root / "vocabulary.json"),
            ],
            v8_runner=fake_runner,
        )
        == 0
    )

    assert captured["kwargs"] == {"ensure_ascii": False, "sort_keys": True}
    assert capsys.readouterr().out == "serialized\n"


def test_cli_parser_exposes_the_separate_v9_sentence_topic_filter_command() -> None:
    parser = _build_parser()

    parsed = parser.parse_args(
        [
            "filter-v9",
            "--input",
            "v8.jsonl",
            "--output",
            "v9.jsonl",
            "--manifest",
            "manifest.json",
            "--vocabulary",
            "vocabulary.json",
        ]
    )

    assert parsed.command == "filter-v9"
    assert parsed.data_root == DEFAULT_DATA_ROOT
    assert parsed.input == Path("v8.jsonl")
    assert parsed.output == Path("v9.jsonl")
    assert parsed.manifest == Path("manifest.json")
    assert parsed.vocabulary == Path("vocabulary.json")


def test_cli_v9_parser_requires_all_paths_and_exposes_its_help_text() -> None:
    parser = _build_parser()
    full = [
        "filter-v9",
        "--input",
        "v8.jsonl",
        "--output",
        "v9.jsonl",
        "--manifest",
        "manifest.json",
        "--vocabulary",
        "vocabulary.json",
    ]

    help_text = parser.format_help()
    assert "filter V8 rows to local topic-relevant sentences" in help_text
    assert "XX" not in help_text
    for option in ("--input", "--output", "--manifest", "--vocabulary"):
        option_index = full.index(option)
        with pytest.raises(SystemExit):
            parser.parse_args(full[:option_index] + full[option_index + 2 :])


def test_cli_v9_serializes_with_explicit_stable_json_options(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    data_root = tmp_path / "external"
    captured: dict[str, object] = {}

    def fake_dumps(value: object, **kwargs: object) -> str:
        captured["value"] = value
        captured["kwargs"] = kwargs
        return "serialized"

    def fake_runner(config):
        return V9RunSummary(
            output_path=data_root / "v9.jsonl",
            manifest_path=data_root / "manifest.json",
            rows_processed=0,
            rows_kept=0,
            rows_filtered=0,
            sentences_processed=0,
            relevant_sentences_written=0,
            category_sentences={},
            result_sha256="sha",
        )

    monkeypatch.setattr(cli_module.json, "dumps", fake_dumps)

    assert (
        main(
            [
                "filter-v9",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v8.jsonl"),
                "--output",
                str(data_root / "v9.jsonl"),
                "--manifest",
                str(data_root / "manifest.json"),
                "--vocabulary",
                str(data_root / "vocabulary.json"),
            ],
            v9_runner=fake_runner,
        )
        == 0
    )

    assert captured["kwargs"] == {"ensure_ascii": False, "sort_keys": True}
    assert capsys.readouterr().out == "serialized\n"


def test_cli_v9_reports_runner_errors_to_stderr(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "external"

    def failing_runner(config):
        raise ValueError("bad V9 input")

    assert (
        main(
            [
                "filter-v9",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v8.jsonl"),
                "--output",
                str(data_root / "v9.jsonl"),
                "--manifest",
                str(data_root / "manifest.json"),
                "--vocabulary",
                str(data_root / "vocabulary.json"),
            ],
            v9_runner=failing_runner,
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: bad V9 input\n"


def test_cli_parser_exposes_the_v10_model_filter_command() -> None:
    parser = _build_parser()

    parsed = parser.parse_args(
        [
            "filter-v10",
            "--input",
            "v9.jsonl",
            "--output",
            "v10.jsonl",
            "--manifest",
            "manifest.json",
            "--model-path",
            "model",
            "--checkpoint",
            "checkpoint.jsonl",
            "--batch-size",
            "4",
            "--max-new-tokens",
            "3",
        ]
    )

    assert parsed.command == "filter-v10"
    assert parsed.data_root == DEFAULT_DATA_ROOT
    assert parsed.input == Path("v9.jsonl")
    assert parsed.output == Path("v10.jsonl")
    assert parsed.manifest == Path("manifest.json")
    assert parsed.model_path == Path("model")
    assert parsed.checkpoint == Path("checkpoint.jsonl")
    assert parsed.batch_size == 4
    assert parsed.max_new_tokens == 3
    defaults = parser.parse_args(
        [
            "filter-v10",
            "--input",
            "v9.jsonl",
            "--output",
            "v10.jsonl",
            "--manifest",
            "manifest.json",
            "--model-path",
            "model",
        ]
    )
    assert defaults.max_new_tokens == 4


@pytest.mark.parametrize(
    "missing", ["--input", "--output", "--manifest", "--model-path"]
)
def test_cli_v10_requires_all_input_paths(missing: str) -> None:
    parser = _build_parser()
    arguments = {
        "--input": "v9.jsonl",
        "--output": "v10.jsonl",
        "--manifest": "manifest.json",
        "--model-path": "model",
    }
    filtered = [
        value
        for option, value in arguments.items()
        if option != missing
        for value in (option, value)
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(["filter-v10", *filtered])


def test_cli_v10_parser_exposes_stable_defaults_and_help() -> None:
    parser = _build_parser()
    parsed = parser.parse_args(
        [
            "filter-v10",
            "--input",
            "v9.jsonl",
            "--output",
            "v10.jsonl",
            "--manifest",
            "manifest.json",
            "--model-path",
            "model",
        ]
    )

    assert parsed.batch_size == 8
    assert parsed.runtime_model_path is None
    assert parsed.checkpoint is None
    assert "classify V9 candidate sentences with a local LFM model" in (
        parser.format_help()
    )


def test_cli_runs_v10_and_serializes_its_summary(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "external"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return V10RunSummary(
            output_path=data_root / "artifacts" / "v10.jsonl",
            manifest_path=data_root / "runs" / "v10" / "manifest.json",
            checkpoint_path=data_root / "runs" / "v10" / "checkpoint.jsonl",
            rows_processed=2,
            rows_kept=1,
            rows_filtered=1,
            candidate_sentences_processed=3,
            yes_sentences_written=1,
            no_sentences=2,
            result_sha256="result-sha",
        )

    assert (
        main(
            [
                "filter-v10",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "artifacts" / "v9.jsonl"),
                "--output",
                str(data_root / "artifacts" / "v10.jsonl"),
                "--manifest",
                str(data_root / "runs" / "v10" / "manifest.json"),
                "--model-path",
                "/Volumes/Seagate M3/projects/models/lfm",
                "--runtime-model-path",
                "/Volumes/Seagate M3/projects/models/lfm-mlx",
                "--checkpoint",
                str(data_root / "runs" / "v10" / "checkpoint.jsonl"),
                "--batch-size",
                "4",
                "--max-new-tokens",
                "3",
            ],
            v10_runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].input_path == data_root / "artifacts" / "v9.jsonl"
    assert captured["config"].output_path == data_root / "artifacts" / "v10.jsonl"
    assert captured["config"].manifest_path == (
        data_root / "runs" / "v10" / "manifest.json"
    )
    assert captured["config"].model_path == Path(
        "/Volumes/Seagate M3/projects/models/lfm"
    )
    assert captured["config"].runtime_model_path == Path(
        "/Volumes/Seagate M3/projects/models/lfm-mlx"
    )
    assert captured["config"].checkpoint_path == (
        data_root / "runs" / "v10" / "checkpoint.jsonl"
    )
    assert captured["config"].batch_size == 4
    assert captured["config"].max_new_tokens == 3
    assert json.loads(capsys.readouterr().out) == {
        "candidate_sentences_processed": 3,
        "checkpoint_path": str(data_root / "runs" / "v10" / "checkpoint.jsonl"),
        "manifest_path": str(data_root / "runs" / "v10" / "manifest.json"),
        "no_sentences": 2,
        "output_path": str(data_root / "artifacts" / "v10.jsonl"),
        "result_sha256": "result-sha",
        "rows_filtered": 1,
        "rows_kept": 1,
        "rows_processed": 2,
        "yes_sentences_written": 1,
    }


def test_cli_reports_v10_runner_errors_to_stderr(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "external"

    def failing_runner(config):
        raise ValueError("bad V10 input")

    assert (
        main(
            [
                "filter-v10",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v9.jsonl"),
                "--output",
                str(data_root / "v10.jsonl"),
                "--manifest",
                str(data_root / "manifest.json"),
                "--model-path",
                str(data_root / "model"),
            ],
            v10_runner=failing_runner,
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: bad V10 input\n"


def test_cli_v10_serialization_is_utf8_and_sorted(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "external"
    result_path = data_root / "résultats" / "v10.jsonl"
    manifest_path = data_root / "runs" / "v10" / "manifest.json"
    checkpoint_path = data_root / "runs" / "v10" / "checkpoint.jsonl"

    def fake_runner(config):
        return V10RunSummary(
            output_path=result_path,
            manifest_path=manifest_path,
            checkpoint_path=checkpoint_path,
            rows_processed=2,
            rows_kept=1,
            rows_filtered=1,
            candidate_sentences_processed=3,
            yes_sentences_written=1,
            no_sentences=2,
            result_sha256="result-sha",
        )

    assert (
        main(
            [
                "filter-v10",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v9.jsonl"),
                "--output",
                str(result_path),
                "--manifest",
                str(manifest_path),
                "--model-path",
                str(data_root / "model"),
                "--checkpoint",
                str(checkpoint_path),
            ],
            v10_runner=fake_runner,
        )
        == 0
    )

    expected = {
        "candidate_sentences_processed": 3,
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "no_sentences": 2,
        "output_path": str(result_path),
        "result_sha256": "result-sha",
        "rows_filtered": 1,
        "rows_kept": 1,
        "rows_processed": 2,
        "yes_sentences_written": 1,
    }
    assert capsys.readouterr().out == (
        json.dumps(expected, ensure_ascii=False, sort_keys=True) + "\n"
    )


def test_cli_runs_v9_sentence_topic_filter_and_serializes_its_summary(
    tmp_path: Path, capsys
) -> None:
    data_root = tmp_path / "external"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return V9RunSummary(
            output_path=data_root / "artifacts" / "v9.jsonl",
            manifest_path=data_root / "runs" / "v9" / "manifest.json",
            rows_processed=3,
            rows_kept=1,
            rows_filtered=2,
            sentences_processed=10,
            relevant_sentences_written=1,
            category_sentences={"land_use": 1},
            result_sha256="result-sha",
        )

    assert (
        main(
            [
                "filter-v9",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "artifacts" / "v8.jsonl"),
                "--output",
                str(data_root / "artifacts" / "v9.jsonl"),
                "--manifest",
                str(data_root / "runs" / "v9" / "manifest.json"),
                "--vocabulary",
                str(data_root / "vocabulary.json"),
            ],
            v9_runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].input_path == data_root / "artifacts" / "v8.jsonl"
    assert captured["config"].output_path == data_root / "artifacts" / "v9.jsonl"
    assert captured["config"].manifest_path == (
        data_root / "runs" / "v9" / "manifest.json"
    )
    assert captured["config"].vocabulary_path == data_root / "vocabulary.json"
    assert json.loads(capsys.readouterr().out) == {
        "category_sentences": {"land_use": 1},
        "manifest_path": str(data_root / "runs" / "v9" / "manifest.json"),
        "output_path": str(data_root / "artifacts" / "v9.jsonl"),
        "relevant_sentences_written": 1,
        "result_sha256": "result-sha",
        "rows_filtered": 2,
        "rows_kept": 1,
        "rows_processed": 3,
        "sentences_processed": 10,
    }


def test_cli_v7_passes_explicit_json_serialization_flags(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    data_root = tmp_path / "external"
    captured: dict[str, object] = {}

    def fake_dumps(value: object, **kwargs: object) -> str:
        captured["value"] = value
        captured["kwargs"] = kwargs
        return "serialized"

    def fake_runner(config):
        return V7RunSummary(
            output_path=data_root / "v7.jsonl",
            manifest_path=data_root / "manifest.json",
            rows_processed=1,
            sentences_written=2,
            result_sha256="sha",
        )

    monkeypatch.setattr(cli_module.json, "dumps", fake_dumps)

    assert (
        main(
            [
                "segment-v7",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v6.jsonl"),
                "--output",
                str(data_root / "v7.jsonl"),
                "--manifest",
                str(data_root / "manifest.json"),
            ],
            v7_runner=fake_runner,
        )
        == 0
    )

    assert captured["kwargs"] == {"ensure_ascii": False, "sort_keys": True}
    assert capsys.readouterr().out == "serialized\n"


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
    assert captured["config"].country_name == "Monaco"
    assert json.loads(capsys.readouterr().out) == {
        "manifest_path": str(data_root / "runs" / "default" / "manifest.json"),
        "matches_written": 6,
        "partitions_completed": 3,
        "partitions_skipped": 4,
        "result_path": str(data_root / "artifacts" / "result.jsonl"),
        "rows_scanned": 5,
    }


def test_cli_passes_a_non_default_country_name_to_the_runner(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_root = tmp_path / "external"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return RunSummary(
            result_path=data_root / "artifacts" / "result.jsonl",
            manifest_path=data_root / "runs" / "case" / "manifest.json",
            partitions_completed=0,
            partitions_skipped=0,
            rows_scanned=0,
            matches_written=0,
        )

    assert (
        main(
            [
                "scan",
                "--data-root",
                str(data_root),
                "--shard",
                str(data_root / "raw" / "shard.parquet"),
                "--country-name",
                "Liechtenstein",
            ],
            runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].country_name == "Liechtenstein"
    capsys.readouterr()


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


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        OSError("disk"),
        RuntimeError("runtime"),
        ValueError("bad"),
    ],
)
def test_cli_reports_v7_runner_errors_as_exit_code_two(
    tmp_path: Path, capsys, error: Exception
) -> None:
    def failing_runner(config):
        raise error

    assert (
        main(
            [
                "segment-v7",
                "--data-root",
                str(tmp_path / "external"),
                "--input",
                str(tmp_path / "external" / "v6.jsonl"),
                "--output",
                str(tmp_path / "external" / "v7.jsonl"),
                "--manifest",
                str(tmp_path / "external" / "manifest.json"),
            ],
            v7_runner=failing_runner,
        )
        == 2
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing"),
        OSError("disk"),
        ValueError("bad"),
    ],
)
def test_cli_reports_v8_runner_errors_as_exit_code_two(
    tmp_path: Path, capsys, error: Exception
) -> None:
    def failing_runner(config):
        raise error

    data_root = tmp_path / "external"
    assert (
        main(
            [
                "filter-v8",
                "--data-root",
                str(data_root),
                "--input",
                str(data_root / "v7.jsonl"),
                "--output",
                str(data_root / "v8.jsonl"),
                "--manifest",
                str(data_root / "manifest.json"),
                "--vocabulary",
                str(data_root / "vocabulary.json"),
            ],
            v8_runner=failing_runner,
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {error}\n"
