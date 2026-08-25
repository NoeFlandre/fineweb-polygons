"""Command-line entry points for retrieval and V7 post-processing."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from fineweb_polygons.foundation import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    DEFAULT_DATA_ROOT,
    ProjectPaths,
    validate_data_path,
    validate_external_data_root,
)
from fineweb_polygons.runs import RunSummary, ScanRunConfig, execute_run
from fineweb_polygons.v7 import V7RunConfig, V7RunSummary, run_v7

FOUNDATION_MESSAGE = "fineweb-polygons foundation only; no pipeline executed"
Runner = Callable[[ScanRunConfig], RunSummary]
V7Runner = Callable[[V7RunConfig], V7RunSummary]


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = execute_run,
    v7_runner: V7Runner = run_v7,
) -> int:
    """Run the requested command and return a shell exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(FOUNDATION_MESSAGE)
        return 0
    parser = _build_parser()
    parsed = parser.parse_args(arguments)
    if parsed.command == "scan":
        return _run_scan(parsed, runner)
    if parsed.command == "segment-v7":
        return _run_segment_v7(parsed, v7_runner)
    parser.error(f"unknown command: {parsed.command}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fineweb-polygons")
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan", help="scan one FineWeb Parquet shard")
    scan.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    scan.add_argument("--pbf", type=Path)
    scan.add_argument("--shard", type=Path, required=True)
    scan.add_argument("--run-id", default="v1-10bt-000-v2")
    scan.add_argument("--batch-size", type=int, default=8192)
    scan.add_argument("--country-name", default="Monaco")
    scan.add_argument(
        "--retrieval-version",
        choices=("v1", "v2", "v3", "v4", "v5", "v6"),
        default="v1",
    )
    segment = commands.add_parser(
        "segment-v7", help="split V6 documents into exact sentence lists"
    )
    segment.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    segment.add_argument("--input", type=Path, required=True)
    segment.add_argument("--output", type=Path, required=True)
    segment.add_argument("--manifest", type=Path, required=True)
    segment.add_argument("--batch-size", type=int, default=32)
    segment.add_argument("--model-id", choices=("sat-3l-sm",), default="sat-3l-sm")
    return parser


def _run_scan(parsed: argparse.Namespace, runner: Runner) -> int:
    data_root = parsed.data_root.expanduser().resolve()
    paths = ProjectPaths.from_environment(
        Path.cwd(),
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(data_root)},
    )
    pbf_path = parsed.pbf or paths.raw_dir / "monaco-latest.osm.pbf"
    config = ScanRunConfig(
        paths=paths,
        pbf_path=pbf_path,
        shard_path=parsed.shard,
        run_id=parsed.run_id,
        batch_size=parsed.batch_size,
        retrieval_version=parsed.retrieval_version,
        country_name=parsed.country_name,
    )
    try:
        summary = runner(config)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(_summary_record(summary), ensure_ascii=False, sort_keys=True))
    return 0


def _run_segment_v7(parsed: argparse.Namespace, runner: V7Runner) -> int:
    data_root = parsed.data_root.expanduser().resolve()
    paths = ProjectPaths.from_environment(
        Path.cwd(),
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(data_root)},
    )
    try:
        validate_external_data_root(paths)
        config = V7RunConfig(
            input_path=validate_data_path(paths, parsed.input),
            output_path=validate_data_path(paths, parsed.output),
            manifest_path=validate_data_path(paths, parsed.manifest),
            model_id=parsed.model_id,
            batch_size=parsed.batch_size,
        )
        summary = runner(config)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(_v7_summary_record(summary), ensure_ascii=False, sort_keys=True))
    return 0


def _summary_record(summary: RunSummary) -> dict[str, object]:
    return {
        "result_path": str(summary.result_path),
        "manifest_path": str(summary.manifest_path),
        "partitions_completed": summary.partitions_completed,
        "partitions_skipped": summary.partitions_skipped,
        "rows_scanned": summary.rows_scanned,
        "matches_written": summary.matches_written,
    }


def _v7_summary_record(summary: V7RunSummary) -> dict[str, object]:
    return {
        "manifest_path": str(summary.manifest_path),
        "output_path": str(summary.output_path),
        "result_sha256": summary.result_sha256,
        "rows_processed": summary.rows_processed,
        "sentences_written": summary.sentences_written,
    }
