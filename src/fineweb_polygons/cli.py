"""Command-line entry points for retrieval and post-processing versions."""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from fineweb_polygons.direction2 import (
    Direction2RunConfig,
    Direction2RunSummary,
    run_direction2,
)
from fineweb_polygons.foundation import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    DEFAULT_DATA_ROOT,
    ProjectPaths,
    validate_data_path,
    validate_external_data_root,
)
from fineweb_polygons.runs import RunSummary, ScanRunConfig, execute_run
from fineweb_polygons.v7 import V7RunConfig, V7RunSummary, run_v7
from fineweb_polygons.v8 import V8RunConfig, V8RunSummary, run_v8
from fineweb_polygons.v9 import V9RunConfig, V9RunSummary, run_v9
from fineweb_polygons.v10 import (
    V10_MAX_NEW_TOKENS,
    V10RunConfig,
    V10RunSummary,
    run_v10,
)

FOUNDATION_MESSAGE = "fineweb-polygons foundation only; no pipeline executed"
Runner = Callable[[ScanRunConfig], RunSummary]
V7Runner = Callable[[V7RunConfig], V7RunSummary]
V8Runner = Callable[[V8RunConfig], V8RunSummary]
V9Runner = Callable[[V9RunConfig], V9RunSummary]
V10Runner = Callable[[V10RunConfig], V10RunSummary]
Direction2Runner = Callable[[Direction2RunConfig], Direction2RunSummary]


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: Runner = execute_run,
    v7_runner: V7Runner = run_v7,
    v8_runner: V8Runner = run_v8,
    v9_runner: V9Runner = run_v9,
    v10_runner: V10Runner = run_v10,
    direction2_runner: Direction2Runner = run_direction2,
) -> int:
    """Run the requested command and return a shell exit code."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(FOUNDATION_MESSAGE)
        return 0
    parser = _build_parser()
    parsed = parser.parse_args(arguments)
    handlers: dict[str, Callable[[], int]] = {
        "scan": lambda: _run_scan(parsed, runner),
        "segment-v7": lambda: _run_segment_v7(parsed, v7_runner),
        "filter-v8": lambda: _run_filter_v8(parsed, v8_runner),
        "filter-v9": lambda: _run_filter_v9(parsed, v9_runner),
        "filter-v10": lambda: _run_filter_v10(parsed, v10_runner),
        "direction2-lexical-v1": lambda: _run_direction2(parsed, direction2_runner),
    }
    handler = handlers.get(parsed.command)
    if handler is None:
        parser.error(f"unknown command: {parsed.command}")
    return handler()


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
    topic = commands.add_parser(
        "filter-v8", help="filter V7 documents with the approved topic vocabulary"
    )
    topic.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    topic.add_argument("--input", type=Path, required=True)
    topic.add_argument("--output", type=Path, required=True)
    topic.add_argument("--manifest", type=Path, required=True)
    topic.add_argument("--vocabulary", type=Path, required=True)
    sentence_topic = commands.add_parser(
        "filter-v9", help="filter V8 rows to local topic-relevant sentences"
    )
    sentence_topic.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    sentence_topic.add_argument("--input", type=Path, required=True)
    sentence_topic.add_argument("--output", type=Path, required=True)
    sentence_topic.add_argument("--manifest", type=Path, required=True)
    sentence_topic.add_argument("--vocabulary", type=Path, required=True)
    llm = commands.add_parser(
        "filter-v10", help="classify V9 candidate sentences with a local LFM model"
    )
    llm.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    llm.add_argument("--input", type=Path, required=True)
    llm.add_argument("--output", type=Path, required=True)
    llm.add_argument("--manifest", type=Path, required=True)
    llm.add_argument("--model-path", type=Path, required=True)
    llm.add_argument("--runtime-model-path", type=Path)
    llm.add_argument("--checkpoint", type=Path)
    llm.add_argument("--batch-size", type=int, default=8)
    llm.add_argument("--max-new-tokens", type=int, default=V10_MAX_NEW_TOKENS)
    lexical = commands.add_parser(
        "direction2-lexical-v1",
        help="scan FineWeb for OSM polygon name candidates with Aho-Corasick",
    )
    lexical.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    lexical.add_argument("--monaco-pbf", type=Path)
    lexical.add_argument("--liechtenstein-pbf", type=Path)
    lexical.add_argument("--shard", type=Path, required=True)
    lexical.add_argument("--output-dir", type=Path)
    lexical.add_argument("--manifest", type=Path)
    lexical.add_argument("--dataset-card", type=Path)
    lexical.add_argument("--log", type=Path)
    lexical.add_argument("--batch-size", type=int, default=8192)
    lexical.add_argument("--output-batch-size", type=int, default=4096)
    return parser


def _run_scan(parsed: argparse.Namespace, runner: Runner) -> int:
    paths = _project_paths(parsed.data_root)
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
    _print_summary(summary, _summary_record)
    return 0


def _run_segment_v7(parsed: argparse.Namespace, runner: V7Runner) -> int:
    return _run_external_stage(
        parsed.data_root,
        config_factory=lambda paths: V7RunConfig(
            input_path=validate_data_path(paths, parsed.input),
            output_path=validate_data_path(paths, parsed.output),
            manifest_path=validate_data_path(paths, parsed.manifest),
            model_id=parsed.model_id,
            batch_size=parsed.batch_size,
        ),
        runner=runner,
        summary_record=_v7_summary_record,
    )


def _run_filter_v8(parsed: argparse.Namespace, runner: V8Runner) -> int:
    return _run_external_stage(
        parsed.data_root,
        config_factory=lambda paths: V8RunConfig(
            input_path=validate_data_path(paths, parsed.input),
            output_path=validate_data_path(paths, parsed.output),
            manifest_path=validate_data_path(paths, parsed.manifest),
            vocabulary_path=validate_data_path(paths, parsed.vocabulary),
        ),
        runner=runner,
        summary_record=_v8_summary_record,
    )


def _run_filter_v9(parsed: argparse.Namespace, runner: V9Runner) -> int:
    return _run_external_stage(
        parsed.data_root,
        config_factory=lambda paths: V9RunConfig(
            input_path=validate_data_path(paths, parsed.input),
            output_path=validate_data_path(paths, parsed.output),
            manifest_path=validate_data_path(paths, parsed.manifest),
            vocabulary_path=validate_data_path(paths, parsed.vocabulary),
        ),
        runner=runner,
        summary_record=_v9_summary_record,
    )


def _run_filter_v10(parsed: argparse.Namespace, runner: V10Runner) -> int:
    return _run_external_stage(
        parsed.data_root,
        config_factory=lambda paths: V10RunConfig(
            input_path=validate_data_path(paths, parsed.input),
            output_path=validate_data_path(paths, parsed.output),
            manifest_path=validate_data_path(paths, parsed.manifest),
            model_path=parsed.model_path.expanduser().resolve(),
            runtime_model_path=(
                parsed.runtime_model_path.expanduser().resolve()
                if parsed.runtime_model_path is not None
                else None
            ),
            checkpoint_path=(
                validate_data_path(paths, parsed.checkpoint)
                if parsed.checkpoint is not None
                else None
            ),
            batch_size=parsed.batch_size,
            max_new_tokens=parsed.max_new_tokens,
        ),
        runner=runner,
        summary_record=_v10_summary_record,
    )


def _run_direction2(parsed: argparse.Namespace, runner: Direction2Runner) -> int:
    return _run_external_stage(
        parsed.data_root,
        config_factory=lambda paths: _direction2_config(parsed, paths),
        runner=runner,
        summary_record=_direction2_summary_record,
    )


def _direction2_config(
    parsed: argparse.Namespace,
    paths: ProjectPaths,
) -> Direction2RunConfig:
    return Direction2RunConfig(
        monaco_pbf=_direction2_path(
            paths, parsed.monaco_pbf, paths.raw_dir / "monaco-latest.osm.pbf"
        ),
        liechtenstein_pbf=_direction2_path(
            paths,
            parsed.liechtenstein_pbf,
            paths.raw_dir / "liechtenstein-latest.osm.pbf",
        ),
        shard_path=validate_data_path(paths, parsed.shard),
        output_dir=_direction2_path(
            paths, parsed.output_dir, paths.artifacts_dir / "direction-2/lexical-v1"
        ),
        manifest_path=_direction2_path(
            paths,
            parsed.manifest,
            paths.runs_dir / "direction-2/lexical-v1/manifest.json",
        ),
        dataset_card_path=_direction2_path(
            paths,
            parsed.dataset_card,
            paths.artifacts_dir / "direction-2/lexical-v1/dataset-card.md",
        ),
        log_path=_direction2_path(
            paths,
            parsed.log,
            paths.logs_dir / "direction-2/lexical-v1/run.jsonl",
        ),
        batch_size=parsed.batch_size,
        output_batch_size=parsed.output_batch_size,
    )


def _direction2_path(
    paths: ProjectPaths,
    value: Path | None,
    default: Path,
) -> Path:
    return validate_data_path(paths, default if value is None else value)


def _project_paths(data_root: Path) -> ProjectPaths:
    resolved_root = data_root.expanduser().resolve()
    return ProjectPaths.from_environment(
        Path.cwd(),
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(resolved_root)},
    )


def _run_external_stage[ConfigT, SummaryT](
    data_root: Path,
    *,
    config_factory: Callable[[ProjectPaths], ConfigT],
    runner: Callable[[ConfigT], SummaryT],
    summary_record: Callable[[SummaryT], dict[str, object]],
) -> int:
    paths = _project_paths(data_root)
    try:
        validate_external_data_root(paths)
        summary = runner(config_factory(paths))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    _print_summary(summary, summary_record)
    return 0


def _print_summary[SummaryT](
    summary: SummaryT,
    summary_record: Callable[[SummaryT], dict[str, object]],
) -> None:
    print(json.dumps(summary_record(summary), ensure_ascii=False, sort_keys=True))


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


def _v8_summary_record(summary: V8RunSummary) -> dict[str, object]:
    return {
        "category_documents": summary.category_documents,
        "manifest_path": str(summary.manifest_path),
        "output_path": str(summary.output_path),
        "result_sha256": summary.result_sha256,
        "rows_filtered": summary.rows_filtered,
        "rows_kept": summary.rows_kept,
        "rows_processed": summary.rows_processed,
    }


def _v9_summary_record(summary: V9RunSummary) -> dict[str, object]:
    return {
        "category_sentences": summary.category_sentences,
        "manifest_path": str(summary.manifest_path),
        "output_path": str(summary.output_path),
        "relevant_sentences_written": summary.relevant_sentences_written,
        "result_sha256": summary.result_sha256,
        "rows_filtered": summary.rows_filtered,
        "rows_kept": summary.rows_kept,
        "rows_processed": summary.rows_processed,
        "sentences_processed": summary.sentences_processed,
    }


def _v10_summary_record(summary: V10RunSummary) -> dict[str, object]:
    return {
        "candidate_sentences_processed": summary.candidate_sentences_processed,
        "checkpoint_path": str(summary.checkpoint_path),
        "manifest_path": str(summary.manifest_path),
        "no_sentences": summary.no_sentences,
        "output_path": str(summary.output_path),
        "result_sha256": summary.result_sha256,
        "rows_filtered": summary.rows_filtered,
        "rows_kept": summary.rows_kept,
        "rows_processed": summary.rows_processed,
        "yes_sentences_written": summary.yes_sentences_written,
    }


def _direction2_summary_record(
    summary: Direction2RunSummary,
) -> dict[str, object]:
    return summary.to_record()
