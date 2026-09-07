"""The single declaration of every research direction, version, and command.

Adding a version means adding one `Version` (and, if it needs its own entry
point, one `Command`) to the tables at the bottom of this module. Nothing in
`cli.py`, the dataset catalog, or the documentation has to change shape for a
new version, and nothing else in the codebase may declare a version.

A `Version` is the immutable public contract: its ID, the Hugging Face
configuration that publishes it, and the dataset prefix its files live under.
A `Command` is the command-line surface that produces it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fineweb_polygons.core.foundation import ProjectPaths, validate_data_path
from fineweb_polygons.directions import lexical, retrieval
from fineweb_polygons.directions.lexical.v1 import models as lexical_v1
from fineweb_polygons.directions.lexical.v2 import models as lexical_v2

DATASET_REPOSITORY = "NoeFlandre/fineweb-polygons"


@dataclass(frozen=True, slots=True)
class Version:
    """One immutable step inside a direction."""

    id: str
    summary: str
    countries: tuple[str, ...]
    hf_config: str
    data_prefix: str
    files: tuple[tuple[str, str], ...]
    source_version: str | None = None
    country_metadata: tuple[str, ...] = ("manifest.json",)
    version_metadata: tuple[str, ...] = ()

    @property
    def metadata_prefix(self) -> str:
        """Return the dataset prefix holding this version's manifests."""
        return self.data_prefix.replace("data/", "metadata/", 1)

    @property
    def card(self) -> str:
        """Return the path of this version's standalone dataset card."""
        return f"{self.data_prefix}/README.md"

    def data_files(self) -> tuple[dict[str, str], ...]:
        """Return the Hugging Face split-to-path mapping for this version."""
        return tuple(
            {"split": split, "path": f"{self.data_prefix}/{name}"}
            for split, name in self.files
        )

    def metadata_files(self) -> tuple[str, ...]:
        """Return every published metadata path for this version."""
        paths = [f"{self.metadata_prefix}/{name}" for name in self.version_metadata]
        for country in self.countries:
            paths.extend(
                f"{self.metadata_prefix}/{country}/{name}"
                for name in self.country_metadata
            )
        return tuple(sorted(paths))


@dataclass(frozen=True, slots=True)
class Direction:
    """A coherent line of experiments with its own question and contract."""

    id: str
    name: str
    package: str
    status: str
    documentation: str
    versions: tuple[Version, ...]

    @property
    def latest_version(self) -> Version:
        """Return the newest version declared for this direction."""
        return self.versions[-1]

    def version(self, version_id: str) -> Version:
        """Return one declared version or raise for an unknown ID."""
        for version in self.versions:
            if version.id == version_id:
                return version
        raise KeyError(f"unknown version for {self.id}: {version_id}")


@dataclass(frozen=True, slots=True)
class Argument:
    """One declarative command-line option."""

    flag: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Command:
    """One command-line entry point and how to run it."""

    name: str
    help: str
    direction: str
    produces: tuple[str, ...]
    arguments: tuple[Argument, ...]
    build_config: Callable[[argparse.Namespace, ProjectPaths], Any]
    runner: Callable[[Any], Any]
    runner_keyword: str
    requires_external_root: bool = True
    errors: tuple[type[BaseException], ...] = (
        FileNotFoundError,
        OSError,
        RuntimeError,
        ValueError,
    )


# --------------------------------------------------------------------------
# Direction 1 - FineWeb polygon retrieval (frozen at V10)
# --------------------------------------------------------------------------

_RETRIEVAL_PREFIX = "data/direction-1-retrieval"
_BOTH = ("monaco", "liechtenstein")

RETRIEVAL = Direction(
    id="direction-1-fineweb-retrieval",
    name="FineWeb polygon retrieval",
    package="fineweb_polygons.directions.retrieval",
    status="frozen",
    documentation="docs/directions/fineweb-retrieval/README.md",
    versions=(
        Version(
            "v1",
            "Named polygon exact matching; excerpt-only release.",
            ("monaco",),
            "v1",
            f"{_RETRIEVAL_PREFIX}/v1",
            (("train", "monaco.jsonl"),),
        ),
        Version(
            "v2",
            "Meaningful in-boundary exact matching with full text.",
            ("monaco",),
            "v2",
            f"{_RETRIEVAL_PREFIX}/v2",
            (("train", "monaco.jsonl"),),
        ),
        Version(
            "v3",
            "All meaningful areas with strict URL-and-text matching.",
            ("monaco",),
            "v3",
            f"{_RETRIEVAL_PREFIX}/v3",
            (("train", "monaco.jsonl"),),
        ),
        Version(
            "v4",
            "All meaningful areas with text-only matching.",
            ("monaco",),
            "v4",
            f"{_RETRIEVAL_PREFIX}/v4",
            (("train", "monaco.jsonl"),),
        ),
        Version(
            "v5",
            "Specific polygon names with country-in-text matching.",
            _BOTH,
            "v5",
            f"{_RETRIEVAL_PREFIX}/v5",
            (
                ("monaco", "monaco.jsonl"),
                ("liechtenstein", "liechtenstein.jsonl"),
            ),
            country_metadata=("manifest.json", "name-frequency.json"),
        ),
        Version(
            "v6",
            "V5 names constrained to a 500-character local text span.",
            _BOTH,
            "v6",
            f"{_RETRIEVAL_PREFIX}/v6",
            (
                ("monaco", "monaco.jsonl"),
                ("liechtenstein", "liechtenstein.jsonl"),
            ),
            source_version="v5",
            country_metadata=("manifest.json", "name-frequency.json"),
        ),
        Version(
            "v7",
            "Exact sentence lists added to the V6 rows.",
            _BOTH,
            "v7",
            f"{_RETRIEVAL_PREFIX}/v7",
            (
                ("monaco", "monaco.jsonl"),
                ("liechtenstein", "liechtenstein.jsonl"),
            ),
            source_version="v6",
        ),
        Version(
            "v8",
            "Document-level topic-vocabulary filtering of V7.",
            _BOTH,
            "v8",
            f"{_RETRIEVAL_PREFIX}/v8",
            (
                ("monaco", "monaco.jsonl"),
                ("liechtenstein", "liechtenstein.jsonl"),
            ),
            source_version="v7",
            version_metadata=("topic-vocabulary-v1.json",),
        ),
        Version(
            "v9",
            "Sentence-level topic filtering near polygon evidence.",
            _BOTH,
            "v9",
            f"{_RETRIEVAL_PREFIX}/v9",
            (
                ("monaco", "monaco.jsonl"),
                ("liechtenstein", "liechtenstein.jsonl"),
            ),
            source_version="v8",
        ),
        Version(
            "v10",
            "Local LFM land-use classification of V9 candidate sentences.",
            _BOTH,
            "v10",
            f"{_RETRIEVAL_PREFIX}/v10",
            (
                ("monaco", "monaco.jsonl"),
                ("liechtenstein", "liechtenstein.jsonl"),
            ),
            source_version="v9",
        ),
    ),
)


# --------------------------------------------------------------------------
# Direction 2 - lexical polygon candidates (active POC)
# --------------------------------------------------------------------------

LEXICAL = Direction(
    id="direction-2-lexical-candidates",
    name="Lexical polygon candidates",
    package="fineweb_polygons.directions.lexical",
    status="active_poc",
    documentation="docs/directions/lexical-candidates/README.md",
    versions=(
        Version(
            lexical_v1.DIRECTION_VERSION,
            "Broad all-area Aho-Corasick lexical baseline.",
            _BOTH,
            lexical_v1.HF_CONFIG_NAME,
            lexical_v1.DATA_PREFIX,
            (
                ("monaco", "monaco.parquet"),
                ("liechtenstein", "liechtenstein.parquet"),
            ),
            country_metadata=(),
            version_metadata=("manifest.json",),
        ),
        Version(
            lexical_v2.DIRECTION_V2_VERSION,
            "Specificity-aware gating of generic names by measured reuse.",
            _BOTH,
            lexical_v2.HF_CONFIG_NAME_V2,
            lexical_v2.DATA_PREFIX,
            (
                ("monaco", "monaco.parquet"),
                ("liechtenstein", "liechtenstein.parquet"),
            ),
            country_metadata=(),
            version_metadata=("manifest.json", "name-inventory.json"),
        ),
    ),
)

DIRECTIONS: tuple[Direction, ...] = (RETRIEVAL, LEXICAL)


def direction(direction_id: str) -> Direction:
    """Return one declared direction or raise for an unknown ID."""
    for candidate in DIRECTIONS:
        if candidate.id == direction_id:
            return candidate
    raise KeyError(f"unknown direction: {direction_id}")


# --------------------------------------------------------------------------
# Command-line surface
# --------------------------------------------------------------------------

_DATA_ROOT = Argument("--data-root", {"type": Path})
_STAGE_ARGUMENTS = (
    _DATA_ROOT,
    Argument("--input", {"type": Path, "required": True}),
    Argument("--output", {"type": Path, "required": True}),
    Argument("--manifest", {"type": Path, "required": True}),
)
_LEXICAL_ARGUMENTS = (
    _DATA_ROOT,
    Argument("--monaco-pbf", {"type": Path}),
    Argument("--liechtenstein-pbf", {"type": Path}),
    Argument("--shard", {"type": Path, "required": True}),
    Argument("--output-dir", {"type": Path}),
    Argument("--manifest", {"type": Path}),
    Argument("--dataset-card", {"type": Path}),
    Argument("--log", {"type": Path}),
    Argument("--batch-size", {"type": int, "default": 8192}),
    Argument("--output-batch-size", {"type": int, "default": 4096}),
)


def _scan_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> retrieval.ScanRunConfig:
    return retrieval.ScanRunConfig(
        paths=paths,
        pbf_path=parsed.pbf or paths.raw_dir / "monaco-latest.osm.pbf",
        shard_path=parsed.shard,
        run_id=parsed.run_id,
        batch_size=parsed.batch_size,
        retrieval_version=parsed.retrieval_version,
        country_name=parsed.country_name,
    )


def _v7_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> retrieval.V7RunConfig:
    return retrieval.V7RunConfig(
        input_path=validate_data_path(paths, parsed.input),
        output_path=validate_data_path(paths, parsed.output),
        manifest_path=validate_data_path(paths, parsed.manifest),
        model_id=parsed.model_id,
        batch_size=parsed.batch_size,
    )


def _v8_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> retrieval.V8RunConfig:
    return retrieval.V8RunConfig(
        input_path=validate_data_path(paths, parsed.input),
        output_path=validate_data_path(paths, parsed.output),
        manifest_path=validate_data_path(paths, parsed.manifest),
        vocabulary_path=validate_data_path(paths, parsed.vocabulary),
    )


def _v9_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> retrieval.V9RunConfig:
    return retrieval.V9RunConfig(
        input_path=validate_data_path(paths, parsed.input),
        output_path=validate_data_path(paths, parsed.output),
        manifest_path=validate_data_path(paths, parsed.manifest),
        vocabulary_path=validate_data_path(paths, parsed.vocabulary),
    )


def _v10_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> retrieval.V10RunConfig:
    return retrieval.V10RunConfig(
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
    )


def _lexical_path(paths: ProjectPaths, value: Path | None, default: Path) -> Path:
    return validate_data_path(paths, default if value is None else value)


def _lexical_defaults(
    parsed: argparse.Namespace, paths: ProjectPaths, slug: str
) -> dict[str, Any]:
    return {
        "monaco_pbf": _lexical_path(
            paths, parsed.monaco_pbf, paths.raw_dir / "monaco-latest.osm.pbf"
        ),
        "liechtenstein_pbf": _lexical_path(
            paths,
            parsed.liechtenstein_pbf,
            paths.raw_dir / "liechtenstein-latest.osm.pbf",
        ),
        "shard_path": validate_data_path(paths, parsed.shard),
        "output_dir": _lexical_path(
            paths, parsed.output_dir, paths.artifacts_dir / slug
        ),
        "manifest_path": _lexical_path(
            paths, parsed.manifest, paths.runs_dir / slug / "manifest.json"
        ),
        "dataset_card_path": _lexical_path(
            paths, parsed.dataset_card, paths.artifacts_dir / slug / "dataset-card.md"
        ),
        "log_path": _lexical_path(
            paths, parsed.log, paths.logs_dir / slug / "run.jsonl"
        ),
        "batch_size": parsed.batch_size,
        "output_batch_size": parsed.output_batch_size,
    }


def _lexical_v1_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> lexical.Direction2RunConfig:
    return lexical.Direction2RunConfig(
        **_lexical_defaults(parsed, paths, "direction-2/lexical-v1")
    )


def _lexical_v2_config(
    parsed: argparse.Namespace, paths: ProjectPaths
) -> lexical.Direction2V2RunConfig:
    slug = "direction-2/lexical-v2"
    return lexical.Direction2V2RunConfig(
        **_lexical_defaults(parsed, paths, slug),
        name_inventory_path=_lexical_path(
            paths, parsed.name_inventory, paths.runs_dir / slug / "name-inventory.json"
        ),
    )


COMMANDS: tuple[Command, ...] = (
    Command(
        name="scan",
        help="scan one FineWeb Parquet shard",
        direction=RETRIEVAL.id,
        produces=("v1", "v2", "v3", "v4", "v5", "v6"),
        arguments=(
            _DATA_ROOT,
            Argument("--pbf", {"type": Path}),
            Argument("--shard", {"type": Path, "required": True}),
            Argument("--run-id", {"default": "v1-10bt-000-v2"}),
            Argument("--batch-size", {"type": int, "default": 8192}),
            Argument("--country-name", {"default": "Monaco"}),
            Argument(
                "--retrieval-version",
                {"choices": ("v1", "v2", "v3", "v4", "v5", "v6"), "default": "v1"},
            ),
        ),
        build_config=_scan_config,
        runner=retrieval.execute_run,
        runner_keyword="runner",
        requires_external_root=False,
        errors=(FileNotFoundError, OSError, ValueError),
    ),
    Command(
        name="segment-v7",
        help="split V6 documents into exact sentence lists",
        direction=RETRIEVAL.id,
        produces=("v7",),
        arguments=(
            *_STAGE_ARGUMENTS,
            Argument("--batch-size", {"type": int, "default": 32}),
            Argument("--model-id", {"choices": ("sat-3l-sm",), "default": "sat-3l-sm"}),
        ),
        build_config=_v7_config,
        runner=retrieval.run_v7,
        runner_keyword="v7_runner",
    ),
    Command(
        name="filter-v8",
        help="filter V7 documents with the approved topic vocabulary",
        direction=RETRIEVAL.id,
        produces=("v8",),
        arguments=(
            *_STAGE_ARGUMENTS,
            Argument("--vocabulary", {"type": Path, "required": True}),
        ),
        build_config=_v8_config,
        runner=retrieval.run_v8,
        runner_keyword="v8_runner",
    ),
    Command(
        name="filter-v9",
        help="filter V8 rows to local topic-relevant sentences",
        direction=RETRIEVAL.id,
        produces=("v9",),
        arguments=(
            *_STAGE_ARGUMENTS,
            Argument("--vocabulary", {"type": Path, "required": True}),
        ),
        build_config=_v9_config,
        runner=retrieval.run_v9,
        runner_keyword="v9_runner",
    ),
    Command(
        name="filter-v10",
        help="classify V9 candidate sentences with a local LFM model",
        direction=RETRIEVAL.id,
        produces=("v10",),
        arguments=(
            *_STAGE_ARGUMENTS,
            Argument("--model-path", {"type": Path, "required": True}),
            Argument("--runtime-model-path", {"type": Path}),
            Argument("--checkpoint", {"type": Path}),
            Argument("--batch-size", {"type": int, "default": 8}),
            Argument(
                "--max-new-tokens",
                {"type": int, "default": retrieval.V10_MAX_NEW_TOKENS},
            ),
        ),
        build_config=_v10_config,
        runner=retrieval.run_v10,
        runner_keyword="v10_runner",
    ),
    Command(
        name="direction2-lexical-v1",
        help="scan FineWeb for OSM polygon name candidates with Aho-Corasick",
        direction=LEXICAL.id,
        produces=("direction-2-lexical-v1",),
        arguments=_LEXICAL_ARGUMENTS,
        build_config=_lexical_v1_config,
        runner=lexical.run_direction2,
        runner_keyword="direction2_runner",
    ),
    Command(
        name="direction2-lexical-v2",
        help="scan FineWeb with specificity-aware Aho-Corasick matching",
        direction=LEXICAL.id,
        produces=("direction-2-lexical-v2",),
        arguments=(
            *_LEXICAL_ARGUMENTS,
            Argument("--name-inventory", {"type": Path}),
        ),
        build_config=_lexical_v2_config,
        runner=lexical.run_direction2_v2,
        runner_keyword="direction2_v2_runner",
    ),
)


def commands() -> Sequence[Command]:
    """Return every declared command-line entry point."""
    return COMMANDS
