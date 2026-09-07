"""Command-line front end.

This module knows how to parse arguments, resolve project paths, report
errors, and print a summary. It does not know which versions exist: the
command table lives in `fineweb_polygons.registry`, so adding a version
never changes this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from fineweb_polygons.core.foundation import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    DEFAULT_DATA_ROOT,
    ProjectPaths,
    validate_external_data_root,
)
from fineweb_polygons.registry import Command, commands

FOUNDATION_MESSAGE = "fineweb-polygons foundation only; no pipeline executed"


def main(
    argv: Sequence[str] | None = None,
    **runner_overrides: Callable[[Any], Any],
) -> int:
    """Run the requested command and return a shell exit code.

    Each command's runner may be replaced by keyword, using the runner
    keyword declared for it in the registry (for example `v7_runner=`).
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(FOUNDATION_MESSAGE)
        return 0
    table = {command.name: command for command in commands()}
    _reject_unknown_runners(table.values(), runner_overrides)
    parser = _build_parser(table.values())
    parsed = parser.parse_args(arguments)
    command = table.get(parsed.command)
    if command is None:
        parser.error(f"unknown command: {parsed.command}")
    return _execute(
        command,
        parsed,
        runner_overrides.get(command.runner_keyword, command.runner),
    )


def _build_parser(table: Iterable[Command] | None = None) -> argparse.ArgumentParser:
    table = commands() if table is None else table
    parser = argparse.ArgumentParser(prog="fineweb-polygons")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in table:
        subparser = subparsers.add_parser(command.name, help=command.help)
        for argument in command.arguments:
            options = dict(argument.kwargs)
            if argument.flag == "--data-root":
                options.setdefault("default", DEFAULT_DATA_ROOT)
            subparser.add_argument(argument.flag, **options)
    return parser


def _reject_unknown_runners(
    table: Iterable[Command],
    runner_overrides: dict[str, Callable[[Any], Any]],
) -> None:
    known = {command.runner_keyword for command in table}
    unknown = sorted(set(runner_overrides) - known)
    if unknown:
        raise TypeError(f"unknown runner override: {', '.join(unknown)}")


def _execute(
    command: Command,
    parsed: argparse.Namespace,
    runner: Callable[[Any], Any],
) -> int:
    paths = _project_paths(parsed.data_root)
    try:
        if command.requires_external_root:
            validate_external_data_root(paths)
        summary = runner(command.build_config(parsed, paths))
    except command.errors as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary.to_record(), ensure_ascii=False, sort_keys=True))
    return 0


def _project_paths(data_root: Path) -> ProjectPaths:
    resolved_root = data_root.expanduser().resolve()
    return ProjectPaths.from_environment(
        Path.cwd(),
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(resolved_root)},
    )
