from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github" / "workflows" / "quality.yml"
_SEAGATE_ROOT = "/Volumes/Seagate M3/projects/fineweb-polygons"


def _coverage_config(
    *, coverage_file: Path | None = None, json_file: Path | None = None
) -> str:
    environment = os.environ.copy()
    environment.pop("COVERAGE_FILE", None)
    environment.pop("COVERAGE_JSON", None)
    if coverage_file is not None:
        environment["COVERAGE_FILE"] = str(coverage_file)
    if json_file is not None:
        environment["COVERAGE_JSON"] = str(json_file)
    result = subprocess.run(
        [sys.executable, "-m", "coverage", "debug", "config"],
        check=False,
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _has_config_value(output: str, key: str, value: str) -> bool:
    return any(line.strip() == f"{key}: {value}" for line in output.splitlines())


def test_coverage_keeps_seagate_defaults_without_overrides() -> None:
    output = _coverage_config()

    assert _has_config_value(output, "data_file", f"{_SEAGATE_ROOT}/.coverage")
    assert _has_config_value(output, "json_output", f"{_SEAGATE_ROOT}/coverage.json")


def test_coverage_accepts_workspace_paths_for_ci(tmp_path: Path) -> None:
    output = _coverage_config(
        coverage_file=tmp_path / ".coverage",
        json_file=tmp_path / "coverage.json",
    )

    assert _has_config_value(output, "data_file", str(tmp_path / ".coverage"))
    assert _has_config_value(output, "json_output", str(tmp_path / "coverage.json"))


def test_manual_mutation_workflow_runs_the_fail_closed_gate() -> None:
    if not _WORKFLOW_PATH.is_file():
        pytest.skip("GitHub workflow is unavailable in a mutation checkout")

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    mutation_run = "- run: uv run mutmut run --max-children 1"
    mutation_gate = "- run: uv run python scripts/check_mutation.py"

    assert mutation_run in workflow
    assert mutation_gate in workflow
    assert workflow.index(mutation_run) < workflow.index(mutation_gate)
