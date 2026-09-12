from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW_PATH = _REPOSITORY_ROOT / ".github" / "workflows" / "quality.yml"
_JUSTFILE_PATH = _REPOSITORY_ROOT / "justfile"
_MKDOCS_PATH = _REPOSITORY_ROOT / "mkdocs.yml"
_PYPROJECT_PATH = _REPOSITORY_ROOT / "pyproject.toml"
_DOCKERIGNORE_PATH = _REPOSITORY_ROOT / ".dockerignore"
_DEVELOPMENT_DOC_PATH = _REPOSITORY_ROOT / "docs" / "development.md"
_FOUNDATION_DOC_PATH = _REPOSITORY_ROOT / "docs" / "architecture" / "foundation.md"
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


def test_justfile_has_one_configurable_data_root() -> None:
    justfile = _JUSTFILE_PATH.read_text(encoding="utf-8")

    assert (
        'data_root := env_var_or_default("FINEWEB_POLYGONS_DATA_ROOT", '
        f'"{_SEAGATE_ROOT}")' in justfile
    )
    assert justfile.count(_SEAGATE_ROOT) == 1
    assert "cache/uv-cleanup" not in justfile
    assert ".venvs/fineweb-polygons-v8" not in justfile
    assert 'uv build --out-dir "{{ data_root }}/dist"' in justfile
    assert (
        "qa: lock-check format-check lint typecheck catalog-check "
        "test property acceptance "
        "architecture crap docs package mutation smoke" in justfile
    )


def test_justfile_types_all_checked_code_and_exposes_quality_lanes() -> None:
    justfile = _JUSTFILE_PATH.read_text(encoding="utf-8")

    assert "uv run ty check src tests scripts" in justfile
    for target in ("property:", "acceptance:", "architecture:", "smoke:"):
        assert f"\n{target} prepare" in justfile


def test_pytest_registers_explicit_verification_markers() -> None:
    pyproject = _PYPROJECT_PATH.read_text(encoding="utf-8")

    for marker in ("acceptance", "architecture", "property"):
        assert f'"{marker}:' in pyproject


def test_source_distribution_has_an_explicit_allowlist() -> None:
    pyproject = _PYPROJECT_PATH.read_text(encoding="utf-8")

    assert "[tool.hatch.build.targets.sdist]" in pyproject
    assert "only-include = [" in pyproject
    assert '"src"' in pyproject
    assert '"README.md"' in pyproject
    assert '"LICENSE"' in pyproject


def test_coverage_has_a_high_minimum_threshold() -> None:
    pyproject = _PYPROJECT_PATH.read_text(encoding="utf-8")

    assert "[tool.coverage.report]" in pyproject
    assert "fail_under = 98" in pyproject


def test_mkdocs_default_site_directory_is_portable() -> None:
    mkdocs = _MKDOCS_PATH.read_text(encoding="utf-8")

    assert "site_dir: site" in mkdocs
    assert _SEAGATE_ROOT not in mkdocs


def test_docker_context_excludes_local_research_and_build_artifacts() -> None:
    dockerignore = _DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()

    for entry in (
        "archive",
        "artifacts",
        "cache",
        "dist",
        ".hypothesis",
        ".pytest_cache",
        ".ruff_cache",
        "logs",
        "models",
        "quality",
        "raw",
        "runs",
        "site",
        "tmp",
        ".uv-cache-quality",
        ".venv-quality",
    ):
        assert entry in dockerignore


def test_model_documentation_does_not_reference_another_checkout() -> None:
    development = _DEVELOPMENT_DOC_PATH.read_text(encoding="utf-8")

    assert "FINEWEB_POLYGONS_MODEL_PATH" in development
    assert "osm-polygon-web-search" not in development


def test_architecture_documentation_names_real_modules() -> None:
    foundation = _FOUNDATION_DOC_PATH.read_text(encoding="utf-8")

    for stale_module in (
        "run_models.py",
        "v9_models.py",
        "v10_models.py",
        "v10_inference.py",
    ):
        assert stale_module not in foundation


def test_mutation_workflow_runs_the_fail_closed_gate() -> None:
    if not _WORKFLOW_PATH.is_file():
        pytest.skip("GitHub workflow is unavailable in a mutation checkout")

    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    mutation_run = "- run: uv run mutmut run --max-children 1"
    mutation_gate = "- run: uv run python scripts/check_mutation.py"

    assert mutation_run in workflow
    assert mutation_gate in workflow
    assert workflow.index(mutation_run) < workflow.index(mutation_gate)
    mutation_job = workflow[workflow.index("  mutation:") :]
    assert "if:" not in mutation_job


def test_quality_workflow_builds_the_package() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "uv build --out-dir" in workflow


def test_quality_workflow_exposes_repository_import_paths() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "PYTHONPATH: src:." in workflow


def test_quality_workflow_runs_all_explicit_verification_lanes() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    for command in (
        "uv run ty check src tests scripts",
        "uv run pytest --no-cov -m property",
        "uv run pytest --no-cov -m acceptance",
        "uv run pytest --no-cov -m architecture",
        "uv run fineweb-polygons",
    ):
        assert command in workflow


def test_quality_workflow_smoke_tests_the_docker_runtime() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "docker build --tag fineweb-polygons:ci ." in workflow
    assert "docker run --rm fineweb-polygons:ci" in workflow


def test_quality_workflow_writes_docs_to_runner_workspace() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")

    assert (
        '- run: uv run mkdocs build --strict --site-dir "${{ github.workspace }}/site"'
        in workflow
    )


def test_mutation_checkout_copies_justfile() -> None:
    config = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))
    also_copy = config["tool"]["mutmut"]["also_copy"]

    assert "justfile" in also_copy
    assert "Dockerfile" in also_copy
    assert ".dockerignore" in also_copy
    assert "README.md" in also_copy
    assert "tests/" in also_copy
