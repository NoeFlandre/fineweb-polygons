from pathlib import Path

import pytest

from fineweb_polygons.foundation import (
    DEFAULT_DATA_ROOT,
    ProjectPaths,
    validate_data_path,
    validate_external_data_root,
)


def test_default_data_root_is_on_seagate(tmp_path: Path) -> None:
    paths = ProjectPaths.from_environment(tmp_path, environ={})

    assert paths.data_root == DEFAULT_DATA_ROOT


def test_environment_can_override_data_root_for_controlled_runs(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external-data"
    paths = ProjectPaths.from_environment(
        tmp_path,
        environ={"FINEWEB_POLYGONS_DATA_ROOT": str(external_root)},
    )

    assert paths.data_root == external_root


def test_ensure_data_layout_creates_only_external_run_directories(
    tmp_path: Path,
) -> None:
    external_root = tmp_path / "external-data"
    paths = ProjectPaths.from_environment(
        tmp_path,
        environ={"FINEWEB_POLYGONS_DATA_ROOT": str(external_root)},
    )

    paths.ensure_data_layout()

    assert {child.name for child in external_root.iterdir()} == {
        "artifacts",
        "logs",
        "raw",
        "runs",
    }
    assert not (tmp_path / "data").exists()


def test_validate_external_data_root_rejects_the_repository_and_children(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "repo"
    for data_root in (repository_root, repository_root / "data"):
        paths = ProjectPaths(repository_root=repository_root, data_root=data_root)

        with pytest.raises(
            ValueError,
            match=r"^The data root must be external to the repository$",
        ):
            validate_external_data_root(paths)


def test_validate_data_path_requires_an_external_root_child(tmp_path: Path) -> None:
    data_root = tmp_path / "external"
    paths = ProjectPaths(repository_root=tmp_path / "repo", data_root=data_root)
    child = data_root / "raw" / "shard.parquet"

    assert validate_data_path(paths, child) == child.resolve()
    with pytest.raises(ValueError, match=r"^Input path must be inside"):
        validate_data_path(paths, tmp_path / "outside.parquet")
