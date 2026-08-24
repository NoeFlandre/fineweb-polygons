from pathlib import Path

from fineweb_polygons.foundation import DEFAULT_DATA_ROOT, ProjectPaths


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
