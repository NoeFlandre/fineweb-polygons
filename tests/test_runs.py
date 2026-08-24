from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.foundation import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    ProjectPaths,
)
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.runs import ScanRunConfig, execute_run


def write_shard(path: Path, first_text: str = "Fontvieille in Monaco.") -> Path:
    table = pa.table(
        {
            "id": ["doc-0", "doc-1", "doc-2", "doc-3"],
            "text": [first_text, "No match.", "No match.", "Fontvieille in Monaco."],
            "url": ["", "", "", "https://example.test/fontvieille"],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def make_config(tmp_path: Path) -> tuple[ScanRunConfig, Path]:
    repository_root = tmp_path / "repo"
    data_root = tmp_path / "external"
    paths = ProjectPaths.from_environment(
        repository_root,
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(data_root)},
    )
    paths.ensure_data_layout()
    pbf = paths.raw_dir / "mini.osm.pbf"
    pbf.write_bytes(b"synthetic pbf")
    shard = write_shard(paths.raw_dir / "shard.parquet")
    return (
        ScanRunConfig(
            paths=paths,
            pbf_path=pbf,
            shard_path=shard,
            run_id="case",
        ),
        shard,
    )


def test_run_resumes_completed_row_groups(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)

    first = execute_run(config, profiles=profiles)
    first_bytes = first.result_path.read_bytes()
    second = execute_run(config, profiles=profiles)

    assert first.partitions_completed == 2
    assert second.partitions_skipped == first.partitions_completed
    assert second.result_path.read_bytes() == first_bytes


def test_run_rejects_changed_input_fingerprint(tmp_path: Path) -> None:
    config, shard = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    execute_run(config, profiles=profiles)
    write_shard(shard, first_text="Changed input in Monaco.")

    with pytest.raises(ValueError, match="fingerprint"):
        execute_run(config, profiles=profiles)


def test_run_rejects_repository_local_data_root(tmp_path: Path) -> None:
    repository_root = tmp_path / "repo"
    paths = ProjectPaths.from_environment(
        repository_root,
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(repository_root)},
    )
    config = ScanRunConfig(
        paths=paths,
        pbf_path=repository_root / "raw" / "mini.osm.pbf",
        shard_path=repository_root / "raw" / "shard.parquet",
        run_id="case",
    )

    with pytest.raises(ValueError, match="external"):
        execute_run(config, profiles=())
