import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import fineweb_polygons.runs as runs_module
from fineweb_polygons.foundation import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    ProjectPaths,
)
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.polygons import PolygonReadResult
from fineweb_polygons.runs import (
    ScanRunConfig,
    _make_partitions,
    _new_manifest,
    _RowGroup,
    _RunLayout,
    _select_profiles,
    _sha256_file,
    _sha256_payload,
    execute_run,
)


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

    assert first.partitions_completed == 1
    assert first.manifest_path == config.paths.runs_dir / "case" / "manifest.json"
    assert first.rows_scanned == 4
    assert first.matches_written == 2
    assert second.partitions_skipped == first.partitions_completed
    assert second.rows_scanned == 4
    assert second.matches_written == 2
    assert second.result_path.read_bytes() == first_bytes
    log_records = [
        json.loads(line)
        for line in (config.paths.logs_dir / "case.jsonl").read_text().splitlines()
    ]
    assert log_records[0]["event"] == "run_started"
    assert log_records[0]["run_id"] == "case"


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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "bad/id", "run_id"),
        ("batch_size", 0, "batch_size"),
        ("row_groups_per_partition", 0, "row_groups_per_partition"),
    ],
)
def test_run_config_rejects_invalid_runtime_settings(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    config, _ = make_config(tmp_path)
    values: dict[str, Any] = {
        "paths": config.paths,
        "pbf_path": config.pbf_path,
        "shard_path": config.shard_path,
        "run_id": config.run_id,
        "batch_size": config.batch_size,
        "row_groups_per_partition": config.row_groups_per_partition,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        ScanRunConfig(**values)


def test_run_accepts_v2_retrieval_version_and_records_it(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    config = ScanRunConfig(
        paths=config.paths,
        pbf_path=config.pbf_path,
        shard_path=config.shard_path,
        run_id="v2-case",
        retrieval_version="v2",
    )
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)

    execute_run(config, profiles=profiles)

    manifest = json.loads(
        (config.paths.runs_dir / config.run_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["configuration"]["retrieval_version"] == "v2"
    assert manifest["configuration"]["matcher_version"] == (
        "v2-exact-name-url-or-text-with-text-country-context"
    )
    assert manifest["configuration"] == {
        "batch_size": 8192,
        "matcher_version": "v2-exact-name-url-or-text-with-text-country-context",
        "normalization_version": "v1-nfkc-casefold-separators",
        "polygon_profile_version": "v2-in-boundary-meaningful-names",
        "retrieval_version": "v2",
        "row_groups_per_partition": 32,
    }
    assert manifest["polygon_counts"] == {
        "filtered": 0,
        "named": 1,
        "unnamed": 0,
    }
    assert manifest["sources"]["pbf"] == {
        "path": str(config.pbf_path),
        "sha256": _sha256_file(config.pbf_path),
    }
    assert manifest["sources"]["shard"] == {
        "path": str(config.shard_path),
        "sha256": _sha256_file(config.shard_path),
    }
    assert manifest["status"] == "complete"
    assert manifest["run_id"] == config.run_id
    assert manifest["partitions"][0]["status"] == "complete"


def test_execute_run_passes_raw_profile_inputs_and_v2_context_requirement(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = make_config(tmp_path)
    config = ScanRunConfig(
        paths=config.paths,
        pbf_path=config.pbf_path,
        shard_path=config.shard_path,
        run_id="v2-inputs",
        retrieval_version="v2",
    )
    captured: dict[str, object] = {}
    selected = (PolygonProfile.create("way/1", "Fontvieille"),)

    def fake_select(pbf_path, profiles, *, retrieval_version):
        captured["pbf_path"] = pbf_path
        captured["profiles"] = profiles
        captured["retrieval_version"] = retrieval_version
        return selected, 1, 0, 0

    real_matcher = runs_module.EvidenceMatcher

    def fake_matcher(profiles, **kwargs):
        captured["require_text_context"] = kwargs["require_text_context"]
        return real_matcher(profiles, **kwargs)

    monkeypatch.setattr(runs_module, "_select_profiles", fake_select)
    monkeypatch.setattr(runs_module, "EvidenceMatcher", fake_matcher)

    execute_run(config)

    assert captured == {
        "pbf_path": config.pbf_path,
        "profiles": None,
        "retrieval_version": "v2",
        "require_text_context": True,
    }


def test_execute_run_records_elapsed_seconds_and_summary_paths(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = make_config(tmp_path)
    values = iter((10.0, 12.5))
    monkeypatch.setattr(runs_module, "perf_counter", lambda: next(values))

    summary = execute_run(
        config, profiles=(PolygonProfile.create("way/1", "Fontvieille"),)
    )
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))

    assert manifest["elapsed_seconds"] == 2.5
    assert summary.manifest_path == config.paths.runs_dir / "case" / "manifest.json"
    assert summary.rows_scanned == 4
    assert summary.matches_written == 2


def test_select_profiles_uses_the_requested_raw_pbf_reader(
    tmp_path: Path, monkeypatch
) -> None:
    expected = PolygonReadResult(
        profiles=(PolygonProfile.create("way/1", "Fontvieille"),),
        named_count=4,
        unnamed_count=5,
        filtered_count=6,
    )
    calls: list[tuple[str, Path]] = []

    def v1_reader(path: Path) -> PolygonReadResult:
        calls.append(("v1", path))
        return expected

    def v2_reader(path: Path) -> PolygonReadResult:
        calls.append(("v2", path))
        return expected

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "read_named_polygon_profiles", v1_reader)
    monkeypatch.setattr(runs_module, "read_v2_polygon_profiles", v2_reader)
    pbf = tmp_path / "monaco.osm.pbf"

    assert _select_profiles(pbf, None, retrieval_version="v1") == (
        expected.profiles,
        4,
        5,
        6,
    )
    assert _select_profiles(pbf, None, retrieval_version="v2") == (
        expected.profiles,
        4,
        5,
        6,
    )
    assert calls == [("v1", pbf), ("v2", pbf)]


def test_select_profiles_with_explicit_profiles_has_zero_source_counts(
    tmp_path: Path,
) -> None:
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)

    assert _select_profiles(
        tmp_path / "unused.osm.pbf", profiles, retrieval_version="v2"
    ) == (profiles, 1, 0, 0)


def test_new_manifest_contains_pending_partitions_and_all_fingerprints(
    tmp_path: Path,
) -> None:
    config, _ = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    partitions = _make_partitions(
        (_RowGroup(0, 0, 2), _RowGroup(1, 2, 2)), groups_per_partition=1
    )
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    configuration = {"retrieval_version": "v2", "batch_size": 8}

    manifest = _new_manifest(
        run_id="new-case",
        layout=layout,
        partitions=partitions,
        source_fingerprints={"pbf": {"sha256": "pbf"}},
        profiles=profiles,
        configuration=configuration,
        named_count=4,
        unnamed_count=5,
        filtered_count=6,
    )

    assert manifest["schema_version"] == 2
    assert manifest["run_id"] == "new-case"
    assert manifest["status"] == "running"
    assert manifest["sources"] == {"pbf": {"sha256": "pbf"}}
    assert manifest["configuration"] == configuration
    assert manifest["configuration_sha256"]
    assert manifest["polygon_profile_sha256"] == _sha256_payload(
        [
            {
                "polygon_id": "way/1",
                "name": "Fontvieille",
                "normalized_name": "fontvieille",
            }
        ]
    )
    assert manifest["polygon_counts"] == {
        "named": 4,
        "unnamed": 5,
        "filtered": 6,
    }
    assert manifest["result_path"] == str(layout.result_path)
    assert [partition["status"] for partition in manifest["partitions"]] == [
        "pending",
        "pending",
    ]
    assert [partition["stats"] for partition in manifest["partitions"]] == [
        None,
        None,
    ]


def test_run_rejects_changed_partition_structure(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    execute_run(config, profiles=profiles)
    manifest_path = config.paths.runs_dir / config.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"][0]["row_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="partitions"):
        execute_run(config, profiles=profiles)


def test_run_rejects_non_list_partition_manifest(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    execute_run(config, profiles=profiles)
    manifest_path = config.paths.runs_dir / config.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a list"):
        execute_run(config, profiles=profiles)


def test_sha256_file_reads_in_bounded_megabyte_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "payload.bin"
    read_sizes: list[object] = []

    class FakeSource:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            read_sizes.append(size)
            return b"payload" if len(read_sizes) == 1 else b""

    def fake_open(self, *args, **kwargs):
        return FakeSource()

    monkeypatch.setattr(Path, "open", fake_open)

    assert _sha256_file(path) == hashlib.sha256(b"payload").hexdigest()
    assert read_sizes == [1024 * 1024, 1024 * 1024]
