import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

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
    _Partition,
    _RowGroup,
    _RunLayout,
    _select_profiles,
    _sha256_file,
    _sha256_payload,
    execute_run,
)
from fineweb_polygons.scanning import ScanStats
from fineweb_polygons.versions import get_retrieval_definition


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
        "deduplicate_documents": False,
        "matcher_version": "v2-exact-name-url-or-text-with-text-country-context",
        "normalization_version": "v1-nfkc-casefold-separators",
        "polygon_profile_version": "v2-in-boundary-meaningful-names",
        "require_url_name": False,
        "retrieval_version": "v2",
        "row_groups_per_partition": 32,
        "retrieval_definition": get_retrieval_definition("v2").to_record(),
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


def test_v3_run_deduplicates_final_matches_and_records_the_contract(
    tmp_path: Path,
) -> None:
    config, shard = make_config(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": ["doc-1", "doc-1", "doc-1"],
                "text": ["Fontvieille is in Monaco."] * 3,
                "url": ["https://example.test/fontvieille"] * 3,
            }
        ),
        shard,
        row_group_size=1,
    )
    config = ScanRunConfig(
        paths=config.paths,
        pbf_path=config.pbf_path,
        shard_path=config.shard_path,
        run_id="v3-case",
        retrieval_version="v3",
    )
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)

    first = execute_run(config, profiles=profiles)
    first_bytes = first.result_path.read_bytes()
    second = execute_run(config, profiles=profiles)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    assert first.matches_written == 1
    assert second.matches_written == 1
    assert second.partitions_skipped == 1
    assert second.result_path.read_bytes() == first_bytes
    assert len(first.result_path.read_text(encoding="utf-8").splitlines()) == 1
    assert manifest["configuration"]["require_url_name"] is True
    assert manifest["configuration"]["deduplicate_documents"] is True
    assert manifest["matches_written"] == 1
    assert manifest["partitions"][0]["stats"]["matches_written"] == 3


def test_v4_run_deduplicates_text_only_matches_and_records_the_contract(
    tmp_path: Path,
) -> None:
    config, shard = make_config(tmp_path)
    pq.write_table(
        pa.table(
            {
                "id": ["doc-1", "doc-1"],
                "text": ["Fontvieille is in Monaco."] * 2,
                "url": ["https://example.test/unrelated"] * 2,
            }
        ),
        shard,
        row_group_size=1,
    )
    config = ScanRunConfig(
        paths=config.paths,
        pbf_path=config.pbf_path,
        shard_path=config.shard_path,
        run_id="v4-case",
        retrieval_version="v4",
    )
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)

    first = execute_run(config, profiles=profiles)
    second = execute_run(config, profiles=profiles)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in first.result_path.read_text(encoding="utf-8").splitlines()
    ]

    assert first.matches_written == 1
    assert second.matches_written == 1
    assert second.partitions_skipped == 1
    assert records[0]["matched_fields"] == ["text"]
    assert records[0]["context_fields"] == ["text"]
    assert manifest["configuration"]["require_url_name"] is False
    assert manifest["configuration"]["require_text_name"] is True
    assert manifest["configuration"]["deduplicate_documents"] is True
    assert manifest["configuration"]["retrieval_version"] == "v4"


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
        captured["require_url_name"] = kwargs["require_url_name"]
        return real_matcher(profiles, **kwargs)

    monkeypatch.setattr(runs_module, "_select_profiles", fake_select)
    monkeypatch.setattr(runs_module, "EvidenceMatcher", fake_matcher)

    execute_run(config)

    assert captured == {
        "pbf_path": config.pbf_path,
        "profiles": None,
        "retrieval_version": "v2",
        "require_text_context": True,
        "require_url_name": False,
    }


def test_execute_run_passes_v4_text_name_requirement(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = make_config(tmp_path)
    config = ScanRunConfig(
        paths=config.paths,
        pbf_path=config.pbf_path,
        shard_path=config.shard_path,
        run_id="v4-inputs",
        retrieval_version="v4",
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
        captured["require_text_name"] = kwargs["require_text_name"]
        captured["require_url_name"] = kwargs["require_url_name"]
        return real_matcher(profiles, **kwargs)

    monkeypatch.setattr(runs_module, "_select_profiles", fake_select)
    monkeypatch.setattr(runs_module, "EvidenceMatcher", fake_matcher)

    execute_run(config)

    assert captured == {
        "pbf_path": config.pbf_path,
        "profiles": None,
        "retrieval_version": "v4",
        "require_text_context": True,
        "require_text_name": True,
        "require_url_name": False,
    }


def test_select_profiles_dispatches_v3_reader(tmp_path: Path, monkeypatch) -> None:
    pbf = tmp_path / "monaco.osm.pbf"
    expected = PolygonReadResult(
        profiles=(PolygonProfile.create("way/1", "Fontvieille"),),
        named_count=1,
        unnamed_count=2,
        filtered_count=3,
    )
    monkeypatch.setattr(runs_module, "read_v3_polygon_profiles", lambda _: expected)

    result = _select_profiles(pbf, None, retrieval_version="v3")

    assert result == (
        expected.profiles,
        expected.named_count,
        expected.unnamed_count,
        expected.filtered_count,
    )


def test_select_profiles_dispatches_v4_reader(tmp_path: Path, monkeypatch) -> None:
    pbf = tmp_path / "monaco.osm.pbf"
    expected = PolygonReadResult(
        profiles=(PolygonProfile.create("way/1", "Fontvieille"),),
        named_count=1,
        unnamed_count=2,
        filtered_count=3,
    )
    monkeypatch.setattr(runs_module, "read_v3_polygon_profiles", lambda _: expected)

    result = _select_profiles(pbf, None, retrieval_version="v4")

    assert result == (
        expected.profiles,
        expected.named_count,
        expected.unnamed_count,
        expected.filtered_count,
    )


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

    def v3_reader(path: Path) -> PolygonReadResult:
        calls.append(("v3", path))
        return expected

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "read_named_polygon_profiles", v1_reader)
    monkeypatch.setattr(runs_module, "read_v2_polygon_profiles", v2_reader)
    monkeypatch.setattr(runs_module, "read_v3_polygon_profiles", v3_reader)
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
    assert _select_profiles(pbf, None, retrieval_version="v3") == (
        expected.profiles,
        4,
        5,
        6,
    )
    assert calls == [("v1", pbf), ("v2", pbf), ("v3", pbf)]


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
    assert manifest["configuration_sha256"] == _sha256_payload(configuration)
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

    with pytest.raises(
        ValueError, match=r"\ARun manifest fingerprint conflict in partitions\Z"
    ):
        execute_run(config, profiles=profiles)


def test_run_rejects_non_list_partition_manifest(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    execute_run(config, profiles=profiles)
    manifest_path = config.paths.runs_dir / config.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=r"\ARun manifest partitions must be a list\Z"):
        execute_run(config, profiles=profiles)


def test_run_rejects_changed_retrieval_definition(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    execute_run(config, profiles=profiles)
    manifest_path = config.paths.runs_dir / config.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["retrieval_definition"]["title"] = "changed"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="configuration"):
        execute_run(config, profiles=profiles)


def test_validated_inputs_reports_the_missing_path(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    missing = config.paths.raw_dir / "missing.osm.pbf"
    missing_config = ScanRunConfig(
        paths=config.paths,
        pbf_path=missing,
        shard_path=config.shard_path,
        run_id=config.run_id,
    )

    with pytest.raises(FileNotFoundError) as error:
        runs_module._validated_inputs(missing_config)

    assert error.value.args == (missing,)


def test_inspect_row_groups_preserves_cumulative_offsets_and_errors(
    tmp_path: Path,
) -> None:
    config, shard = make_config(tmp_path)

    assert runs_module._inspect_row_groups(shard) == (
        _RowGroup(0, 0, 2),
        _RowGroup(1, 2, 2),
    )
    three_group_shard = config.paths.raw_dir / "three-groups.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["a", "b", "c", "d", "e"],
                "url": ["", "", "", "", ""],
            }
        ),
        three_group_shard,
        row_group_size=2,
    )
    assert runs_module._inspect_row_groups(three_group_shard) == (
        _RowGroup(0, 0, 2),
        _RowGroup(1, 2, 2),
        _RowGroup(2, 4, 1),
    )
    missing = config.paths.raw_dir / "missing-columns.parquet"
    pq.write_table(pa.table({"text": ["missing url"]}), missing)
    with pytest.raises(
        ValueError,
        match=(r"\AParquet shard must contain text and url columns; missing url\Z"),
    ):
        runs_module._inspect_row_groups(missing)
    missing_both = config.paths.raw_dir / "missing-both-columns.parquet"
    pq.write_table(pa.table({"id": ["missing required columns"]}), missing_both)
    with pytest.raises(
        ValueError,
        match=(
            r"\AParquet shard must contain text and url columns; missing text, url\Z"
        ),
    ):
        runs_module._inspect_row_groups(missing_both)


def test_process_partition_records_running_and_complete_states(
    tmp_path: Path, monkeypatch
) -> None:
    config, shard = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    manifest = {"partitions": [{"status": "pending"}]}
    partition_spec = _Partition(0, (_RowGroup(0, 0, 4),))
    writes: list[tuple[Path, dict[str, Any]]] = []
    captured: dict[str, object] = {}

    def fake_write(path: Path, value: dict[str, Any]) -> None:
        writes.append((path, deepcopy(value)))

    def fake_scan(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return ScanStats(rows_scanned=4, matches_written=2)

    monkeypatch.setattr(runs_module, "_atomic_json_write", fake_write)
    monkeypatch.setattr(runs_module, "scan_row_groups", fake_scan)

    skipped, stats = runs_module._process_partition(
        config=config,
        layout=layout,
        manifest=manifest,
        partition_spec=partition_spec,
        shard_path=shard,
        matcher=runs_module.EvidenceMatcher([]),
    )

    assert skipped is False
    assert stats == ScanStats(rows_scanned=4, matches_written=2)
    assert manifest["partitions"][0] == {
        "status": "complete",
        "path": str(layout.partitions_dir / "partition-00000.jsonl"),
        "stats": {"rows_scanned": 4, "matches_written": 2},
    }
    assert writes[0][1]["partitions"][0]["status"] == "running"
    assert writes[1][1]["partitions"][0]["status"] == "complete"
    assert captured["path"] == shard
    assert captured["row_group_indices"] == (0,)
    assert captured["batch_size"] == config.batch_size
    log_record = json.loads(layout.log_path.read_text(encoding="utf-8").strip())
    assert log_record == {
        "event": "partition_complete",
        "matches": 2,
        "partition": 0,
        "rows_scanned": 4,
        "timestamp": log_record["timestamp"],
    }


def test_process_partition_records_failure_state_and_error(
    tmp_path: Path, monkeypatch
) -> None:
    config, shard = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    manifest = {"partitions": [{"status": "pending"}]}
    partition_spec = _Partition(0, (_RowGroup(0, 0, 4),))
    writes: list[tuple[Path, dict[str, Any]]] = []

    def fake_write(path: Path, value: dict[str, Any]) -> None:
        writes.append((path, deepcopy(value)))

    monkeypatch.setattr(runs_module, "_atomic_json_write", fake_write)

    def failing_scan(path, **kwargs):
        del path, kwargs
        raise RuntimeError("scan failed")

    monkeypatch.setattr(runs_module, "scan_row_groups", failing_scan)

    with pytest.raises(RuntimeError, match=r"\Ascan failed\Z"):
        runs_module._process_partition(
            config=config,
            layout=layout,
            manifest=manifest,
            partition_spec=partition_spec,
            shard_path=shard,
            matcher=runs_module.EvidenceMatcher([]),
        )

    assert manifest["partitions"][0]["status"] == "failed"
    assert manifest["partitions"][0]["error"] == "scan failed"
    assert len(writes) == 2
    assert [path for path, _ in writes] == [
        layout.manifest_path,
        layout.manifest_path,
    ]
    assert writes[0][1]["partitions"][0]["status"] == "running"
    assert writes[1][1]["partitions"][0]["status"] == "failed"
    log_record = json.loads(layout.log_path.read_text(encoding="utf-8").strip())
    assert log_record["event"] == "partition_failed"
    assert log_record["partition"] == 0
    assert log_record["error"] == "scan failed"


def test_partition_completion_requires_status_and_file(tmp_path: Path) -> None:
    path = tmp_path / "partition.jsonl"

    assert runs_module._partition_is_complete({"status": "complete"}, path) is False
    path.write_text("{}\n", encoding="utf-8")
    assert runs_module._partition_is_complete({"status": "pending"}, path) is False
    assert runs_module._partition_is_complete({"status": "complete"}, path) is True


def test_complete_run_saves_counters_digest_timestamp_and_log(
    tmp_path: Path, monkeypatch
) -> None:
    config, _ = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    layout.result_path.parent.mkdir(parents=True, exist_ok=True)
    layout.result_path.write_text("évidence\n", encoding="utf-8")
    manifest: dict[str, Any] = {}
    counters = runs_module._RunCounters(2, 1, 7, 3)
    monkeypatch.setattr(runs_module, "_timestamp", lambda: "fixed-time")

    runs_module._complete_run(layout, manifest, counters, elapsed=1.5)

    expected_digest = hashlib.sha256(layout.result_path.read_bytes()).hexdigest()
    assert manifest == {
        "status": "complete",
        "rows_scanned": 7,
        "matches_written": 3,
        "elapsed_seconds": 1.5,
        "result_sha256": expected_digest,
        "completed_at": "fixed-time",
    }
    log_record = json.loads(layout.log_path.read_text(encoding="utf-8").strip())
    assert log_record == {
        "event": "run_complete",
        "elapsed_seconds": 1.5,
        "matches": 3,
        "rows_scanned": 7,
        "timestamp": "fixed-time",
    }


def _minimal_manifest() -> dict[str, Any]:
    configuration = {"retrieval_version": "v1"}
    return {
        "schema_version": 2,
        "run_id": "case",
        "sources": {"shard": {"sha256": "shard"}},
        "configuration": configuration,
        "configuration_sha256": _sha256_payload(configuration),
        "polygon_profile_sha256": "profiles",
        "partitions": [
            {
                "index": 0,
                "row_group_indices": [0],
                "row_start": 0,
                "row_count": 1,
                "path": "partition.jsonl",
            }
        ],
    }


def test_new_manifest_creation_writes_the_full_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "nested" / "manifest.json"
    expected = _minimal_manifest()
    calls: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        runs_module,
        "_atomic_json_write",
        lambda path, value: calls.append((path, value)),
    )

    actual = runs_module._load_or_create_manifest(manifest_path, expected)

    assert actual == expected
    assert calls == [(manifest_path, expected)]


@pytest.mark.parametrize(
    "key",
    [
        "schema_version",
        "run_id",
        "sources",
        "configuration",
        "configuration_sha256",
        "polygon_profile_sha256",
    ],
)
def test_manifest_rejects_each_changed_fingerprint(key: str, tmp_path: Path) -> None:
    expected = _minimal_manifest()
    manifest = dict(expected)
    if key in {"sources", "configuration"}:
        manifest[key] = {"changed": True}
    elif key == "schema_version":
        manifest[key] = 999
    else:
        manifest[key] = "changed"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match=rf"\ARun manifest fingerprint conflict in {key}\Z"
    ):
        runs_module._load_or_create_manifest(manifest_path, expected)


def test_manifest_reads_utf8_with_the_explicit_encoding(
    tmp_path: Path, monkeypatch
) -> None:
    expected = _minimal_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(expected), encoding="utf-8")
    encodings: list[object] = []
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self == manifest_path:
            encodings.append(kwargs.get("encoding"))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    runs_module._load_or_create_manifest(manifest_path, expected)

    assert encodings == ["utf-8"]


def test_manifest_rejects_changed_partition_fingerprint_exactly(
    tmp_path: Path,
) -> None:
    expected = _minimal_manifest()
    manifest = dict(expected)
    manifest["partitions"] = []
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"\ARun manifest fingerprint conflict in partitions\Z"
    ):
        runs_module._load_or_create_manifest(manifest_path, expected)


def test_merge_partitions_creates_nested_result_and_preserves_utf8(
    tmp_path: Path, monkeypatch
) -> None:
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()
    partition = _Partition(0, (_RowGroup(0, 0, 1),))
    (partitions_dir / "partition-00000.jsonl").write_text(
        "évidence\n", encoding="utf-8"
    )
    layout = _RunLayout(
        run_dir=tmp_path / "run",
        partitions_dir=partitions_dir,
        manifest_path=tmp_path / "run" / "manifest.json",
        result_path=tmp_path / "deep" / "nested" / "result.jsonl",
        log_path=tmp_path / "logs" / "run.jsonl",
    )
    open_calls: list[tuple[object, object]] = []
    real_open = Path.open
    default_mode = object()

    def open_file(self, mode=default_mode, *args, **kwargs):
        if self in {
            layout.result_path.with_name(f".{layout.result_path.name}.tmp"),
            partitions_dir / "partition-00000.jsonl",
        }:
            recorded_mode = "<default>" if mode is default_mode else mode
            open_calls.append((recorded_mode, kwargs.get("encoding")))
        actual_mode = "r" if mode is default_mode else cast(str, mode)
        return real_open(self, actual_mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    runs_module._merge_partitions(layout, (partition,))

    assert layout.result_path.read_text(encoding="utf-8") == "évidence\n"
    assert open_calls == [("w", "utf-8"), ("r", "utf-8")]


def test_atomic_json_write_is_pretty_sorted_and_unicode_safe(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "deep" / "nested" / "manifest.json"
    encodings: list[object] = []
    ensure_ascii_values: list[object] = []
    real_write_text = Path.write_text
    real_dumps = runs_module.json.dumps

    def write_text(self, data, *args, **kwargs):
        if self == path.with_name(f".{path.name}.tmp"):
            encodings.append(kwargs.get("encoding"))
        return real_write_text(self, data, *args, **kwargs)

    def dumps(value, *args, **kwargs):
        ensure_ascii_values.append(kwargs.get("ensure_ascii"))
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(runs_module.json, "dumps", dumps)
    runs_module._atomic_json_write(path, {"z": "é", "a": [1]})

    assert path.read_text(encoding="utf-8") == (
        '{\n  "a": [\n    1\n  ],\n  "z": "é"\n}\n'
    )
    assert encodings == ["utf-8"]
    assert ensure_ascii_values == [False]


def test_atomic_json_write_suppresses_cleanup_when_write_fails(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "manifest.json"

    def failing_write(self, data, *args, **kwargs):
        del self, data, args, kwargs
        raise RuntimeError("write failed")

    monkeypatch.setattr(Path, "write_text", failing_write)
    with pytest.raises(RuntimeError, match=r"\Awrite failed\Z"):
        runs_module._atomic_json_write(path, {"value": 1})


def test_merge_partitions_suppresses_cleanup_when_open_fails(
    tmp_path: Path, monkeypatch
) -> None:
    partitions_dir = tmp_path / "partitions"
    partitions_dir.mkdir()
    partition = _Partition(0, (_RowGroup(0, 0, 1),))
    layout = _RunLayout(
        run_dir=tmp_path / "run",
        partitions_dir=partitions_dir,
        manifest_path=tmp_path / "run" / "manifest.json",
        result_path=tmp_path / "result.jsonl",
        log_path=tmp_path / "logs" / "run.jsonl",
    )
    temporary_path = layout.result_path.with_name(f".{layout.result_path.name}.tmp")
    real_open = Path.open

    def fail_open(self, *args, **kwargs):
        if self == temporary_path:
            raise RuntimeError("open failed")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_open)
    with pytest.raises(RuntimeError, match=r"\Aopen failed\Z"):
        runs_module._merge_partitions(layout, (partition,))


def test_log_is_sorted_unicode_safe_and_explicitly_utf8(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "run.jsonl"
    encodings: list[object] = []
    ensure_ascii_values: list[object] = []
    real_open = Path.open
    real_dumps = runs_module.json.dumps

    def open_file(self, mode="r", *args, **kwargs):
        if self == path and mode == "a":
            encodings.append(kwargs.get("encoding"))
        return real_open(self, mode, *args, **kwargs)

    def dumps(value, *args, **kwargs):
        ensure_ascii_values.append(kwargs.get("ensure_ascii"))
        return real_dumps(value, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(runs_module.json, "dumps", dumps)
    monkeypatch.setattr(runs_module, "_timestamp", lambda: "fixed-time")
    runs_module._log(path, "event", message="é")

    assert path.read_text(encoding="utf-8") == (
        '{"event": "event", "message": "é", "timestamp": "fixed-time"}\n'
    )
    assert encodings == ["utf-8"]
    assert ensure_ascii_values == [False]


def test_timestamp_is_timezone_aware_and_payload_hash_is_canonical(
    monkeypatch,
) -> None:
    assert runs_module._timestamp().endswith("+00:00")
    value = {"z": "é", "a": 1}
    ensure_ascii_values: list[object] = []
    encodings: list[str] = []
    real_dumps = runs_module.json.dumps

    class RecordingJson(str):
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            encodings.append(encoding)
            return super().encode(encoding, errors)

    def dumps(value, *args, **kwargs):
        ensure_ascii_values.append(kwargs.get("ensure_ascii"))
        return RecordingJson(real_dumps(value, *args, **kwargs))

    monkeypatch.setattr(runs_module.json, "dumps", dumps)
    expected = hashlib.sha256(
        real_dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    assert runs_module._sha256_payload(value) == expected
    assert runs_module._sha256_payload({"a": 1, "z": "é"}) == expected
    assert runs_module._sha256_payload(value) != runs_module._sha256_payload({"z": "é"})
    assert ensure_ascii_values == [False, False, False, False]
    assert encodings == ["utf-8", "utf-8", "utf-8", "utf-8"]


def test_process_partitions_accumulates_all_partition_counters(
    tmp_path: Path, monkeypatch
) -> None:
    config, shard = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    partitions = tuple(
        _Partition(index, (_RowGroup(index, index, 1),)) for index in range(4)
    )
    outcomes = iter(
        (
            (True, ScanStats(rows_scanned=2, matches_written=3)),
            (False, ScanStats(rows_scanned=5, matches_written=7)),
            (True, ScanStats(rows_scanned=11, matches_written=13)),
            (False, ScanStats(rows_scanned=17, matches_written=19)),
        )
    )
    called: list[int] = []

    def fake_process_partition(**kwargs):
        called.append(kwargs["partition_spec"].index)
        return next(outcomes)

    monkeypatch.setattr(runs_module, "_process_partition", fake_process_partition)

    counters = runs_module._process_partitions(
        config=config,
        layout=layout,
        manifest={},
        partitions=partitions,
        shard_path=shard,
        matcher=runs_module.EvidenceMatcher([]),
    )

    assert counters.partitions_completed == 2
    assert counters.partitions_skipped == 2
    assert counters.rows_scanned == 35
    assert counters.matches_written == 42
    assert called == [0, 1, 2, 3]


def test_process_partition_skips_complete_partition_and_logs_identity(
    tmp_path: Path,
) -> None:
    config, _ = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    layout.partitions_dir.mkdir(parents=True, exist_ok=True)
    partition_path = layout.partitions_dir / "partition-00000.jsonl"
    partition_path.write_text("{}\n", encoding="utf-8")
    manifest = {
        "partitions": [
            {
                "status": "complete",
                "stats": {"rows_scanned": 4, "matches_written": 2},
            }
        ]
    }
    partition_spec = _Partition(0, (_RowGroup(0, 0, 4),))

    skipped, stats = runs_module._process_partition(
        config=config,
        layout=layout,
        manifest=manifest,
        partition_spec=partition_spec,
        shard_path=tmp_path / "unused.parquet",
        matcher=runs_module.EvidenceMatcher([]),
    )

    assert skipped is True
    assert stats == ScanStats(rows_scanned=4, matches_written=2)
    assert manifest["partitions"][0]["path"] == str(partition_path)
    log_record = json.loads(layout.log_path.read_text(encoding="utf-8").strip())
    assert log_record["event"] == "partition_skipped"
    assert log_record["partition"] == 0


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
