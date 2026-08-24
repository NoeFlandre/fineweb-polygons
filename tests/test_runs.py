import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.foundation import (
    DATA_ROOT_ENVIRONMENT_VARIABLE,
    ProjectPaths,
)
from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.polygons import PolygonReadResult
from fineweb_polygons.runs import (
    ScanRunConfig,
    _atomic_json_write,
    _inspect_row_groups,
    _load_or_create_manifest,
    _log,
    _make_partitions,
    _merge_partitions,
    _new_manifest,
    _Partition,
    _partition_is_complete,
    _partition_structure,
    _process_partition,
    _process_partitions,
    _RowGroup,
    _RunLayout,
    _select_profiles,
    _sha256_file,
    _sha256_payload,
    _timestamp,
    _validated_inputs,
    execute_run,
)
from fineweb_polygons.scanning import ScanStats


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
    assert second.partitions_skipped == first.partitions_completed
    assert second.result_path.read_bytes() == first_bytes
    assert first.rows_scanned == 4
    assert first.matches_written == 2
    logs = [
        json.loads(line)
        for line in (config.paths.logs_dir / "case.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert logs[-2]["event"] == "partition_skipped"
    assert logs[-2]["partition"] == 0


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
    values = {
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


def test_run_rejects_changed_partition_structure(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)
    execute_run(config, profiles=profiles)
    manifest_path = config.paths.runs_dir / config.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["partitions"][0]["row_count"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        ValueError, match=r"^Run manifest fingerprint conflict in partitions$"
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

    with pytest.raises(ValueError, match="must be a list"):
        execute_run(config, profiles=profiles)


def test_run_persists_complete_manifest_and_structured_logs(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    profiles = (PolygonProfile.create("way/1", "Fontvieille"),)

    summary = execute_run(config, profiles=profiles)
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    logs = [
        json.loads(line)
        for line in (config.paths.logs_dir / "case.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert manifest["schema_version"] == 2
    assert manifest["run_id"] == "case"
    assert manifest["status"] == "complete"
    assert manifest["configuration"] == {
        "batch_size": 8192,
        "matcher_version": "v1-exact-name-context",
        "normalization_version": "v1-nfkc-casefold-separators",
        "row_groups_per_partition": 32,
    }
    assert manifest["configuration_sha256"] == (
        "a9c72b670761ee9520aa9a8159fc92c4792d966f9f8d0dc89eb72a5b0f71c4fd"
    )
    assert manifest["polygon_profile_sha256"] == (
        "58df75ad9aab73ed75c9cee5199f9fb4842c09a136e319de41030d935c85047a"
    )
    assert manifest["polygon_counts"] == {"named": 1, "unnamed": 0}
    assert manifest["rows_scanned"] == 4
    assert manifest["matches_written"] == 2
    assert manifest["result_path"] == str(summary.result_path)
    assert (
        manifest["result_sha256"]
        == hashlib.sha256(summary.result_path.read_bytes()).hexdigest()
    )
    assert isinstance(manifest["elapsed_seconds"], (int, float))
    assert manifest["elapsed_seconds"] >= 0
    assert manifest["elapsed_seconds"] < 60
    assert datetime.fromisoformat(manifest["completed_at"]).tzinfo == UTC
    for source in manifest["sources"].values():
        assert set(source) == {"path", "sha256"}
        assert len(source["sha256"]) == 64
    assert manifest["sources"]["pbf"]["path"] == str(config.pbf_path.resolve())
    assert manifest["sources"]["shard"]["path"] == str(config.shard_path.resolve())
    partition = manifest["partitions"][0]
    assert partition == {
        "index": 0,
        "row_group_indices": [0, 1],
        "row_start": 0,
        "row_count": 4,
        "status": "complete",
        "stats": {"rows_scanned": 4, "matches_written": 2},
        "path": str(
            config.paths.runs_dir / "case" / "partitions/partition-00000.jsonl"
        ),
    }
    assert [record["event"] for record in logs] == [
        "run_started",
        "partition_complete",
        "run_complete",
    ]
    for record in logs:
        assert datetime.fromisoformat(record.pop("timestamp")).tzinfo == UTC
    assert logs[0] == {"event": "run_started", "run_id": "case"}
    assert logs[1] == {
        "event": "partition_complete",
        "partition": 0,
        "rows_scanned": 4,
        "matches": 2,
    }
    assert logs[2]["event"] == "run_complete"
    assert logs[2]["rows_scanned"] == 4
    assert logs[2]["matches"] == 2
    assert isinstance(logs[2]["elapsed_seconds"], (int, float))
    assert logs[2]["elapsed_seconds"] == manifest["elapsed_seconds"]
    assert summary.partitions_completed == 1
    assert summary.partitions_skipped == 0
    assert summary.rows_scanned == 4
    assert summary.matches_written == 2


def test_new_manifest_contains_pending_partition_and_fingerprints(
    tmp_path: Path,
) -> None:
    layout = _RunLayout(
        run_dir=tmp_path / "run",
        partitions_dir=tmp_path / "run" / "partitions",
        manifest_path=tmp_path / "run" / "manifest.json",
        result_path=tmp_path / "result.jsonl",
        log_path=tmp_path / "run.jsonl",
    )
    partition = _Partition(0, (_RowGroup(0, 0, 4),))

    manifest = _new_manifest(
        run_id="case",
        layout=layout,
        partitions=(partition,),
        source_fingerprints={"pbf": {"sha256": "pbf"}},
        profiles=(),
        configuration={"batch_size": 1},
        named_count=0,
        unnamed_count=2,
    )

    assert manifest == {
        "schema_version": 2,
        "run_id": "case",
        "status": "running",
        "sources": {"pbf": {"sha256": "pbf"}},
        "configuration": {"batch_size": 1},
        "configuration_sha256": (
            "11ed8e9ba95c94ab9aeafedc04fb730c5391bf9894a761148330a5fff65532c6"
        ),
        "polygon_profile_sha256": (
            "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
        ),
        "polygon_counts": {"named": 0, "unnamed": 2},
        "partitions": [
            {
                "index": 0,
                "row_group_indices": [0],
                "row_start": 0,
                "row_count": 4,
                "status": "pending",
                "stats": None,
                "path": str(tmp_path / "run" / "partitions/partition-00000.jsonl"),
            }
        ],
        "result_path": str(tmp_path / "result.jsonl"),
    }


def test_run_helpers_cover_partitioning_hashes_and_timestamps(tmp_path: Path) -> None:
    row_groups = (_RowGroup(0, 0, 2), _RowGroup(1, 2, 2), _RowGroup(2, 4, 2))

    assert _make_partitions(row_groups, groups_per_partition=2) == (
        _Partition(0, row_groups[:2]),
        _Partition(1, row_groups[2:]),
    )
    assert _sha256_payload({"b": "é", "a": 1}) == (
        "72098b2a2208bcd8cda44c39c3aa422be996b505ae0f32eae54f5da06d5ecfd4"
    )
    file_path = tmp_path / "input.bin"
    file_path.write_bytes(b"abc")
    assert _sha256_file(file_path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert datetime.fromisoformat(_timestamp()).tzinfo == UTC


def test_sha256_file_reads_one_megabyte_chunks(monkeypatch, tmp_path: Path) -> None:
    sizes = []

    class FakeSource:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self, size):
            sizes.append(size)
            return b"abc" if len(sizes) == 1 else b""

    def fake_open(self, mode, *args, **kwargs):
        assert mode == "rb"
        return FakeSource()

    monkeypatch.setattr(Path, "open", fake_open)

    assert _sha256_file(tmp_path / "synthetic.bin") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert sizes == [1024 * 1024, 1024 * 1024]


def test_sha256_payload_uses_stable_unicode_json_and_utf8(monkeypatch) -> None:
    import fineweb_polygons.runs as runs_module

    original_dumps = runs_module.json.dumps
    seen = {}

    class EncodedText(str):
        def encode(self, encoding="utf-8", errors="strict"):
            seen["encoding"] = encoding
            return super().encode(encoding, errors)

    def recording_dumps(value, *args, **kwargs):
        seen.update(kwargs)
        return EncodedText(original_dumps(value, *args, **kwargs))

    monkeypatch.setattr(runs_module.json, "dumps", recording_dumps)

    assert _sha256_payload({"é": 1}) == hashlib.sha256('{"é": 1}'.encode()).hexdigest()
    assert seen["ensure_ascii"] is False
    assert seen["sort_keys"] is True
    assert seen["encoding"] == "utf-8"


def test_inspect_row_groups_reports_schema_and_global_starts(tmp_path: Path) -> None:
    shard = write_shard(tmp_path / "shard.parquet")

    assert _inspect_row_groups(shard) == (
        _RowGroup(0, 0, 2),
        _RowGroup(1, 2, 2),
    )

    missing = tmp_path / "missing-columns.parquet"
    pq.write_table(pa.table({"title": ["Monaco"]}), missing)
    with pytest.raises(
        ValueError,
        match=r"^Parquet shard must contain text and url columns; missing text, url$",
    ):
        _inspect_row_groups(missing)

    uneven = tmp_path / "uneven.parquet"
    pq.write_table(
        pa.table(
            {
                "text": ["a", "b", "c", "d", "e"],
                "url": ["", "", "", "", ""],
            }
        ),
        uneven,
        row_group_size=2,
    )
    assert _inspect_row_groups(uneven)[-1] == _RowGroup(2, 4, 1)


def test_validated_inputs_reports_the_missing_path(tmp_path: Path) -> None:
    config, _ = make_config(tmp_path)
    missing = config.paths.raw_dir / "missing.osm.pbf"
    invalid = ScanRunConfig(
        paths=config.paths,
        pbf_path=missing,
        shard_path=config.shard_path,
        run_id=config.run_id,
    )

    with pytest.raises(FileNotFoundError) as error:
        _validated_inputs(invalid)
    assert error.value.args == (missing.resolve(),)


def test_process_partitions_accumulates_completed_and_skipped_counts(
    monkeypatch, tmp_path: Path
) -> None:
    config, shard = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    partitions = (
        _Partition(0, (_RowGroup(0, 0, 2),)),
        _Partition(1, (_RowGroup(1, 2, 2),)),
        _Partition(2, (_RowGroup(2, 4, 2),)),
        _Partition(3, (_RowGroup(3, 6, 2),)),
    )

    def fake_process_partition(**kwargs):
        index = kwargs["partition_spec"].index
        return (
            index in {0, 2},
            ScanStats(rows_scanned=index + 2, matches_written=index + 3),
        )

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "_process_partition", fake_process_partition)

    counters = _process_partitions(
        config=config,
        layout=layout,
        manifest={},
        partitions=partitions,
        shard_path=shard,
        matcher=EvidenceMatcher([]),
    )

    assert counters.partitions_completed == 2
    assert counters.partitions_skipped == 2
    assert counters.rows_scanned == 14
    assert counters.matches_written == 18


def test_process_partition_forwards_batch_size_and_persists_running_state(
    monkeypatch, tmp_path: Path
) -> None:
    config, shard = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    partition_spec = _Partition(0, (_RowGroup(0, 0, 2),))
    manifest = {
        "partitions": [
            {
                "status": "pending",
                "stats": None,
            }
        ]
    }
    captured = {}

    def fake_scan_row_groups(*args, **kwargs):
        captured.update(kwargs)
        on_disk = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        assert on_disk["partitions"][0]["status"] == "running"
        kwargs["output_path"].parent.mkdir(parents=True)
        kwargs["output_path"].write_text("", encoding="utf-8")
        return ScanStats(rows_scanned=2, matches_written=0)

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "scan_row_groups", fake_scan_row_groups)

    skipped, stats = _process_partition(
        config=config,
        layout=layout,
        manifest=manifest,
        partition_spec=partition_spec,
        shard_path=shard,
        matcher=EvidenceMatcher([]),
    )

    assert not skipped
    assert stats == ScanStats(rows_scanned=2, matches_written=0)
    assert captured["batch_size"] == config.batch_size
    assert captured["row_group_indices"] == (0,)
    assert manifest["partitions"][0]["status"] == "complete"
    on_disk = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert on_disk["partitions"][0]["stats"] == {
        "rows_scanned": 2,
        "matches_written": 0,
    }


def test_process_partition_failure_persists_error_and_log(
    monkeypatch, tmp_path: Path
) -> None:
    config, shard = make_config(tmp_path)
    layout = _RunLayout.from_config(config)
    partition_spec = _Partition(0, (_RowGroup(0, 0, 2),))
    manifest = {"partitions": [{"status": "pending", "stats": None}]}

    def fail_scan(*args, **kwargs):
        raise RuntimeError("synthetic scan failure")

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "scan_row_groups", fail_scan)

    with pytest.raises(RuntimeError, match="synthetic scan failure"):
        _process_partition(
            config=config,
            layout=layout,
            manifest=manifest,
            partition_spec=partition_spec,
            shard_path=shard,
            matcher=EvidenceMatcher([]),
        )

    assert manifest["partitions"][0] == {
        "status": "failed",
        "stats": None,
        "path": str(layout.partitions_dir / "partition-00000.jsonl"),
        "error": "synthetic scan failure",
    }
    log = json.loads(layout.log_path.read_text(encoding="utf-8").strip())
    assert log["event"] == "partition_failed"
    assert log["partition"] == 0
    assert log["error"] == "synthetic scan failure"
    on_disk = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    assert on_disk["partitions"][0]["status"] == "failed"
    assert on_disk["partitions"][0]["error"] == "synthetic scan failure"


def test_manifest_creation_and_each_fingerprint_conflict_are_checked(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "deep" / "run" / "manifest.json"
    expected = {
        "schema_version": 2,
        "run_id": "case",
        "sources": {"pbf": {"path": "pbf", "sha256": "hash"}},
        "configuration_sha256": "configuration",
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
    original_read_text = Path.read_text
    read_calls = []

    def recording_read_text(self, *args, **kwargs):
        read_calls.append((self, args, kwargs))
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert _load_or_create_manifest(manifest_path, expected) == expected
    assert _load_or_create_manifest(manifest_path, expected) == expected
    assert read_calls[0][2]["encoding"] == "utf-8"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == expected

    for key, changed in (
        ("schema_version", 3),
        ("run_id", "other"),
        ("sources", {"pbf": {"path": "other", "sha256": "other"}}),
        ("configuration_sha256", "other"),
        ("polygon_profile_sha256", "other"),
    ):
        changed_manifest = dict(expected)
        changed_manifest[key] = changed
        manifest_path.write_text(
            json.dumps(changed_manifest),
            encoding="utf-8",
        )
        with pytest.raises(
            ValueError,
            match=rf"^Run manifest fingerprint conflict in {key}$",
        ):
            _load_or_create_manifest(manifest_path, expected)


def test_atomic_json_write_is_sorted_unicode_and_replaces_atomically(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "nested" / "deeper" / "manifest.json"
    original_write_text = Path.write_text
    calls = []
    import fineweb_polygons.runs as runs_module

    original_dumps = runs_module.json.dumps
    dump_kwargs = {}

    def recording_write_text(self, data, *args, **kwargs):
        calls.append((self, args, kwargs))
        return original_write_text(self, data, *args, **kwargs)

    def recording_dumps(value, *args, **kwargs):
        dump_kwargs.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording_write_text)
    monkeypatch.setattr(runs_module.json, "dumps", recording_dumps)

    _atomic_json_write(path, {"z": "é", "a": {"b": 1}})

    assert path.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "b": 1\n  },\n  "z": "é"\n}\n'
    )
    assert not path.with_name(".manifest.json.tmp").exists()
    assert calls[0][2]["encoding"] == "utf-8"
    assert dump_kwargs["ensure_ascii"] is False
    assert dump_kwargs["sort_keys"] is True


def test_atomic_json_write_cleanup_preserves_write_failure(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "manifest.json"
    temporary_path = path.with_name(".manifest.json.tmp")
    original_write_text = Path.write_text

    def failing_write_text(self, data, *args, **kwargs):
        if self == temporary_path:
            raise RuntimeError("synthetic write failure")
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing_write_text)

    with pytest.raises(RuntimeError, match="synthetic write failure"):
        _atomic_json_write(path, {"value": 1})
    assert not temporary_path.exists()


def test_log_is_structured_unicode_and_sorted(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    import fineweb_polygons.runs as runs_module

    original_dumps = runs_module.json.dumps
    seen = {}
    original_open = Path.open
    open_calls = []

    def recording_dumps(value, *args, **kwargs):
        seen.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    def recording_open(self, *args, **kwargs):
        open_calls.append((self, args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(runs_module.json, "dumps", recording_dumps)
    monkeypatch.setattr(Path, "open", recording_open)

    _log(path, "évent", z="é", a=1)

    line = path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert record["event"] == "évent"
    assert record["z"] == "é"
    assert "é" in line
    assert line.index('"a"') < line.index('"event"') < line.index('"timestamp"')
    assert line.index('"timestamp"') < line.index('"z"')
    assert seen["ensure_ascii"] is False
    assert seen["sort_keys"] is True
    log_open = next(call for call in open_calls if call[1] == ("a",))
    assert log_open[2]["encoding"] == "utf-8"


def test_merge_partitions_is_ordered_utf8_and_atomic(
    monkeypatch, tmp_path: Path
) -> None:
    partitions_dir = tmp_path / "run" / "partitions"
    partitions_dir.mkdir(parents=True)
    layout = _RunLayout(
        run_dir=tmp_path / "run",
        partitions_dir=partitions_dir,
        manifest_path=tmp_path / "run" / "manifest.json",
        result_path=tmp_path / "deep" / "artifacts" / "result.jsonl",
        log_path=tmp_path / "run.jsonl",
    )
    (partitions_dir / "partition-00000.jsonl").write_text("é-first\n", encoding="utf-8")
    (partitions_dir / "partition-00001.jsonl").write_text("second\n", encoding="utf-8")
    partitions = (
        _Partition(0, (_RowGroup(0, 0, 1),)),
        _Partition(1, (_RowGroup(1, 1, 1),)),
    )
    original_open = Path.open
    calls = []

    def recording_open(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    _merge_partitions(layout, partitions)

    assert layout.result_path.read_text(encoding="utf-8") == "é-first\nsecond\n"
    assert all(call[2]["encoding"] == "utf-8" for call in calls)
    partition_calls = [call for call in calls if call[0].parent == partitions_dir]
    assert all(call[1] == ("r",) for call in partition_calls)
    assert not layout.result_path.with_name(".result.jsonl.tmp").exists()


def test_merge_cleanup_preserves_an_open_failure(monkeypatch, tmp_path: Path) -> None:
    layout = _RunLayout(
        run_dir=tmp_path / "run",
        partitions_dir=tmp_path / "run" / "partitions",
        manifest_path=tmp_path / "run" / "manifest.json",
        result_path=tmp_path / "result.jsonl",
        log_path=tmp_path / "run.jsonl",
    )
    temporary_path = layout.result_path.with_name(".result.jsonl.tmp")
    original_open = Path.open

    def failing_open(self, *args, **kwargs):
        if self == temporary_path:
            raise RuntimeError("synthetic merge open failure")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(RuntimeError, match="synthetic merge open failure"):
        _merge_partitions(layout, ())
    assert not temporary_path.exists()


def test_partition_helpers_and_structure_validation(tmp_path: Path) -> None:
    path = tmp_path / "partition.jsonl"
    path.write_text("", encoding="utf-8")
    complete = {"status": "complete"}
    pending = {"status": "pending"}

    assert _partition_is_complete(complete, path)
    assert not _partition_is_complete(pending, path)
    assert not _partition_is_complete(complete, path.with_name("missing.jsonl"))
    assert _partition_structure(
        [
            {
                "index": 0,
                "row_group_indices": [0],
                "row_start": 0,
                "row_count": 1,
                "path": "x",
                "status": "complete",
            }
        ]
    ) == [
        {
            "index": 0,
            "row_group_indices": [0],
            "row_start": 0,
            "row_count": 1,
            "path": "x",
        }
    ]
    with pytest.raises(ValueError, match=r"^Run manifest partitions must be a list$"):
        _partition_structure({})


def test_select_profiles_handles_osm_and_explicit_profiles(
    monkeypatch, tmp_path: Path
) -> None:
    profile = PolygonProfile.create("way/1", "Fontvieille")
    result = PolygonReadResult((profile,), named_count=1, unnamed_count=3)
    captured = {}

    def fake_reader(path):
        captured["path"] = path
        return result

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "read_named_polygon_profiles", fake_reader)
    pbf = tmp_path / "monaco.osm.pbf"

    assert _select_profiles(pbf, None) == ((profile,), 1, 3)
    assert captured["path"] == pbf
    assert _select_profiles(pbf, [profile]) == ((profile,), 1, 0)


def test_execute_run_uses_the_pbf_when_profiles_are_omitted(
    monkeypatch, tmp_path: Path
) -> None:
    config, _ = make_config(tmp_path)
    profile = PolygonProfile.create("way/1", "Fontvieille")
    captured = {}

    def fake_reader(path):
        captured["path"] = path
        return PolygonReadResult((profile,), named_count=1, unnamed_count=0)

    import fineweb_polygons.runs as runs_module

    monkeypatch.setattr(runs_module, "read_named_polygon_profiles", fake_reader)

    execute_run(config)

    assert captured["path"] == config.pbf_path.resolve()
