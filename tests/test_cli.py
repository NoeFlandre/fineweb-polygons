import json
from pathlib import Path

from fineweb_polygons.cli import main
from fineweb_polygons.runs import RunSummary


def test_cli_reports_foundation_only(capsys) -> None:
    assert main([]) == 0

    assert capsys.readouterr().out == (
        "fineweb-polygons foundation only; no pipeline executed\n"
    )


def test_cli_scan_passes_external_paths_to_runner(
    tmp_path: Path,
    capsys,
) -> None:
    data_root = tmp_path / "external"
    pbf = data_root / "raw" / "monaco.osm.pbf"
    shard = data_root / "raw" / "shard.parquet"
    captured = {}

    def fake_runner(config):
        captured["config"] = config
        return RunSummary(
            result_path=data_root / "artifacts" / "matches.jsonl",
            manifest_path=data_root / "runs" / "case" / "manifest.json",
            partitions_completed=1,
            partitions_skipped=0,
            rows_scanned=4,
            matches_written=2,
        )

    assert (
        main(
            [
                "scan",
                "--data-root",
                str(data_root),
                "--pbf",
                str(pbf),
                "--shard",
                str(shard),
                "--run-id",
                "case",
                "--batch-size",
                "16",
            ],
            runner=fake_runner,
        )
        == 0
    )

    assert captured["config"].pbf_path == pbf
    assert captured["config"].shard_path == shard
    assert captured["config"].run_id == "case"
    assert captured["config"].batch_size == 16
    summary = json.loads(capsys.readouterr().out)
    assert summary["matches_written"] == 2
