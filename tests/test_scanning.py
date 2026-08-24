import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fineweb_polygons.matching import EvidenceMatcher
from fineweb_polygons.models import PolygonProfile
from fineweb_polygons.scanning import scan_row_group


def write_fixture_shard(path: Path) -> Path:
    table = pa.table(
        {
            "id": ["doc-0", "doc-1"],
            "text": ["No relevant place.", "Fontvieille is in Monaco."],
            "url": ["https://example.test/other", "https://example.test/fontvieille"],
        }
    )
    pq.write_table(table, path, row_group_size=2)
    return path


def test_scan_row_group_writes_matching_evidence(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path / "shard.parquet")
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    stats = scan_row_group(
        shard,
        row_group_index=0,
        matcher=matcher,
        output_path=output,
    )

    assert stats.rows_scanned == 2
    assert stats.matches_written == 1
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["fineweb_row_index"] == 1
    assert record["polygon_id"] == "way/1"


def test_scan_row_group_requires_text_and_url(tmp_path: Path) -> None:
    shard = tmp_path / "missing-url.parquet"
    pq.write_table(pa.table({"text": ["Monaco"]}), shard)

    with pytest.raises(ValueError, match=r"text.*url"):
        scan_row_group(
            shard,
            row_group_index=0,
            matcher=EvidenceMatcher([]),
            output_path=tmp_path / "partition.jsonl",
        )
