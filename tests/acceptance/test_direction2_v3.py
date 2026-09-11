"""Executable Gherkin acceptance coverage for the V3 public workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest_bdd import given, scenarios, then, when

from fineweb_polygons.directions.lexical.v3.models import (
    OUTPUT_COLUMNS_V3,
    Direction2V3RunConfig,
    Direction2V3RunSummary,
)
from fineweb_polygons.directions.lexical.v3.pipeline import run_direction2_v3

pytestmark = pytest.mark.acceptance

scenarios("features/direction2_v3.feature")


_OSM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<osm version="0.6" generator="acceptance-test">
  <node id="1" lat="43.70" lon="7.40" />
  <node id="2" lat="43.70" lon="7.41" />
  <node id="3" lat="43.71" lon="7.41" />
  <node id="4" lat="43.71" lon="7.40" />
  <way id="10">
    <nd ref="1" />
    <nd ref="2" />
    <nd ref="3" />
    <nd ref="4" />
    <nd ref="1" />
    <tag k="name" v="{name}" />
    <tag k="building" v="yes" />
  </way>
</osm>
"""


@given(
    "a tiny shard with named areas from Monaco and Liechtenstein",
    target_fixture="v3_config",
)
def tiny_v3_config(tmp_path: Path) -> Direction2V3RunConfig:
    monaco_pbf = tmp_path / "monaco.osm"
    liechtenstein_pbf = tmp_path / "liechtenstein.osm"
    monaco_pbf.write_text(
        _OSM_TEMPLATE.format(name="Palais du Prince"), encoding="utf-8"
    )
    liechtenstein_pbf.write_text(
        _OSM_TEMPLATE.format(name="Alps View"), encoding="utf-8"
    )

    shard = tmp_path / "shard.parquet"
    pq.write_table(
        pa.table(
            {
                "text": [
                    "Palais du Prince is visible in Monaco.",
                    "Alps View is visible in Liechtenstein.",
                    "This document has no named area.",
                ],
                "url": [
                    "https://example.test/palais-du-prince",
                    "https://example.test/alps-view",
                    "https://example.test/unrelated",
                ],
            }
        ),
        shard,
    )
    return Direction2V3RunConfig(
        monaco_pbf=monaco_pbf,
        liechtenstein_pbf=liechtenstein_pbf,
        shard_path=shard,
        output_dir=tmp_path / "artifacts",
        manifest_path=tmp_path / "runs" / "manifest.json",
        dataset_card_path=tmp_path / "artifacts" / "dataset-card.md",
        log_path=tmp_path / "logs" / "run.jsonl",
        name_inventory_path=tmp_path / "runs" / "name-inventory.json",
    )


@when("I run the Direction 2 V3 pipeline", target_fixture="v3_summary")
def run_v3(v3_config: Direction2V3RunConfig) -> Direction2V3RunSummary:
    return run_direction2_v3(v3_config)


@then("it writes one Parquet result for each source")
def results_exist(v3_config: Direction2V3RunConfig) -> None:
    assert (v3_config.output_dir / "monaco.parquet").is_file()
    assert (v3_config.output_dir / "liechtenstein.parquet").is_file()


@then("each result uses the published V3 schema")
def results_use_public_schema(v3_config: Direction2V3RunConfig) -> None:
    for source in ("monaco", "liechtenstein"):
        assert pq.read_schema(v3_config.output_dir / f"{source}.parquet").names == list(
            OUTPUT_COLUMNS_V3
        )


@then("the manifest marks the run complete")
def manifest_is_complete(
    v3_config: Direction2V3RunConfig, v3_summary: Direction2V3RunSummary
) -> None:
    manifest = json.loads(v3_config.manifest_path.read_text(encoding="utf-8"))

    assert v3_summary.matches_found == 2
    assert manifest["status"] == "complete"
