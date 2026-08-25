# Development guide

## Setup

```bash
export UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache"
export UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons"
uv sync --locked
uv run pre-commit install
```

Keep raw and generated data on `/Volumes/Seagate M3/projects/fineweb-polygons`. Do not copy PBF, Parquet, JSONL, database, or run-output files into this checkout.

## V1 shard scan

The first V1 input is FineWeb's `sample/10BT/000_00000.parquet` shard. Keep the Hugging Face cache on the Seagate and download only that file:

```bash
export HF_HOME="/Volumes/Seagate M3/projects/fineweb-polygons/.hf"
hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" \
  --local-dir "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb"
```

Run the resumable scan with:

```bash
uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v1-10bt-000-v2
```

V1 keeps only named polygon profiles. A match requires the normalized name in `text` or `url`, plus `Monaco` or `Principality of Monaco` in `text` or `url`; matching both fields is not required. The default checkpoint covers 32 row groups, so the scanner opens the shard once per checkpoint and can resume after an interruption.

Use `--retrieval-version v2`, `--retrieval-version v3`, or `--retrieval-version v4` to run the corresponding contract. V4 reuses V3's meaningful polygon profile and deduplication, but requires the polygon name and Monaco context in FineWeb text and does not use the URL to select documents. Version definitions are immutable and are copied into each run manifest; use a new version ID for a changed rule.

## V5 country runs

Use retrieval version v5 with the relevant country name:

```bash
uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v5-monaco-10bt-000-v3 \
  --retrieval-version v5 \
  --country-name "Monaco"

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/liechtenstein-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v5-liechtenstein-10bt-000-v2 \
  --retrieval-version v5 \
  --country-name "Liechtenstein"
```

V5 starts from all meaningful named PBF areas. It counts name frequency in OSM
and in FineWeb text, keeps OSM-unique names at or below the 0.1% document
cutoff, then requires the selected name and country name in the same text.
The URL is evidence only. The frequency artifact is saved in the run directory
and reused on restart.

## V6 country runs

Run V6 with the same country-specific inputs and a distinct run ID:

```bash
uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v6-monaco-10bt-000-v1 \
  --retrieval-version v6 \
  --country-name "Monaco"

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/liechtenstein-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v6-liechtenstein-10bt-000-v1 \
  --retrieval-version v6 \
  --country-name "Liechtenstein"
```

V6 applies V5's PBF name cleanup, OSM uniqueness filter, and 0.1% FineWeb
frequency cutoff. It then requires the polygon name and configured country name
in the FineWeb text within 500 normalized characters. The URL is not a
selection condition. Each output row keeps the full text, the original-text
sentence containing the polygon name, the sentence containing the country name,
and the closest normalized distance. V6 does not write `text_excerpt` or
`url_excerpt`.

## Red-green-refactor

New behavior follows TDD:

1. Write one focused failing test.
2. Run it and verify the expected RED failure.
3. Add the smallest implementation that makes it pass.
4. Run the focused test and the full suite.
5. Refactor only while the suite remains green.

The V1 pipeline is intentionally narrow. Do not add aliases, OSM tags, fuzzy matching, embeddings, or classifiers until the exact-match baseline has been evaluated.

## Quality commands

```bash
just format-check
just lint
just typecheck
just test
just crap
just docs
just mutation
just qa
```

The CRAP gate rejects any measured function with a score of 6 or higher. Mutation testing is serialized with one worker to keep Mac resource use bounded.
