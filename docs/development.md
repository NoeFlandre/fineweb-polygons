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
  --run-id v1-10bt-000
```

V1 keeps only named polygon profiles. A match requires the normalized name in `text` or `url`, plus `Monaco` or `Principality of Monaco` in `text` or `url`; matching both fields is not required.

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
