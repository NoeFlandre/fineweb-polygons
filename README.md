---
license: odc-by
pretty_name: FineWeb Polygons
language:
  - en
tags:
  - geospatial
  - openstreetmap
  - fineweb
  - information-retrieval
  - reproducibility
task_categories:
  - text-retrieval
configs:
  - config_name: v1
    data_files:
      - split: train
        path: data/monaco-v1-10bt-000-v3.jsonl
  - config_name: v2
    data_files:
      - split: train
        path: data/v2/monaco-v2-10bt-000-v1.jsonl
---

# FineWeb Polygons

FineWeb Polygons finds high-confidence FineWeb documents that are directly tied to OpenStreetMap polygons. V1 and V2 start with Monaco; the raw OSM extract is kept outside the repository at:

`/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf`

V1 scans one FineWeb Parquet shard at a time. It matches a polygon name exactly after Unicode normalization and case folding in either FineWeb `text` or `url`, and requires `Monaco` or `Principality of Monaco` in either field as context.

V2 derives its polygon profile from the raw PBF itself. It keeps meaningful names inside the Monaco city boundary, removes numeric and very short labels, deduplicates normalized names, and applies this selection rule:

- a polygon name in the URL is sufficient;
- a polygon name in the text is accepted only when the same text also contains `Monaco` or `Principality of Monaco`.

## Foundation contract

- `uv` owns the locked Python environment.
- Ruff, ty, pytest, mutation testing, and a CRAP gate are wired into local and CI commands.
- Docker and MkDocs Material are configured from the start.
- `LICENSE` and `CITATION.cff` are public project artifacts.
- Raw input, run manifests, checkpoints, logs, and generated artifacts stay on the Seagate project volume.
- V1/V2 processing is resumable in chunks of 32 Parquet row groups, appends structured JSON logs, and records input/configuration fingerprints in a manifest.
- The implementation uses small, deep modules with stable interfaces and YAGNI scope.

V1/V2 intentionally skip aliases, fuzzy matching, embeddings, and classifiers. They produce evidence JSONL on the Seagate; the raw shard is never uploaded.

## Public tiny-shard artifacts

The public dataset contains the filtered evidence from the first shard:

- V1: `data/monaco-v1-10bt-000-v3.jsonl`
- V2: `data/v2/monaco-v2-10bt-000-v1.jsonl`

These are evidence records, not a copy of the raw FineWeb shard.
Each record contains the complete matched FineWeb document in `text`; `text_excerpt`
is retained as a short preview.

## First-shard run

Keep the Hugging Face cache, virtual environment, raw data, and run outputs on the Seagate:

```bash
export HF_HOME="/Volumes/Seagate M3/projects/fineweb-polygons/.hf"
export UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache"
export UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons"

uv sync --locked
hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" \
  --local-dir "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb"

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v1-10bt-000-v2

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v2-10bt-000-v1 \
  --retrieval-version v2
```

Each run stores chunk checkpoints and a manifest under its run ID, logs under `logs/`, and merged evidence under `artifacts/`, all below the Seagate project root. One chunk opens the Parquet shard once and covers at most 32 row groups.

## License and upstream data

The dataset and project artifacts use the Open Data Commons Attribution License (ODC-By) v1.0, the same license shown on the [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb). FineWeb also notes that its Common Crawl source is subject to Common Crawl's terms of use; downstream releases must preserve applicable upstream notices and rights.

## Development

```bash
uv sync
just qa
just mutation
```

See the [development guide](https://noeflandre.github.io/fineweb-polygons/development/) and [foundation architecture](https://noeflandre.github.io/fineweb-polygons/architecture/foundation/) for the current scope.
