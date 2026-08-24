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
  - config_name: v3
    data_files:
      - split: train
        path: data/v3/monaco-v3-10bt-000-v1.jsonl
---

# FineWeb Polygons

FineWeb Polygons finds high-confidence FineWeb documents that are directly tied to OpenStreetMap polygons. V1, V2, and V3 start with Monaco; the raw OSM extract is kept outside the repository at:

`/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf`

The version ID is part of the public contract. Existing version IDs are not silently changed; a behavior change gets a new version ID.

## Version contracts

### V1 — named polygon exact matching

Run with `--retrieval-version v1`. From the PBF, V1 keeps every named closed way and polygon relation. A FineWeb document is kept when a polygon name and `Monaco` or `Principality of Monaco` occur in the text or URL. Matching is case-insensitive and exact after normalization. The output keeps the complete FineWeb text.

The uploaded V1 file is the original excerpt-only publication and therefore has
`text_excerpt` and `url_excerpt` rather than a `text` column. New V1 runs made by
the current code keep the complete text; the old public file is preserved at its
original path and is not silently replaced.

### V2 — meaningful in-boundary exact matching

Run with `--retrieval-version v2`. From the raw PBF, V2 keeps named area objects inside the Monaco `admin_level=8` city boundary. It rejects names shorter than three characters, numeric-only names, and labels without letters, then deduplicates normalized names.

For FineWeb, a URL name match is enough. A text name match is kept only when the same text also contains `Monaco` or `Principality of Monaco`. The output keeps the complete FineWeb text, URL, and evidence fields.

### V3 — all meaningful polygon areas with strict URL-and-text matching

Run with `--retrieval-version v3`. From the raw PBF, V3 keeps every valid area produced from a closed way or relation. It uses only the main `name` tag, rejects names shorter than three characters, numeric-only names, and labels without letters, then deduplicates normalized names.

A FineWeb document is kept only when the polygon name appears in the URL and the polygon name plus `Monaco` or `Principality of Monaco` appear in the text. Final evidence is deduplicated per polygon and document, using the FineWeb ID when available and otherwise the URL plus a hash of the complete text. The output keeps the complete FineWeb text, URL, and evidence fields.

The exact definitions are stored in [`src/fineweb_polygons/versions.py`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/src/fineweb_polygons/versions.py). Every run manifest copies the selected definition and hashes it as part of the configuration, so a changed definition cannot silently resume an old run. See the [version guide](https://noeflandre.github.io/fineweb-polygons/versions/) for the same contract in a readable format.

## Foundation contract

- `uv` owns the locked Python environment.
- Ruff, ty, pytest, mutation testing, and a CRAP gate are wired into local and CI commands.
- Docker and MkDocs Material are configured from the start.
- `LICENSE` and `CITATION.cff` are public project artifacts.
- Raw input, run manifests, checkpoints, logs, and generated artifacts stay on the Seagate project volume.
- V1/V2/V3 processing is resumable in chunks of 32 Parquet row groups, appends structured JSON logs, and records input/configuration fingerprints in a manifest.
- Every result is tied to an explicit retrieval version; new retrieval behavior must use a new version ID.
- The implementation uses small, deep modules with stable interfaces and YAGNI scope.

V1/V2/V3 intentionally skip aliases, fuzzy matching, embeddings, and classifiers. They produce evidence JSONL on the Seagate; the raw shard is never uploaded.

## Public tiny-shard artifacts

The public dataset contains the filtered evidence from the first shard:

- V1: `data/monaco-v1-10bt-000-v3.jsonl`
- V2: `data/v2/monaco-v2-10bt-000-v1.jsonl`
- V3: `data/v3/monaco-v3-10bt-000-v1.jsonl`

These are evidence records, not a copy of the raw FineWeb shard.
The published schemas are intentionally documented separately:

- V1 is the original excerpt-only release: it contains `text_excerpt` and
  `url_excerpt`, but no `text` field.
- V2 is the full-text release: it contains the complete FineWeb document in
  `text`, plus the short preview fields.
- V3 is the strict URL-and-text release: it contains the complete FineWeb
  document in `text`, plus the short preview fields and deduplicated evidence.

The V3 first-shard run read 1,048,581 FineWeb documents from a 2.0 GB Parquet
file. The shard contains 539,338,878 whitespace-separated words (and a FineWeb
`token_count` sum of 726,306,534). V3 retained 77 final evidence records across
7 polygon names.

The code and manifests still preserve the retrieval definition for each version;
new output should use a new artifact path rather than overwriting a published
file.

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

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v3-10bt-000-v1 \
  --retrieval-version v3
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
