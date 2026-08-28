# Development guide

## Setup

```bash
export UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup"
export UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8"
uv sync --locked
uv run pre-commit install
```

Keep raw and generated data on `/Volumes/Seagate M3/projects/fineweb-polygons`. Do not copy PBF, Parquet, JSONL, database, or run-output files into this checkout.

The [dataset catalog](dataset-catalog.md) is the index for public V1–V9 files.
Each version keeps its own standalone README and manifest paths. Historical
files are immutable; use a new version and output path for a changed contract.
The Seagate cleanup archive is recoverable and records SHA-256 values in
`archive/legacy/2026-08-26/move-manifest.json`.

## V1 shard scan

The first V1 input is FineWeb's `sample/10BT/000_00000.parquet` shard. Keep the Hugging Face cache on the Seagate and download only that file:

```bash
export HF_HUB_CACHE="/Volumes/Seagate M3/projects/fineweb-polygons/cache/huggingface-v7/hub"
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
  --run-id v1-10bt-000-v3
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

## V7 sentence lists

V7 is a post-processing step over the two V6 artifacts. Install the locked
`wtpsplit[onnx-cpu]` dependency and keep the model cache on the Seagate:

```bash
export HF_HUB_CACHE="/Volumes/Seagate M3/projects/fineweb-polygons/cache/huggingface-v7/hub"
export TRANSFORMERS_CACHE="/Volumes/Seagate M3/projects/fineweb-polygons/cache/huggingface-v7/transformers"

uv run fineweb-polygons segment-v7 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v6-monaco-10bt-000-v1-matches.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v7-monaco-10bt-000-v1-sentences.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v7-monaco-10bt-000-v1/manifest.json"
```

Repeat the command with the Liechtenstein V6 input and V7 output paths. The
default V7 model is `sat-3l-sm`; it uses ONNX Runtime, prefers CoreML on this
Mac, and keeps CPU available as a fallback. V7 preserves the complete `text`,
adds an ordered `sentences` list, and fails if joining that list does not
reconstruct the original text exactly. A matching completed manifest lets a
restart reuse the output without loading the model.

## V8 topic filtering

V8 reads the V7 artifacts and keeps a document when its complete `text` has at
least one of the approved strong topic terms. Matching is case-insensitive,
NFKC-normalized, and whole-word based; the URL is ignored. The same vocabulary
is used for both countries, and the full V7 row is preserved.

```bash
uv run fineweb-polygons filter-v8 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v7-monaco-10bt-000-v1-sentences.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v8-monaco-10bt-000-v1-topic.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v8-monaco-10bt-000-v1/manifest.json" \
  --vocabulary "/Volumes/Seagate M3/projects/fineweb-polygons/v8-topic-vocabulary-v1.json"
```

Repeat with the Liechtenstein V7 input and V8 output paths. A matching
completed manifest makes a restart reuse the result without scanning the input.
The vocabulary and run manifests remain on the Seagate and are published as
HF metadata for reproducibility.

## V9 local sentence-topic filtering

V9 reads the V8 artifacts and keeps only vocabulary-matching sentences within
two sentence positions of polygon-name evidence. It preserves the full text,
URL, sentence list, and compact topic evidence, then adds a text-only
`sentences_with_topic_term` list plus an aligned `relevant_sentence_metadata`
list.
The V8 vocabulary remains the single source of topic terms. V9 output schema
version 3 removes the redundant `context_fields`, `context_phrase`,
`country_name_sentence`, `matched_fields`, `matched_name`, and
`polygon_name_sentence` row fields; selection is unchanged:

```bash
uv run fineweb-polygons filter-v9 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v8-monaco-10bt-000-v1-topic.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v9-monaco-10bt-000-v1-topic-sentences.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v9-monaco-10bt-000-v1/manifest.json" \
  --vocabulary "/Volumes/Seagate M3/projects/fineweb-polygons/v8-topic-vocabulary-v1.json"

uv run fineweb-polygons filter-v9 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v8-liechtenstein-10bt-000-v1-topic.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v9-liechtenstein-10bt-000-v1-topic-sentences.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v9-liechtenstein-10bt-000-v1/manifest.json" \
  --vocabulary "/Volumes/Seagate M3/projects/fineweb-polygons/v8-topic-vocabulary-v1.json"
```

V9 keeps a sentence only when it has a whole-word topic match and the polygon
name occurs in the same sentence or within two sentence positions. The country
distance is saved for audit; it is not a second sentence-level gate because V8
already applied the document-level 500-character name/country rule. The
metadata list is aligned with `sentences_with_topic_term` by position.

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
