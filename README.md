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
        path: data/v1/monaco-v1-10bt-000-v3.jsonl
  - config_name: v2
    data_files:
      - split: train
        path: data/v2/monaco-v2-10bt-000-v2.jsonl
  - config_name: v3
    data_files:
      - split: train
        path: data/v3/monaco-v3-10bt-000-v1.jsonl
  - config_name: v4
    data_files:
      - split: train
        path: data/v4/monaco-v4-10bt-000-v1.jsonl
  - config_name: v5
    data_files:
      - split: monaco
        path: data/v5/monaco-v5-10bt-000-v3.jsonl
      - split: liechtenstein
        path: data/v5/liechtenstein-v5-10bt-000-v2.jsonl
  - config_name: v6
    data_files:
      - split: monaco
        path: data/v6/monaco-v6-10bt-000-v1.jsonl
      - split: liechtenstein
        path: data/v6/liechtenstein-v6-10bt-000-v1.jsonl
  - config_name: v7
    data_files:
      - split: monaco
        path: data/v7/monaco-v7-10bt-000-v1.jsonl
      - split: liechtenstein
        path: data/v7/liechtenstein-v7-10bt-000-v1.jsonl
  - config_name: v8
    data_files:
      - split: monaco
        path: data/v8/monaco-v8-10bt-000-v1-topic.jsonl
      - split: liechtenstein
        path: data/v8/liechtenstein-v8-10bt-000-v1-topic.jsonl
  - config_name: v9
    data_files:
      - split: monaco
        path: data/v9/monaco-v9-10bt-000-v1-topic-sentences.jsonl
      - split: liechtenstein
        path: data/v9/liechtenstein-v9-10bt-000-v1-topic-sentences.jsonl
  - config_name: v10
    data_files:
      - split: monaco
        path: data/v10/monaco-v10-10bt-000-v1-landuse.jsonl
      - split: liechtenstein
        path: data/v10/liechtenstein-v10-10bt-000-v1-landuse.jsonl
---

# FineWeb Polygons

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

The [public dataset catalog](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/catalog.json) maps every immutable V1–V10 file to its country split, standalone version README, and reproducibility manifest. The readable [GitHub catalog](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/dataset-catalog.md) and [HF metadata README](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/README.md) link back to both projects.

FineWeb Polygons finds high-confidence FineWeb documents that are directly tied to OpenStreetMap polygons. V1 through V4 use Monaco; V5 through V10 run the experiment for Monaco and Liechtenstein. Raw OSM extracts are kept outside the repository at:

`/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf`
and `/Volumes/Seagate M3/projects/fineweb-polygons/raw/liechtenstein-latest.osm.pbf`.

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

### V4 — all meaningful polygon areas with text-only matching

Run with `--retrieval-version v4`. V4 uses the same polygon profile as V3: every valid named area from a closed way or polygon relation, with the same name cleanup and normalized-name deduplication.

A FineWeb document is kept when the polygon name and `Monaco` or `Principality of Monaco` both appear in the text. The URL is retained as metadata and evidence, but it is not a selection condition. Final evidence is deduplicated per polygon and document, and the output keeps the complete FineWeb text.

The exact definitions are stored in [`src/fineweb_polygons/versions.py`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/src/fineweb_polygons/versions.py). Every run manifest copies the selected definition and hashes it as part of the configuration, so a changed definition cannot silently resume an old run. See the [version guide](https://noeflandre.github.io/fineweb-polygons/versions/) for the same contract in a readable format.

### V5 - specific polygon areas with country-in-text matching

Run with the v5 retrieval version and pass the relevant country name.
V5 starts with every meaningful named area from the PBF, just like V3 and V4.
It does not search for or require a country boundary.

Before retrieving documents, V5 counts each normalized polygon name:

1. A name must occur in only one OSM area.
2. It must occur in no more than 0.1% of the FineWeb shard's documents.
3. The configured country name is context only; it is never kept as a polygon
   candidate.

A document is kept only when the selected polygon name and the exact country
name both occur in the FineWeb text. The URL is retained as evidence but does
not select a document. Matching is case-insensitive and exact after
normalization. Results keep the complete FineWeb text and are deduplicated per
polygon and FineWeb document.

The first V5 pass writes a name-frequency.json file inside the run directory.
Its input fingerprints, counts, cutoff, and retained names are recorded there
and its SHA-256 is copied into the run manifest. A restart reuses this artifact
and skips the frequency pass.

## Foundation contract

- `uv` owns the locked Python environment.
- Ruff, ty, pytest, mutation testing, and a CRAP gate are wired into local and CI commands.
- Docker and MkDocs Material are configured from the start.
- `LICENSE` and `CITATION.cff` are public project artifacts.
- Raw input, run manifests, checkpoints, logs, and generated artifacts stay on the Seagate project volume.
- V1/V2/V3/V4/V5/V6 processing is resumable in chunks of 32 Parquet row groups, appends structured JSON logs, and records input/configuration fingerprints in a manifest. V5 and V6 also checkpoint their name-frequency pass. V7, V8, and V9 are content-fingerprinted, atomic post-processing runs that can reuse completed outputs. V10 adds an append-only per-row classification checkpoint and fingerprints both model runtimes.
- Every result is tied to an explicit retrieval version; new retrieval behavior must use a new version ID.
- The implementation uses small, deep modules with stable interfaces and YAGNI scope.

V1/V2/V3/V4/V5/V6 intentionally skip aliases, fuzzy matching, embeddings, and classifiers. V7 only segments already selected V6 documents; it does not change document selection. V8 filters already selected V7 documents with a fixed topic vocabulary at document level. V9 filters V8 sentences with the same vocabulary near polygon-name evidence; it does not add new polygon names or run an LLM. V9 publishes the full text, sentence list, URL, and compact topic evidence, while omitting redundant V6 matching fields from its rows. V10 classifies V9 candidate sentences with a local LFM model and publishes only its `yes` sentences. They produce evidence JSONL on the Seagate; the raw shard, model cache, and checkpoints are never uploaded.

## Public tiny-shard artifacts

The public dataset contains the filtered evidence from the first shard:

V5 and V6 publish one split for Monaco and one split for Liechtenstein. Both use the
same full-text evidence schema and can be inspected independently in the
Hugging Face viewer.

- V1: `data/v1/monaco-v1-10bt-000-v3.jsonl`
- V2: `data/v2/monaco-v2-10bt-000-v2.jsonl`
- V3: `data/v3/monaco-v3-10bt-000-v1.jsonl`
- V4: `data/v4/monaco-v4-10bt-000-v1.jsonl`
- V5 Monaco: `data/v5/monaco-v5-10bt-000-v3.jsonl`
- V5 Liechtenstein: `data/v5/liechtenstein-v5-10bt-000-v2.jsonl`
- V6 Monaco: `data/v6/monaco-v6-10bt-000-v1.jsonl`
- V6 Liechtenstein: `data/v6/liechtenstein-v6-10bt-000-v1.jsonl`
- V7 Monaco: `data/v7/monaco-v7-10bt-000-v1.jsonl`
- V7 Liechtenstein: `data/v7/liechtenstein-v7-10bt-000-v1.jsonl`
- V8 Monaco: `data/v8/monaco-v8-10bt-000-v1-topic.jsonl`
- V8 Liechtenstein: `data/v8/liechtenstein-v8-10bt-000-v1-topic.jsonl`
- V9 Monaco: `data/v9/monaco-v9-10bt-000-v1-topic-sentences.jsonl`
- V9 Liechtenstein: `data/v9/liechtenstein-v9-10bt-000-v1-topic-sentences.jsonl`
- V10 Monaco: `data/v10/monaco-v10-10bt-000-v1-landuse.jsonl`
- V10 Liechtenstein: `data/v10/liechtenstein-v10-10bt-000-v1-landuse.jsonl`

These are evidence records, not a copy of the raw FineWeb shard.
The published schemas are intentionally documented separately:

- V1 is the original excerpt-only release: it contains `text_excerpt` and
  `url_excerpt`, but no `text` field.
- V2 is the full-text release: it contains the complete FineWeb document in
  `text`, plus the short preview fields.
- V3 is the strict URL-and-text release: it contains the complete FineWeb
  document in `text`, plus the short preview fields and deduplicated evidence.
- V4 is the text-only release: it contains the complete FineWeb document in
  `text`, plus the short preview fields and deduplicated evidence. Its URL is
  retained for inspection but does not decide selection.

The V3 first-shard run read 1,048,581 FineWeb documents from a 2.0 GB Parquet
file. The shard contains 539,338,878 whitespace-separated words (and a FineWeb
`token_count` sum of 726,306,534). V3 retained 77 final evidence records across
7 polygon names.

The V4 first-shard run scanned the same 1,048,581 documents and retained 1,282
deduplicated evidence records across 42 polygon names. It contains 1,205
records selected from text alone and 77 where the URL also contains the name;
all 1,282 records retain the complete FineWeb text. Because V4 intentionally
removes the URL condition, the result is a higher-recall baseline and is still
broad for generic names: 1,067 records use the polygon name `Monaco`. This is
documented as an experiment, not as a claim that every text-only match is
already semantically correct.

The corrected V5 runs use the same 1,048,581-document shard. Monaco started
with 741 normalized names, kept 713 after removing 23 repeated OSM names, four
names above the 0.1% FineWeb cutoff, and the country name itself, then wrote
61 deduplicated records across 33 polygon names. Liechtenstein started with
565 names, kept 524, and wrote 12 deduplicated records across 11 polygon
names. Every result has the complete text; 57 Monaco and 10 Liechtenstein
records matched only in text, while the remaining four and two also had the
name in the URL. The country name appeared only as context.

These are still lexical results, not semantic proof. Manual review found
some false positives such as Buckingham, Bel Air, P 1, and Maxi. Frequency
filtering removes very common labels but cannot know whether a short name is
the intended place; a later version will need stronger evidence for that.

### V6 — local text-span evidence

V6 keeps the V5 polygon and frequency rules, then requires the selected polygon
name and configured country name to be within 500 normalized characters in the
FineWeb text. The URL is not a selection condition. Each V6 row keeps the full
FineWeb text, the closest normalized distance, and the original-text sentence
for the polygon name and country name. V6 does not include `text_excerpt` or
`url_excerpt` columns. The published Monaco and Liechtenstein files are separate
viewer splits under `data/v6/`.

### V7 — exact sentence lists from V6

V7 is a post-processing version, not a new retrieval rule. It reads the
published V6 rows, keeps every V6 field including the complete `text`, and adds
an ordered `sentences` list produced by `sat-3l-sm` from `wtpsplit`. The splitter
uses `split_on_input_newlines=false` and `strip_whitespace=false`; the pipeline
requires `''.join(sentences) == text`, so it cannot silently rewrite the source
document. Monaco has 45 rows and 4,569 sentences; Liechtenstein has 6 rows and
328 sentences. The model and segmentation settings are recorded in each V7
manifest. V7 does not upload the model or any raw FineWeb/OSM data.

The code and manifests still preserve the retrieval definition for each version;
new output should use a new artifact path rather than overwriting a published
file.

### V8 — topic-vocabulary filtering from V7

V8 is a post-processing version, not a new polygon retrieval rule. It reads
each V7 JSONL artifact and keeps a complete V7 row only when its full `text`
contains at least one of the 136 approved strong topic terms. Matching is
case-insensitive, uses Unicode NFKC normalization and whole-word boundaries,
and never examines the URL. The same vocabulary is used for Monaco and
Liechtenstein; it is not country-aware.

V8 preserves every kept V7 field, including the complete `text` and ordered
`sentences` list. It records the source, result, and vocabulary SHA-256 values,
term categories, row counts, and matching settings in its manifest. The
vocabulary JSON and both country manifests are published as HF metadata; raw
FineWeb, OSM, and model files remain on the Seagate.

On the first shard, V8 kept 39 of 45 Monaco V7 documents and filtered 6; the
kept rows cover 25 polygon names and contain 4,472 sentences. It kept all 6
Liechtenstein V7 documents, covering 5 polygon names and 328 sentences.
Category counts overlap because one document can match several categories.

### V9 — sentence-level topic filtering from V8

V9 is a post-processing version. It reads V8 rows and examines each original
sentence using the same 136-term vocabulary. A sentence is kept only when it
contains at least one whole-word topic term and is at most two sentence
positions from a sentence containing the polygon name. This bounded window
keeps place context without treating an unrelated topic word elsewhere in a
long document as evidence. The country remains an audit anchor; V8 has already
required the polygon name and country to be within 500 normalized characters.

V9 preserves the complete `text` and original `sentences` list, then adds
`sentences_with_topic_term` as a plain list of sentence strings for quick
review. The aligned `relevant_sentence_metadata` list stores the original sentence
index, matched terms, matched categories, topic counts, and polygon/country
sentence distances without repeating the sentence text. Rows with no local
topic sentence are filtered. The output and manifest are atomic and reusable
when the V8 input, vocabulary, settings, and result hash match.
This is V9 output schema version 4; the selection rule is unchanged. V9 rows
omit `context_fields`, `context_phrase`,
`country_name_sentence`, `matched_fields`,
`matched_name`, and `polygon_name_sentence` columns. The
internal V8 `context_phrase` input remains an anchor used to validate the
existing V8 sentence list; it is not published in V9 rows.

On the first shard, V9 kept 29 of 39 Monaco V8 rows and wrote 93 relevant
sentences. It kept 4 of 6 Liechtenstein V8 rows and wrote 9 relevant sentences.
The polygon-name evidence was in the same sentence for 33 Monaco sentences and
3 Liechtenstein sentences; the remaining matches were one or two sentence
positions away.

### V10 — local LFM land-use classification

V10 is a post-processing version over the V9 candidate sentence lists. It sends
each `sentences_with_topic_term` sentence to the local
`LiquidAI/LFM2.5-2.6B` model with the exact V10 classification prompt. The
prompt asks whether the sentence describes land use, land cover, geographic
environment, or an observable proxy. The classifier must return exactly
lowercase `yes` or `no`; any other answer stops the run instead of publishing
an uncertain result.

Only sentences classified `yes` are published. Rows with no `yes` sentences
are removed. The public V10 rows retain polygon/document identifiers, URL,
topic fields, the filtered `sentences_with_topic_term` list, and aligned
`relevant_sentence_metadata`; they do not contain the original full `text`,
the original `sentences` list, or `no` sentences. The source model snapshot,
the Seagate-derived MLX q4 runtime used on Apple Silicon, prompt hash, source
hash, checkpoint hash, and result hash are recorded in each manifest. The
checkpoint is local and resumable; model files, raw data, logs, and
checkpoints are not uploaded.

For speed, the default run uses deterministic greedy generation with the
`</think>` prefill and only four post-prefill tokens, and sends sentences in
batches of eight. The original sentence batch composition is preserved for
MLX output compatibility, while model fingerprints are reused within a
process. These are runtime optimizations only: the V10 prompt, label contract,
selection rule, output schema, and reproducibility fingerprints remain
unchanged.

On the first shard, V10 kept 21 of 29 Monaco V9 rows and wrote 62 `yes`
sentences. It kept 3 of 4 Liechtenstein V9 rows and wrote 4 `yes` sentences.

## First-shard run

Keep the Hugging Face cache, virtual environment, raw data, and run outputs on the Seagate:

```bash
export HF_HUB_CACHE="/Volumes/Seagate M3/projects/fineweb-polygons/cache/huggingface-v7/hub"
export UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup"
export UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8"

uv sync --locked
hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" \
  --local-dir "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb"

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v1-10bt-000-v3

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v2-10bt-000-v2 \
  --retrieval-version v2

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v3-10bt-000-v1 \
  --retrieval-version v3

uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet" \
  --run-id v4-10bt-000-v1 \
  --retrieval-version v4
```

Each run stores chunk checkpoints and a manifest under its run ID, logs under `logs/`, and merged evidence under `artifacts/`, all below the Seagate project root. One chunk opens the Parquet shard once and covers at most 32 row groups.

V5 commands for the two corrected runs:

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

V7 sentence segmentation reads those V6 artifacts and writes new artifacts;
all paths below remain on the Seagate:

```bash
export HF_HUB_CACHE="/Volumes/Seagate M3/projects/fineweb-polygons/cache/huggingface-v7/hub"

uv run fineweb-polygons segment-v7 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v6-monaco-10bt-000-v1-matches.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v7-monaco-10bt-000-v1-sentences.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v7-monaco-10bt-000-v1/manifest.json"

uv run fineweb-polygons segment-v7 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v6-liechtenstein-10bt-000-v1-matches.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v7-liechtenstein-10bt-000-v1-sentences.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v7-liechtenstein-10bt-000-v1/manifest.json"
```

V8 topic filtering reads those V7 artifacts and keeps only rows whose full
`text` contains at least one approved vocabulary term:

```bash
uv run fineweb-polygons filter-v8 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v7-monaco-10bt-000-v1-sentences.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v8-monaco-10bt-000-v1-topic.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v8-monaco-10bt-000-v1/manifest.json" \
  --vocabulary "/Volumes/Seagate M3/projects/fineweb-polygons/v8-topic-vocabulary-v1.json"

uv run fineweb-polygons filter-v8 \
  --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" \
  --input "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v7-liechtenstein-10bt-000-v1-sentences.jsonl" \
  --output "/Volumes/Seagate M3/projects/fineweb-polygons/artifacts/v8-liechtenstein-10bt-000-v1-topic.jsonl" \
  --manifest "/Volumes/Seagate M3/projects/fineweb-polygons/runs/v8-liechtenstein-10bt-000-v1/manifest.json" \
  --vocabulary "/Volumes/Seagate M3/projects/fineweb-polygons/v8-topic-vocabulary-v1.json"
```

## License and upstream data

The dataset and project artifacts use the Open Data Commons Attribution License (ODC-By) v1.0, the same license shown on the [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb). FineWeb also notes that its Common Crawl source is subject to Common Crawl's terms of use; downstream releases must preserve applicable upstream notices and rights.

## Development

```bash
uv sync
just qa
just mutation
```

See the [development guide](https://noeflandre.github.io/fineweb-polygons/development/) and [foundation architecture](https://noeflandre.github.io/fineweb-polygons/architecture/foundation/) for the current scope.
