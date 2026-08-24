# FineWeb V1 Retrieval Design

## Goal

Run a small, reproducible retrieval experiment against one FineWeb 10BT Parquet
shard. The experiment finds documents that are likely related to named Monaco
OSM polygons while keeping the matching rules simple enough to audit.

## Scope

The V1 input is the Monaco OSM PBF already stored under the external data root.
Only polygon records with a non-empty canonical `name` are queries. Unnamed
polygons are skipped and counted in the run summary.

For each query, V1 normalizes the name with Unicode normalization, case folding,
and whitespace collapsing. It searches the FineWeb `text` and `url` fields for
the normalized name. A document is high-confidence when the normalized name
appears in at least one of those fields and a case-insensitive Monaco context
phrase (`Monaco` or `Principality of Monaco`) appears in the combined text and
URL evidence. The name does not need to appear in both fields. The matcher is
exact after normalization; it does not use fuzzy matching, aliases, OSM tags,
embeddings, or a classifier.

## Architecture and data flow

The pipeline has four small modules:

1. `polygon_profile` reads the OSM polygon layer and emits named polygon query
   records with stable OSM identifiers.
2. `normalization` applies the shared Unicode, case, and whitespace rules.
3. `fineweb_scan` streams one Parquet shard in bounded batches and checks only
   `text` and `url`, avoiding a full in-memory dataset.
4. `run` owns the manifest, partition checkpoints, structured logs, and JSONL
   evidence output.

The shard is divided into deterministic row partitions. Each partition is
processed once and marked complete only after its output is atomically written.
Rerunning a run skips completed partitions after validating the input checksum
and configuration fingerprint. A failed partition can be retried without
reprocessing completed partitions.

## Evidence output

Each match records the run ID, polygon ID, polygon name, FineWeb row index,
FineWeb document ID when present, URL, matched field(s), matched name, context
phrase, and a short evidence excerpt. The output is JSONL on the external data
root. A manifest records source paths, SHA-256 checksums, configuration,
normalization version, row counts, partition status, and timestamps.

## Performance and reproducibility

The first implementation uses column projection for `text`, `url`, and stable
FineWeb identifiers, batch scanning, compiled regular expressions, and one
process by default. It keeps raw inputs immutable and writes checkpoints,
artifacts, and logs only under the configured external data root.

## Error handling

The run fails before scanning if the PBF or Parquet shard is missing, the raw
input checksum changes, required FineWeb columns are absent, or the data root
is on the local repository volume. A malformed row is logged with its row index
and does not silently become a match. Output writes use temporary files followed
by atomic renames.

## Testing and quality gates

Tests cover normalization, exact matching across text and URL, context
requirements, polygon identity handling, checkpoint resumption, and malformed
input behavior. The repository quality gates remain Ruff, ty, pytest with
coverage, CRAP below 6, strict MkDocs, and mutation testing.

## Explicit non-goals

V1 does not download or process the full 10BT sample, build a persistent search
index, use fuzzy or semantic retrieval, infer aliases, score generic OSM tags,
or publish retrieved FineWeb documents to the Hugging Face dataset.
