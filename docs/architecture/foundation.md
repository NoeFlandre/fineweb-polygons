# Foundation architecture

The foundation keeps the V1/V2 lexical baselines safe to run while leaving later retrieval methods open.

## Storage boundary

The repository contains code, tests, configuration, documentation, and small metadata. Data belongs under:

```text
/Volumes/Seagate M3/projects/fineweb-polygons/
├── raw/          # immutable source inputs, including Monaco OSM extracts
├── runs/         # manifests, checkpoints, and V5 frequency artifacts
├── logs/         # run logs and diagnostics
├── artifacts/    # generated, reviewable outputs, including V7/V8/V9/V10 results
└── archive/      # dated, hash-recorded legacy items; nothing is deleted
```

`ProjectPaths` is the only current code surface for this layout. It accepts an environment override for tests and controlled deployments, while defaulting to the Seagate path. The repository must never silently fall back to a local data directory.

## V1/V2 resumability contract

Each run has a stable identifier and writes a manifest before processing. The manifest records raw-input checksums, the complete immutable retrieval definition selected from `versions.py`, configuration and version fingerprints, polygon-profile fingerprints, output schema version, and deterministic chunk identities. A chunk covers up to 32 contiguous Parquet row groups, writes its JSONL output atomically, and can be skipped on restart after its checkpoint is complete. Existing manifests are rejected when their saved version definition changes.

The result is merged atomically only after all chunks have completed. V5 first
creates a name-frequency.json artifact from the PBF names and projected FineWeb
text. The artifact records the shard fingerprint, base polygon profiles, OSM
counts, FineWeb document counts, and the 0.1% cutoff. A matching artifact is
reused on restart. Raw inputs, checkpoints, logs, and results remain below the
configured Seagate data root.

## Code organization

The runner and V9 modules are stable compatibility façades. Their value objects
live in focused modules so orchestration can evolve without changing imports:

- `artifact_io.py` owns stable JSONL writes, atomic text and JSON publication,
  manifest reads, temporary sibling paths, and bounded file hashing for every
  pipeline stage.
- `run_models.py` owns scan configuration, summaries, partition identities, run
  layout, and profile preparation records.
- `runs.py` owns scan orchestration, matching wiring, checkpoints, and frequency
  processing while re-exporting the historical runner names.
- `v9_models.py` owns V9 configuration, summaries, and output counters.
- `v9.py` owns V9 decoding, evidence selection, serialization, and manifest
  coordination while re-exporting the historical V9 names.
- `v10_models.py` owns V10 configuration, summaries, and the classifier
  protocol.
- `v10_inference.py` owns the model-runtime boundary, exact prompt rendering,
  and strict yes/no parsing.
- `v10.py` owns V9 candidate decoding, resumable classification checkpoints,
  yes-only serialization, and model/prompt manifest coordination.

This boundary is intentionally small: it reduces coupling without introducing
a new abstraction layer into the retrieval rules.

## V2 profile and matching contract

V2 reads area geometry directly from the raw OSM PBF, finds the Monaco `admin_level=8` city boundary, and keeps only meaningful named areas whose representative point is inside that boundary. Names shorter than three normalized characters and numeric-only names are excluded; equivalent normalized names are represented once.

V2 accepts a document when a normalized polygon name appears in the URL, or when it appears in the text and that same text contains `Monaco` or `Principality of Monaco`. The evidence record retains the complete matched `text`, the URL, the fields that matched, and short excerpts for review.

## V5 specificity and matching contract

V5 uses the V3 all-area reader, so it does not find a country boundary. It
counts normalized names before building the matcher. An OSM name must occur in
one area and in no more than 0.1% of FineWeb documents. The configured country
name is context only, never a polygon candidate. A document then needs both the selected name and the
exact country name in the same text. The URL is evidence only. The matcher is
an Aho-Corasick exact-token matcher, and the full text is retained.

## V7 sentence post-processing contract

V7 reads a completed V6 JSONL artifact rather than the FineWeb shard. It sends
the complete `text` values to `sat-3l-sm` in bounded batches, preserves row
order and all V6 fields, and adds an ordered `sentences` list.
`split_on_input_newlines=false` and `strip_whitespace=false` keep the source
representation auditable. Every row must satisfy `''.join(sentences) == text`;
output and its manifest are published with atomic replacement. The manifest
fingerprints the V6 input, output, model, providers, and segmentation settings,
so a completed run can be reused safely. The model cache and all generated
files remain on the Seagate.

## V8 topic-filter post-processing contract

V8 reads a completed V7 JSONL artifact and never reopens the FineWeb shard.
For each full `text` value it uses a fixed 136-term vocabulary. A row is kept
when any term matches as a case-insensitive NFKC whole word. The URL is not
searched. Kept rows are copied unchanged, including `text` and `sentences`.

The vocabulary, source artifact, and output artifact are SHA-256 fingerprinted
in the manifest. The manifest also records matching settings, vocabulary
categories, row counts, and category document counts. Output and manifest
writes are atomic, and a completed run is reusable without re-reading the
vocabulary or source rows when all fingerprints still match.

## V10 model-classification contract

V10 reads only V9's candidate sentence list. It never reopens the FineWeb
shard or the OSM PBF. Every candidate is sent to the exact recorded prompt and
the local LFM runtime. The model chat template and `</think>` assistant
prefill are part of the reproducibility contract. A label is valid only when
it is exactly lowercase `yes` or `no`; malformed output fails the run.

The classifier writes a checkpoint record after each completed source row.
Output order follows V9 input order even when a batch contains multiple rows.
Only `yes` sentences are written, with aligned metadata; rows without a `yes`
sentence are omitted. Source and runtime model fingerprints, prompt hash,
settings, checkpoint hash, and final result hash are saved in the manifest.
The native source model and the derived Seagate MLX q4 runtime are local
inputs, not public dataset files.

The default runtime is optimized for the exact binary-label contract: it uses
the `</think>` prefill, a four-token generation cap, batches of eight, and an
in-run exact-string label cache. The cache reuses labels for duplicate
sentences, including labels recovered from a resumable checkpoint. These
optimizations do not change the prompt, accepted labels, row order, or output
schema.

## Logging contract

Each run must have a dedicated log path on the Seagate volume. Logs should include timestamps, severity, run ID, work-unit identity, input reference, event name, and enough exception context to diagnose a failed restart. Human-readable console output may be derived from the same events; it must not be the only record.

## Deferred decisions

The following remain intentionally open:

- how FineWeb is accessed and scaled beyond the first shard;
- whether the V2 exact lexical rule is sufficient for “directly tied” and “high confidence”;
- whether retrieval uses lexical, semantic, geospatial, or hybrid signals;
- the published schema, split strategy, and acceptance sample.
