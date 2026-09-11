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

The public runners are compatibility façades over focused domain and I/O
modules:

- `core/foundation.py` owns the Seagate-backed project layout and path
  validation; `core/normalization.py` owns shared text normalization.
- `core/artifact_io.py` owns stable JSONL writes, atomic text and JSON
  publication, manifest reads, temporary sibling paths, and bounded file
  hashing for every pipeline stage.
- `directions/retrieval/osm.py` and `directions/retrieval/scanning.py` own OSM
  profile extraction and FineWeb row-group scanning. `retrieval/runs.py` owns
  resumable scan orchestration, checkpoints, matching wiring, and frequency
  processing.
- `directions/retrieval/stages/v7.py`, `v8.py`, and `v9.py` own their
  post-processing contracts. `stages/v10.py` owns resumable classification,
  while `stages/inference.py` is the model-runtime boundary with exact prompt
  rendering and strict yes/no parsing.
- `directions/lexical/` owns the independent Direction 2 candidate-generation
  POCs. Its `v3/pipeline.py` is the public orchestration boundary for that
  direction and uses shared lexical matching and OSM readers.
- `registry.py` composes public commands and versions. Direction modules do not
  import the registry or one another.

This boundary is intentionally small: it reduces coupling without introducing
a second abstraction layer into the retrieval rules.

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
the `</think>` prefill, a four-token generation cap, and batches of eight.
Historical sentence batch composition is preserved because changing batch
contents can change MLX numerical results. Model fingerprints are cached within
one process. These optimizations do not change the prompt, accepted labels, row
order, or output schema.

## Logging contract

Each run must have a dedicated log path on the Seagate volume. Logs should include timestamps, severity, run ID, work-unit identity, input reference, event name, and enough exception context to diagnose a failed restart. Human-readable console output may be derived from the same events; it must not be the only record.

## Deferred decisions

The following remain intentionally open:

- how FineWeb is accessed and scaled beyond the first shard;
- whether the V2 exact lexical rule is sufficient for “directly tied” and “high confidence”;
- whether retrieval uses lexical, semantic, geospatial, or hybrid signals;
- the published schema, split strategy, and acceptance sample.
