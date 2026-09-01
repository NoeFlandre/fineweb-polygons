# Dataset catalog

The public dataset is [NoeFlandre/fineweb-polygons on Hugging Face](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons). The source code and runnable contracts are in the [GitHub repository](https://github.com/NoeFlandre/fineweb-polygons). The machine-readable catalog is available as [`docs/dataset-catalog.json`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/dataset-catalog.json) and as [`metadata/catalog.json`](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/catalog.json).

Direction 1 is the frozen first approach and its V1–V10 artifacts remain
available for comparison. Direction 2 is a separate lexical POC with its own
code, contract, and public paths. A genuinely different approach must use a
new direction ID and new public paths.

All published V1–V10 paths are historical and immutable. A changed rule gets a
new version and a new path. Each version also has a standalone README beside
its data and a manifest or metadata file describing its inputs and settings.

| Version | Stage | Public splits | Main decision or addition |
| --- | --- | --- | --- |
| V1 | Retrieval | Monaco | Named polygon name plus Monaco context in FineWeb text or URL; original release is excerpt-only. |
| V2 | Retrieval | Monaco | Meaningful named areas inside the Monaco boundary; URL name matches are sufficient, text matches need Monaco context. |
| V3 | Retrieval | Monaco | All meaningful polygon areas; polygon name in both URL and text, with Monaco context in text. |
| V4 | Retrieval | Monaco | V3 polygon profile; polygon name and Monaco context in text, without a URL condition. |
| V5 | Retrieval | Monaco, Liechtenstein | V4-style text match after OSM uniqueness and the 0.1% FineWeb document-frequency filter. |
| V6 | Retrieval | Monaco, Liechtenstein | V5 plus a 500 normalized-character name/country limit and sentence evidence columns. |
| V7 | Post-processing | Monaco, Liechtenstein | Splits each V6 full text with `sat-3l-sm` and verifies exact reconstruction. |
| V8 | Post-processing | Monaco, Liechtenstein | Keeps V7 documents containing at least one of 136 strong topic terms. |
| V9 | Post-processing | Monaco, Liechtenstein | Keeps V8 topic sentences within two sentence positions of polygon-name evidence; publishes compact schema version 4 without six redundant V6 matching fields. |
| V10 | Post-processing | Monaco, Liechtenstein | Classifies every V9 candidate sentence with the local LFM prompt and publishes only exact `yes` sentences; no full text or rejected sentences. |

## How to inspect a release

1. Open the version’s `data/v*/README.md` for its standalone contract.
2. Open its `metadata/` manifest for source fingerprints, settings, counts, and
   result hashes.
3. Use the HF viewer on the JSONL file to inspect the structured columns. V9’s
   `sentences_with_topic_term` column is the quickest pre-classification review
   entry point; V10's same column contains only LFM-`yes` sentences.

The files are filtered evidence from the first FineWeb 10BT shard. Raw FineWeb,
OSM PBFs, model caches, checkpoints, and logs remain on the Seagate project
volume and are not uploaded. V10's manifest records both the supplied native
LFM snapshot and the derived MLX q4 runtime used for Apple Silicon inference.

## Direction 2: lexical candidate generation

Direction 2 uses separate versioned configurations:
[`direction_2_lexical_v1`](directions/lexical-candidates/README.md) and
[`direction_2_lexical_v2`](directions/lexical-candidates/lexical-v2/README.md).
Both read every OSM area from both extracts, index the main `name` and all
non-empty `name:*` values with Aho–Corasick, search FineWeb `text`, and write
one row per boundary-aware mention with sentence ±1 context. V2 adds its
document-frequency and generic-name country gate; neither version uses the URL
as a condition, calls a model, or performs geographic disambiguation.

| Version | Stage | Public splits | Main decision or addition |
| --- | --- | --- | --- |
| Direction 2 lexical-v1 | Candidate generation | Monaco, Liechtenstein | All osmium areas and their names/aliases are matched in streamed FineWeb text; one Parquet row is written per mention. |
| Direction 2 lexical-v2 | Candidate generation | Monaco, Liechtenstein | V1 plus OSM reuse, FineWeb document-frequency, and short-name rules; generic names require the source country in the same sentence. |

The standalone card is published at
`data/direction-2/lexical-v1/README.md`; its deterministic run manifest is at
`metadata/direction-2/lexical-v1/manifest.json`.

V2 has its own standalone card at
`data/direction-2/lexical-v2/README.md`, deterministic name inventory at
`metadata/direction-2/lexical-v2/name-inventory.json`, and run manifest at
`metadata/direction-2/lexical-v2/manifest.json`.
