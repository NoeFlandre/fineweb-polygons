# Dataset catalog

The public dataset is [NoeFlandre/fineweb-polygons on Hugging Face](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons). The source code and runnable contracts are in the [GitHub repository](https://github.com/NoeFlandre/fineweb-polygons). The machine-readable catalog is available as [`docs/dataset-catalog.json`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/dataset-catalog.json) and as [`metadata/catalog.json`](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/catalog.json).

All published V1–V9 paths are historical and immutable. A changed rule gets a
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
| V9 | Post-processing | Monaco, Liechtenstein | Keeps V8 topic sentences within two sentence positions of polygon-name evidence; publishes compact schema version 3 without six redundant V6 matching fields. |

## How to inspect a release

1. Open the version’s `data/v*/README.md` for its standalone contract.
2. Open its `metadata/` manifest for source fingerprints, settings, counts, and
   result hashes.
3. Use the HF viewer on the JSONL file to inspect the structured columns. V9’s
   `relevant_sentences` column is the quickest human-review entry point.

The files are filtered evidence from the first FineWeb 10BT shard. Raw FineWeb,
OSM PBFs, model caches, checkpoints, and logs remain on the Seagate project
volume and are not uploaded.
