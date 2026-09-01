# FineWeb Polygons

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

This project connects OpenStreetMap polygons to relevant, high-confidence
documents from FineWeb. The first complete approach is preserved as
[Direction 1: FineWeb polygon retrieval](directions/fineweb-retrieval/README.md). The
initial working areas are Monaco and Liechtenstein.

The current V1, V2, V3, and V4 releases add narrow, resumable exact-match baselines over one FineWeb 10BT Parquet shard and Monaco polygon profiles. V5 adds the first frequency-filtered runs for Monaco and Liechtenstein. V6 adds a 500-character local text-span rule and sentence evidence. V7 post-processes V6 with `sat-3l-sm` and stores an ordered sentence list while retaining the complete text. V8 filters V7 with a fixed strong-topic vocabulary. V9 keeps topic sentences near polygon evidence. V10 classifies those candidates with a local LFM model and publishes only `yes` sentences. Their immutable contracts are documented in the version guide.

The [dataset catalog](dataset-catalog.md) maps every public data file to its
version, country split, standalone README, and manifest. The same catalog is
available as [JSON in the repository](dataset-catalog.json) and in the public
[Hugging Face metadata directory](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/tree/main/metadata).

[Direction 2: lexical polygon candidates](directions/lexical-candidates/README.md)
is a separate POC for validating fast lexical candidate generation. It reads
all areas from the Monaco and Liechtenstein extracts, indexes main names and
`name:*` aliases with Aho–Corasick, and streams the FineWeb shard into Parquet.
Its immutable V1 baseline is followed by V2, which measures name specificity
before retrieval and requires country evidence for generic names.

## Direction boundary

Direction 1 is a frozen exploratory baseline, not the only planned solution.
Its V1–V10 meanings and public HF paths are preserved for comparison. A new
retrieval idea should receive a new direction ID, documentation, code/output
namespace, and HF paths rather than changing this line.

## Current boundaries

- Source code and documentation live in this repository.
- The raw Monaco extract lives on the Seagate project volume.
- GitHub contains the code and metadata; Hugging Face contains versioned filtered evidence artifacts. The raw FineWeb shard and sentence-segmentation/classification models are not uploaded.
- V1, V2, V3, V4, V5, and V6 runs record immutable input references, the complete selected version definition, configuration fingerprints, chunk checkpoints, structured logs, and output manifests. V5 and V6 also record their reusable name-frequency artifacts. V7 records V6 and model fingerprints and publishes atomically. V8 records V7, vocabulary, and result fingerprints and publishes atomically. V9 records its vocabulary and local-sentence fingerprints. V10 records source/runtime model fingerprints, the exact prompt hash, a resumable sentence checkpoint, and the final result hash.
- The active Seagate layout keeps `raw/`, `runs/`, `logs/`, `artifacts/`, and reproducibility caches separate. Confirmed obsolete variants are moved, never deleted, under the dated `archive/legacy/` directory with a hash manifest.

V1 requires a normalized polygon name in either FineWeb `text` or `url`, plus a case-insensitive Monaco context phrase in either field. V2 builds its profile from meaningful areas inside the Monaco boundary and requires Monaco context in the same text when the name is matched in text; a URL name match is sufficient. V3 uses every meaningful polygon area and requires the polygon name in both URL and text, with Monaco context in the text. All versions deliberately exclude aliases, fuzzy matching, and semantic retrieval.

V4 uses the same polygon profile as V3 but requires the polygon name and Monaco context in the text; the URL is not a selection condition. V5 uses all meaningful areas from either PBF, drops repeated/common names, and requires the selected name plus the configured country name in the same text. V6 applies the same V5 filters and additionally requires the closest name/country spans to be at most 500 normalized characters apart. V6 keeps the complete text, the two matching sentences, and the saved distance. V7 does not select new documents: it splits every V6 `text` value into an ordered `sentences` list and verifies exact reconstruction. V8 keeps a V7 document only when full `text` contains at least one of 136 strong topic terms, using case-insensitive NFKC whole-word matching; URLs are ignored. V9 keeps local topic sentences near polygon evidence, adds a skim-friendly `sentences_with_topic_term` column, and removes six redundant legacy matching columns from its published rows. V10 sends every V9 candidate sentence to the local LFM classifier, keeps only exact `yes` labels, removes rows with no `yes` sentence, and publishes no full text or rejected sentences.

## License

The project and future dataset are released under ODC-By 1.0. See [`LICENSE`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/LICENSE) and [`CITATION.cff`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/CITATION.cff).
