# FineWeb Polygons

This project will connect OpenStreetMap polygons to relevant, high-confidence documents from FineWeb. The initial working area is Monaco.

The current V1 and V2 releases add narrow, resumable exact-match baselines over one FineWeb 10BT Parquet shard and Monaco polygon profiles.

## Current boundaries

- Source code and documentation live in this repository.
- The raw Monaco extract lives on the Seagate project volume.
- GitHub contains the code and metadata; Hugging Face contains versioned filtered evidence artifacts. The raw FineWeb shard is not uploaded.
- V1 runs record immutable input references, configuration fingerprints, chunk checkpoints, structured logs, and output manifests.

V1 requires a normalized polygon name in either FineWeb `text` or `url`, plus a case-insensitive Monaco context phrase in either field. V2 builds its profile from meaningful areas inside the Monaco boundary and requires Monaco context in the same text when the name is matched in text; a URL name match is sufficient. Both versions deliberately exclude aliases, fuzzy matching, and semantic retrieval.

## License

The project and future dataset are released under ODC-By 1.0. See [`LICENSE`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/LICENSE) and [`CITATION.cff`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/CITATION.cff).
