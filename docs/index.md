# FineWeb Polygons

This project will connect OpenStreetMap polygons to relevant, high-confidence documents from FineWeb. The initial working area is Monaco.

The current V1 release adds a narrow, resumable exact-match baseline over one FineWeb 10BT Parquet shard and the named Monaco polygon profiles.

## Current boundaries

- Source code and documentation live in this repository.
- The raw Monaco extract lives on the Seagate project volume.
- Public GitHub and Hugging Face repositories contain metadata and code only until a later data-release decision.
- V1 runs record immutable input references, configuration fingerprints, row-group checkpoints, structured logs, and output manifests.

V1 requires a normalized polygon name in either FineWeb `text` or `url`, plus a case-insensitive Monaco context phrase in either field. It deliberately excludes aliases, tags, fuzzy matching, and semantic retrieval.

## License

The project and future dataset are released under ODC-By 1.0. See [`LICENSE`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/LICENSE) and [`CITATION.cff`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/CITATION.cff).
