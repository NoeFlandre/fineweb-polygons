# FineWeb Polygons

This project will connect OpenStreetMap polygons to relevant, high-confidence documents from FineWeb. The initial working area is Monaco.

The current release is deliberately foundation-only. It establishes the storage, packaging, testing, quality, documentation, and reproducibility boundaries without choosing a retrieval approach or processing the input.

## Current boundaries

- Source code and documentation live in this repository.
- The raw Monaco extract lives on the Seagate project volume.
- Public GitHub and Hugging Face repositories contain metadata and code only until a later data-release decision.
- Future runs will record immutable input references, configuration, code revision, checkpoints, logs, and output manifests.

## License

The project and future dataset are released under ODC-By 1.0. See [`LICENSE`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/LICENSE) and [`CITATION.cff`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/CITATION.cff).
