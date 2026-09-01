# FineWeb Polygons metadata

This directory is the navigation layer for the public
[FineWeb Polygons dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons).

The current public experiment is [Direction 1: FineWeb polygon retrieval](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/fineweb-retrieval/README.md),
covering immutable V1–V10 artifacts. Its machine-readable record is
[`directions/direction-1-fineweb-retrieval.json`](directions/direction-1-fineweb-retrieval.json).

The separate [Direction 2: lexical polygon candidates](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/lexical-candidates/README.md)
contains the immutable `direction-2-lexical-v1` baseline and the
specificity-aware `direction-2-lexical-v2` POC. Its machine-readable record
is [`directions/direction-2-lexical-candidates.json`](directions/direction-2-lexical-candidates.json).

- [`catalog.json`](catalog.json) maps every public Direction 1 and Direction 2
  data path to its country/split, source version, standalone README, and
  manifest files.
- Each `data/v*/README.md` is a concise contract for that version.
- Each manifest records the source fingerprints, settings, counts, and output
  hash needed to reproduce or audit the release.

The [GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) contains
the code and readable [version guide](https://noeflandre.github.io/fineweb-polygons/versions/).
Raw FineWeb, OSM PBFs, model caches, checkpoints, and logs stay on the Seagate
project volume; this public dataset contains filtered evidence only.
