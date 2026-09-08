# FineWeb Polygons metadata

This directory is the navigation layer for the public
[FineWeb Polygons dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons).

## Layout

Every research direction publishes under the same shape, so a new direction
adds a prefix rather than a new convention:

```
data/<direction>/<version>/<country>.<ext>      # published evidence + its README
metadata/<direction>/<version>/                 # manifests and side artifacts
```

- `data/direction-1-retrieval/v1` … `v10` — Direction 1, frozen.
- `data/direction-2-lexical/v1`, `v2` — Direction 2, active POC.
- `data/direction-3-text-geographic-footprint/` — reserved for Direction 3;
  no version or data file exists yet because the direction is still planned.

Hugging Face configuration names are stable and independent of these paths, so
`load_dataset("NoeFlandre/fineweb-polygons", "v10")` and
`load_dataset("NoeFlandre/fineweb-polygons", "direction_2_lexical_v2")`
keep working.

## Where to look

- [`catalog.json`](catalog.json) is the machine-readable index: every
  direction, version, split, configuration name, data path, metadata path, and
  standalone card. It is generated from
  [`src/fineweb_polygons/registry.py`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/src/fineweb_polygons/registry.py),
  so it cannot drift from the code.
- [`dataset-catalog.md`](dataset-catalog.md) is the same index in readable form.
- [`huggingface-configs.json`](huggingface-configs.json) is the configuration
  block used by the dataset card.
- `directions/<direction-id>.json` describes one direction's status and complete
  version list; its standalone GitHub README explains the research question and
  scope.
- Each `data/<direction>/<version>/README.md` is a concise contract for that
  version.
- Each manifest records the source fingerprints, settings, counts, and output
  hash needed to reproduce or audit that release.

Manifests are immutable records of the run that produced a file. A manifest
written before this reorganization still names the path the file had when it
was published; `catalog.json` is the authority on where a file lives now.

## Directions

- [Direction 1: FineWeb polygon retrieval](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/fineweb-retrieval/README.md)
  — frozen, V1–V10. HF record:
  [`directions/direction-1-fineweb-retrieval.json`](directions/direction-1-fineweb-retrieval.json).
- [Direction 2: lexical polygon candidates](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/lexical-candidates/README.md)
  — active POC. HF record:
  [`directions/direction-2-lexical-candidates.json`](directions/direction-2-lexical-candidates.json).
- [Direction 3: text to geographic footprint](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/text-geographic-footprint/README.md)
  — planned foundation only. HF record:
  [`directions/direction-3-text-geographic-footprint.json`](directions/direction-3-text-geographic-footprint.json).

The [GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) contains
the code and documentation. Raw FineWeb, OSM PBFs, model caches, checkpoints,
and logs stay on the Seagate project volume; this public dataset contains
filtered evidence only.
