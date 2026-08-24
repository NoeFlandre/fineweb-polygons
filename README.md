---
license: odc-by
pretty_name: FineWeb Polygons
language:
  - en
tags:
  - geospatial
  - openstreetmap
  - fineweb
  - information-retrieval
  - reproducibility
task_categories:
  - text-retrieval
---

# FineWeb Polygons

FineWeb Polygons is a foundation for finding high-confidence FineWeb documents that are directly tied to OpenStreetMap polygons. The first input is Monaco; the raw OSM extract is kept outside the repository at:

`/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf`

This repository currently contains only the project foundation. It does not scan OSM, download FineWeb, define relevance, or publish dataset rows.

## Foundation contract

- `uv` owns the locked Python environment.
- Ruff, ty, pytest, mutation testing, and a CRAP gate are wired into local and CI commands.
- Docker and MkDocs Material are configured from the start.
- `LICENSE` and `CITATION.cff` are public project artifacts.
- Raw input, run manifests, checkpoints, logs, and generated artifacts stay on the Seagate project volume.
- Future processing must be resumable, append useful structured context to run logs, and record enough input/configuration/version information to reproduce a run.
- The future implementation will use small, deep modules with stable interfaces and YAGNI scope.

The polygon-to-document matching approach, FineWeb access strategy, confidence definition, data schema, and partitioning strategy are intentionally deferred for design discussion.

## License and upstream data

The dataset and project artifacts use the Open Data Commons Attribution License (ODC-By) v1.0, the same license shown on the [FineWeb dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb). FineWeb also notes that its Common Crawl source is subject to Common Crawl's terms of use; downstream releases must preserve applicable upstream notices and rights.

## Development

```bash
uv sync
just qa
just mutation
```

See the [development guide](https://noeflandre.github.io/fineweb-polygons/development/) and [foundation architecture](https://noeflandre.github.io/fineweb-polygons/architecture/foundation/) for the current scope.
