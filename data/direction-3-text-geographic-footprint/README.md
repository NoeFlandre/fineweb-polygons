# Direction 3: text to geographic footprint

This is the Hugging Face foundation card for
[`direction-3-text-geographic-footprint`](https://github.com/NoeFlandre/fineweb-polygons/tree/main/docs/directions/text-geographic-footprint).

**Status:** Planned foundation only<br>
**Public versions:** None yet<br>
**Dataset configuration:** None yet

Direction 3 will explore the reverse retrieval path:

`FineWeb text → geographic entity or footprint → topic evidence → related OSM polygons`

The planned work is to detect geographic references in FineWeb, resolve them to
stable geographic identities or footprints, identify land-use or geographic-
environment evidence, and relate that evidence to OSM polygons with explicit,
auditable links.

This release contains documentation and metadata only. It intentionally has no
data split, parquet file, dataset configuration, model, geocoder, embedding
index, LLM classification step, or geographic-disambiguation rule. The first
real output will receive its own immutable version, data path, manifest, and
Hugging Face configuration.

See the [GitHub direction README](https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/text-geographic-footprint/README.md),
the [machine-readable direction record](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/directions/direction-3-text-geographic-footprint.json),
and the [dataset catalog](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/catalog.json).
