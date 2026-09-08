# Direction 3: text to geographic footprint

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons) · [Direction registry](../README.md)

**Status:** Planned foundation only<br>
**Direction ID:** `direction-3-text-geographic-footprint`<br>
**Public versions:** None yet

Direction 3 explores the reverse of the first two approaches. Instead of
starting from an OSM polygon name and looking for matching text, it starts from
FineWeb text and asks:

> Which geographic entity or footprint does this text describe, what topic does
> it provide evidence for, and which OSM polygons does that footprint relate to?

## Proposed flow

1. Stream FineWeb documents or sentences.
2. Detect geographic entities and references in the text.
3. Resolve each reference to a stable geographic identity, such as OSM,
   Wikidata, or GeoNames.
4. Recover a point, boundary, administrative area, or other geographic
   footprint.
5. Identify land-use, land-cover, or geographic-environment evidence.
6. Relate the evidence footprint to OSM polygons and retain auditable links.

The first POC should keep candidate generation and evidence preservation
separate from later semantic ranking. A future record may therefore contain the
FineWeb document ID and URL, source sentence, detected mention, resolved entity
ID, footprint source, topic evidence, related OSM polygon ID, overlap or
distance, and a deterministic confidence breakdown.

## Boundary with the existing work

- [Direction 1](../fineweb-retrieval/README.md) is frozen at V10. It starts
  from polygon names and narrows exact FineWeb matches with topic processing.
- [Direction 2](../lexical-candidates/README.md) is an active lexical POC. It
  validates fast polygon-name candidate generation with Aho–Corasick matching.
- Direction 3 is a new question and must not overwrite either direction's code,
  outputs, manifests, or version meaning.

## Current state and non-goals

This commit creates the direction boundary, registry entry, catalog record, and
standalone documentation. It intentionally creates no version, dataset split,
Hugging Face configuration, model choice, geocoder, embedding index, LLM step,
or geographic-disambiguation rule. Those choices belong to the first version
after the POC contract is agreed.

Raw FineWeb, OSM PBFs, model files, caches, logs, and checkpoints remain on the
Seagate project volume. GitHub stores code and documentation; Hugging Face
stores public filtered outputs and their metadata.
