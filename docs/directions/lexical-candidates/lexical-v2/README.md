# Direction 2 lexical V2: specificity-aware candidates

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

This is V2 of Direction 2. It is a new, reproducible artifact path; it does
not change the V1 result.

## Question

Can simple, measurable name-specificity rules remove the worst generic-name
noise before lexical retrieval at scale?

## Inputs

- the Monaco OSM PBF extract;
- the Liechtenstein OSM PBF extract;
- FineWeb `sample/10BT/000_00000.parquet`.

Raw files remain on the Seagate project volume. They are not uploaded.

## Measured run

The first run used the complete 1,048,581-document FineWeb shard.

| Source | Polygons read | Names indexed | Matches | Unique polygons |
| --- | ---: | ---: | ---: | ---: |
| Monaco | 2,341 | 908 | 3,184 | 120 |
| Liechtenstein | 24,800 | 647 | 1,558 | 33 |
| **Total** | **27,141** | **1,555** | **4,742** | **153** |

Across both sources, V2 considered 1,578 normalized names, discarded 23,
classified 215 as generic, and wrote 4,542 distinctive-name matches plus 200
generic-name matches with country evidence. V1 wrote 29,226,160 matches on the
same shard, so V2 reduced the lexical candidate volume by about 99.98%.

## Polygon inventory

The reader keeps every closed way and relation area emitted by osmium. For
each area it records its stable ID, main `name`, every non-empty `name:*`
alias, all tags, and centroid. No country-boundary or OSM-tag filter is used.

## Name selection

V2 groups normalized main names and aliases. Normalization uses Unicode NFKC,
case folding, and separator normalization. A name is discarded when it has no
letters, fewer than three letters, or is exactly the source country name.

For every other name, V2 records two frequencies (the FineWeb document-frequency
counter is part of the saved audit record):

1. how many distinct OSM polygons use it;
2. how many FineWeb documents contain it, counted once per document.

A name is generic when it is reused by multiple OSM polygons, appears in more
than 0.1% of the FineWeb documents, or is a single token of at most eight
letters. All remaining names are distinctive. These decisions and counts are
saved in the name inventory so the run can be audited and resumed.

## Document matching

V2 streams the FineWeb shard twice. The first pass counts name occurrences.
The second pass uses one Aho–Corasick matcher over accepted names and searches
the document `text` only. URL text is retained as provenance and is not a
selection condition.

- A distinctive name keeps every boundary-aware mention.
- A generic name keeps a mention only when the source country name appears
  independently in the same sentence.

The output has one row per accepted polygon/name mention. Each row contains
the matching sentence and up to one sentence on either side, plus the polygon
metadata and audit counts. There is no LLM, embedding, thematic filter,
deduplication, or geographic disambiguation.

## Public files

- HF config: `direction_2_lexical_v2`
- Monaco split: `data/direction-2/lexical-v2/monaco.parquet`
- Liechtenstein split: `data/direction-2/lexical-v2/liechtenstein.parquet`
- dataset card: `data/direction-2/lexical-v2/README.md`
- run manifest: `metadata/direction-2/lexical-v2/manifest.json`
- name inventory: `metadata/direction-2/lexical-v2/name-inventory.json`

The exact measured counts and SHA-256 hashes are generated from the run and
written into the dataset card and manifest.

Implementation: [`src/fineweb_polygons/direction2/`](https://github.com/NoeFlandre/fineweb-polygons/tree/main/src/fineweb_polygons/direction2).
