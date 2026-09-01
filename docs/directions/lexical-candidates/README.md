# Direction 2: lexical polygon candidates

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

Direction 2 is a separate lexical candidate-generation POC. It tests whether
an efficient name matcher can connect FineWeb documents to OSM polygon areas
before any semantic or geographic filtering is added.

## Versions

This direction currently has two versions:

- [`direction-2-lexical-v1`](#direction-2-lexical-v1), the broad lexical baseline;
- [`direction-2-lexical-v2`](lexical-v2/README.md), the specificity-aware candidate pass.

## Inputs

- `monaco-latest.osm.pbf`
- `liechtenstein-latest.osm.pbf`
- FineWeb `sample/10BT/000_00000.parquet`

The raw files stay on the Seagate project volume. The public results are
filtered evidence only.

## Measured run

The first run used the complete 1,048,581-document FineWeb shard listed above.

| Source | Polygon objects | Names indexed | Matches written | Unique polygons matched |
| --- | ---: | ---: | ---: | ---: |
| Monaco | 2,341 | 926 | 28,245,639 | 160 |
| Liechtenstein | 24,800 | 652 | 980,521 | 77 |
| **Total** | **27,141** | **1,578** | **29,226,160** | **237** |

The two Parquet files contain 29,226,160 rows in total. Their SHA-256 hashes
are `5b4c1912100e4666827945de32b4dbff3468760e593b26df9a697c4996564d85`
(Monaco) and
`967aa17e8bc2b00ab17a07c5f3f94da36c00fc2af133ea799d38ff2c2fac1f69`
(Liechtenstein).

## Polygon inventory

The reader uses osmium area processing and keeps every emitted OSM area. It
does not search for a country boundary or apply OSM tag filters. For each area
it records:

- a stable `source/way-or-relation/osm-id` identifier;
- the main `name` value;
- every non-empty `name:*` value as an alias;
- all OSM tags;
- an area-weighted longitude/latitude centroid.

Unnamed areas are counted but cannot produce a name match.

## Direction 2 lexical V1

V1 is the broad baseline. Its exact measured run and output contract are
documented below. It remains immutable and is not replaced by V2.

## Candidate rule

All non-empty main names and aliases are normalized with Unicode NFKC,
case-folding, and separator normalization. Unique normalized patterns are
loaded into one Aho–Corasick automaton. A match is boundary-aware, so a name
inside a longer word is not accepted.

The FineWeb `text` field is searched. The URL is retained as provenance but is
not searched or used as a condition. Every name mention creates one output row;
there is no deduplication or geographic disambiguation in this POC.

For each mention, the output stores the containing sentence and the sentence
immediately before and after it when those sentences exist. Sentence boundaries
use a deterministic punctuation splitter; no model is called.

## Output

The run writes one Parquet file per source under the Seagate artifact directory:

- `artifacts/direction-2/lexical-v1/monaco.parquet`
- `artifacts/direction-2/lexical-v1/liechtenstein.parquet`

Each file has exactly these columns:

`polygon_id`, `polygon_name`, `matched_alias`, `osm_tags`, `centroid`,
`fineweb_url`, `sentence`, `context`.

`osm_tags` and `centroid` are deterministic JSON strings so the Hugging Face
viewer exposes them as readable scalar columns. The manifest, JSONL progress
log, and deterministic dataset card are written beside the run artifacts on
the Seagate. The card reports polygon count, indexed-name count, FineWeb
documents scanned, mentions written, and unique polygons matched.

## Scope boundary

This direction intentionally has no LLM, embeddings, thematic vocabulary,
remote-sensing classifier, URL matching, frequency filter, or geographic
disambiguation. Those are separate future experiments and must not change this
version's meaning.

The implementation is in [`src/fineweb_polygons/direction2/`](https://github.com/NoeFlandre/fineweb-polygons/tree/main/src/fineweb_polygons/direction2).
The frozen first approach remains [Direction 1: FineWeb polygon retrieval](../fineweb-retrieval/README.md).

## Direction 2 lexical V2

V2 is documented in its own standalone contract:
[`lexical-v2/README.md`](lexical-v2/README.md). It retains the same polygon
inventory and matcher, but measures OSM name reuse and FineWeb document
frequency first. It discards unusable names, keeps distinctive names directly,
and requires the source country in the same sentence for generic names. Its
outputs live under the separate HF configuration `direction_2_lexical_v2` and
the separate `data/direction-2/lexical-v2/` path. On the first shard, it wrote
4,742 matches across 153 polygons, versus 29,226,160 matches across 237
polygons for V1. The reduction removes the worst high-frequency noise, but
rare names that are generic in ordinary language can still pass; V2 has no
semantic or geographic disambiguation.
