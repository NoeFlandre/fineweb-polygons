# Direction 2 lexical V1: broad lexical baseline

[Direction 2 overview](../README.md) · [GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

**Version ID:** `direction-2-lexical-v1`
**Hugging Face config:** `direction_2_lexical_v1`
**Public path:** `data/direction-2-lexical/v1/`
**Code:** [`src/fineweb_polygons/directions/lexical/v1/`](https://github.com/NoeFlandre/fineweb-polygons/tree/main/src/fineweb_polygons/directions/lexical/v1)

V1 is the broad baseline: every name and alias from every OSM area, matched
against FineWeb with no frequency, specificity, or geographic filter. It is
immutable and is not replaced by V2; V2 is a separate version with its own
path and configuration.

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

- `artifacts/direction-2-lexical/v1/monaco.parquet`
- `artifacts/direction-2-lexical/v1/liechtenstein.parquet`

Each file has exactly these columns:

`polygon_id`, `polygon_name`, `matched_alias`, `osm_tags`, `centroid`,
`fineweb_url`, `sentence`, `context`.

`osm_tags` and `centroid` are deterministic JSON strings so the Hugging Face
viewer exposes them as readable scalar columns. The manifest, JSONL progress
log, and deterministic dataset card are written beside the run artifacts on
the Seagate. The card reports polygon count, indexed-name count, FineWeb
documents scanned, mentions written, and unique polygons matched.


## Public release

The published files are `data/direction-2-lexical/v1/monaco.parquet` and
`data/direction-2-lexical/v1/liechtenstein.parquet`, loadable as
`load_dataset("NoeFlandre/fineweb-polygons", "direction_2_lexical_v1")`. The
run manifest is `metadata/direction-2-lexical/v1/manifest.json`.
