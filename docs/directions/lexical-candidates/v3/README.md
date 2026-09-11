# Direction 2 lexical V3: evidence-scored candidates

[GitHub source](https://github.com/NoeFlandre/fineweb-polygons/tree/main/src/fineweb_polygons/directions/lexical/v3) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons) · [Direction 2 overview](../README.md)

V3 keeps every candidate that V2's lexical matcher finds, then adds cheap,
auditable evidence so generic names can be ranked instead of treated as equally
credible. V2 is unchanged and remains available for comparison.

## Inputs

- Monaco and Liechtenstein OSM PBF extracts;
- FineWeb `sample/10BT/000_00000.parquet`;
- all polygon areas emitted by osmium, with `name` and non-empty `name:*`
  aliases.

The FineWeb shard and raw PBFs stay on the Seagate project volume. V3 uses the
same Unicode NFKC, case-folding, separator normalization, sentence windows,
and measured name inventory as V2.

## Candidate scoring

Each V2 candidate occurrence receives these signals:

| Signal | Points | Meaning |
| --- | ---: | --- |
| `distinctive_name` | 2 | the name is unique in OSM and rare in FineWeb under V2's measured rules |
| `name_in_url` | 3 | the matched alias appears in the document URL |
| `country_in_sentence` | 3 | the source country appears in the matching sentence |
| `country_in_context` | 1 | the source country appears only in the one-sentence context window |
| `same_polygon_alias_nearby` | 2 | another alias of the same polygon appears nearby |
| `other_polygon_name_nearby` | 1 | another name from the same source appears nearby |

The score starts at 2 for a distinctive name and 0 for a generic name.
`country_in_context` is not added when the country is already in the matching
sentence.

The tiers are:

- `high_confidence`: score `>= 4`;
- `possible`: score `2` or `3`;
- `rejected`: score `< 2`.

All tiers are written. For the strictest view, filter the HF table on
`decision_tier == "high_confidence"`. This preserves recall while making every
decision inspectable.

## Processing

1. Read every OSM area and collect polygon metadata.
2. Count normalized-name document frequencies in a streaming FineWeb pass.
3. Build one Aho–Corasick matcher over V2-accepted names.
4. Stream FineWeb again, split only matched documents into sentence windows,
   derive nearby-name evidence from existing matches, and score each row.
5. Write one row per polygon/name occurrence to atomically published Parquet
   files.

The run saves a fingerprinted name inventory, manifest, JSONL progress log,
dataset card, input hashes, output hashes, and per-tier counts. An unchanged
inventory is reused on later runs.

## Measured run

The measured run used the existing 10BT shard and scanned 1,048,581 FineWeb
documents in each streaming pass. It read 27,141 OSM polygon objects, considered
1,578 normalized names, indexed 1,555, and discarded 23. It wrote 202,318
candidate rows across both sources:

| Source | Rows | Unique polygons | High confidence | Possible | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| Monaco | 34,108 | 144 | 778 | 7,253 | 26,077 |
| Liechtenstein | 168,210 | 73 | 1,083 | 13,387 | 153,740 |

The V3 candidate occurrence multiset is identical to V2, including the same
duplicate rows; V3 adds ranking evidence without changing recall. The strict
high-confidence view is much smaller (1,861 rows, 0.92% of candidates), but
the run still contains generic-name false positives: 560 high-confidence rows
use a generic name. The report gives stable examples and the remaining failure
modes in the published metadata comparison file.

## Output columns

V3 retains V2's polygon, URL, sentence, context, name-class, OSM-reuse, and
FineWeb-frequency columns, and adds:

- `decision_tier`;
- `evidence_score`;
- `evidence_reasons`;
- `name_in_url`;
- `country_in_sentence`;
- `country_in_context`;
- `same_polygon_alias_nearby`;
- `other_polygon_name_nearby`.

## Scope limits

V3 has no LLM, NER model, embeddings, thematic vocabulary, remote-sensing
classifier, deduplication, or geographic resolver. It is still lexical
candidate generation; the score is a ranking aid, not a proof that a document
describes the polygon.

## Public files

- HF config: `direction_2_lexical_v3`
- Monaco split: `data/direction-2-lexical/v3/monaco.parquet`
- Liechtenstein split: `data/direction-2-lexical/v3/liechtenstein.parquet`
- dataset card: `data/direction-2-lexical/v3/README.md`
- run manifest: `metadata/direction-2-lexical/v3/manifest.json`
- name inventory: `metadata/direction-2-lexical/v3/name-inventory.json`
- V2/V3 comparison: `metadata/direction-2-lexical/v3/comparison-v2-v3.md`

The generated HF card is derived from the run manifest. Once the measured run
is published, this page will link to its exact counts and hashes.
