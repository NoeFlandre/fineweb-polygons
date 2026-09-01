# Direction 1: FineWeb polygon retrieval

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons) · [Direction registry](../README.md)

**Status:** Frozen exploratory direction
**Direction ID:** `direction-1-fineweb-retrieval`
**Latest version:** V10
**GitHub snapshot:** [`direction-1-fineweb-retrieval-v10`](https://github.com/NoeFlandre/fineweb-polygons/releases/tag/direction-1-fineweb-retrieval-v10)

This is the first complete approach in the project. It starts with OpenStreetMap
polygon names, searches a tiny FineWeb 10BT shard with exact matching, and then
narrows the matches through topic terms, sentence proximity, and a local binary
classifier. It is preserved for comparison; it is not a claim that the method
is the final retrieval strategy.

## Scope and inputs

- FineWeb input: `HuggingFaceFW/fineweb`,
  `sample/10BT/000_00000.parquet`.
- First-shard size: 1,048,581 documents, about 2.0 GB, 539,338,878
  whitespace-separated words, and 726,306,534 FineWeb `token_count` tokens.
- OSM input: Monaco and Liechtenstein Geofabrik PBF extracts.
- Polygon profile: depending on the version, closed ways and valid polygon
  relations with a usable main `name` tag. The later versions do not use OSM
  tags to infer land use.
- Public output: filtered evidence records only. The raw FineWeb shard, PBFs,
  models, caches, checkpoints, logs, and run artifacts stay on the Seagate
  project volume.

## Complete experiment line

V1–V6 are document-retrieval versions. V7–V10 are post-processing versions;
they consume the previous version's public-shaped artifacts rather than
searching the FineWeb shard again.

### V1

Keeps named closed ways and polygon relations. A document is selected when the
polygon name and `Monaco` or `Principality of Monaco` occur in FineWeb text or
URL. Matching is exact, normalized, and case-insensitive. The original public
release contains excerpts.

### V2

Keeps meaningful named areas inside the Monaco `boundary=administrative`,
`admin_level=8`, `place=city`, `name=Monaco` boundary. It rejects names shorter
than three characters, numeric-only names, and labels without letters. A URL
name match is sufficient; a text name match also needs Monaco context in the
same text.

### V3

Keeps every valid area produced from a closed way or relation, using only the
main `name` tag and the same name cleanup. A document needs the polygon name in
the URL and the polygon name plus Monaco context in the text. Results are
deduplicated per polygon and FineWeb document.

### V4

Uses the V3 all-area polygon profile and cleanup, but removes the URL condition.
A document needs the polygon name and Monaco context in its text. The URL is
retained as evidence only.

### V5

Uses all meaningful named polygon areas from each PBF. A name must occur in one
OSM area and in no more than 0.1% of FineWeb documents; the configured country
name is context, never a polygon candidate. A document needs both the selected
polygon name and the exact country name in the same text. Monaco and
Liechtenstein use their own country names.

First-shard result: 61 Monaco records across 33 names and 12 Liechtenstein
records across 11 names.

### V6

Keeps the V5 rules and requires the polygon name and country name to be within
500 normalized characters in the document text. It saves the closest distance,
the polygon-name sentence, and the country-name sentence. The URL is not a
selection condition.

### V7

Reads V6 and splits each complete document into an ordered `sentences` list
with `sat-3l-sm` from `wtpsplit`. It requires that joining the sentence strings
reconstructs the original text exactly. It does not select new documents.

First-shard result: Monaco had 45 documents and 4,569 sentences;
Liechtenstein had 6 documents and 328 sentences.

### V8

Reads V7 and keeps a document when its complete text contains at least one of
136 approved strong topic terms. Matching is case-insensitive, Unicode NFKC
normalized, and whole-word based. The URL is ignored. It does not add polygon
names or use an LLM.

First-shard result: 39 Monaco documents and 6 Liechtenstein documents remained.

### V9

Reads V8 and keeps topic-matching sentences that are at most two sentence
positions from a sentence containing the polygon name. It adds the skim-friendly
`sentences_with_topic_term` list and aligned sentence metadata. It preserves
the full text for audit and removes redundant legacy matching fields from the
published rows.

First-shard result: Monaco had 29 documents and 93 relevant sentences;
Liechtenstein had 4 documents and 9 relevant sentences.

### V10

Reads V9 candidate sentences and sends every one to the local
`LiquidAI/LFM2.5-2.6B` model with the recorded land-use/geographic-environment
prompt. Only exact lowercase `yes` or `no` answers are valid. It publishes only
`yes` sentences and drops documents with no `yes` sentence; it does not publish
the original full text, original sentence list, or rejected sentences.

First-shard result: Monaco had 21 documents and 62 `yes` sentences;
Liechtenstein had 3 documents and 4 `yes` sentences.

The full contracts, manifests, exact prompt, schemas, and reproducibility
commands are in the [version guide](../../versions.md), [dataset catalog](../../dataset-catalog.md),
and [development guide](../../development.md). The public HF files and
manifests are listed in the [machine-readable catalog](https://github.com/NoeFlandre/fineweb-polygons/blob/main/metadata/catalog.json).

## What is frozen

The following are historical Direction 1 outputs and must not be overwritten:

- the V1–V10 selection and post-processing meanings;
- the public `data/v1/` through `data/v10/` HF paths;
- the corresponding manifests, vocabulary, schemas, and result hashes;
- the first-shard input and model fingerprints recorded by those manifests.

Quality, documentation, and test maintenance may continue on `main`. A change
that alters a selection rule, sentence boundary, classifier contract, or public
schema must receive a new version or a new direction and a new output path.

## Known limits

This line uses exact lexical matching, a fixed vocabulary, sentence proximity,
and one local binary classifier. It does not use aliases, fuzzy matching,
embeddings, web search, or a human-labelled evaluation set. OSM names can still
be ambiguous, frequency filtering is not semantic disambiguation, and the
published results come from only one tiny FineWeb shard. These limits are part
of the reason to explore a separate direction.

## Handoff to the next direction

Direction 1 is complete enough to serve as a reproducible baseline. Direction 2
can now be explored without changing it:

1. Keep this archive and the V10 HF output as the comparison baseline.
2. Define the new retrieval idea under a new `direction-2-*` ID.
3. Give it its own documentation, configuration, artifacts, and HF paths.
4. Report recall, precision or review quality, deduplication, runtime, and
   storage cost against the same source scope before scaling beyond the shard.

The repository is ready for that next research direction.
