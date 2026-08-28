# Retrieval versions

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

Version IDs are immutable contracts. A future change to a selection or matching rule must be published as a new version, such as `v3`; it must not overwrite the meaning of `v1` or `v2`. V7, V8, and V9 are post-processing versions: V7 adds sentence lists to V6, V8 filters those V7 rows at document level, and V9 filters V8 sentences without changing polygon matching.

## V1

CLI flag: `--retrieval-version v1`

- Polygon profiles: keep every named closed way and polygon relation read from the PBF.
- Document rule: keep a document when a polygon name and `Monaco` or `Principality of Monaco` appear in the FineWeb text or URL.
- Matching: exact normalized, case-insensitive matching; the name and context do not need to be in the same field.
- Evidence: save the URL and complete FineWeb text.

## V2

CLI flag: `--retrieval-version v2`

- Polygon profiles: keep named area objects inside the Monaco object tagged `name=Monaco`, `boundary=administrative`, `admin_level=8`, and `place=city`.
- Name cleanup: reject names shorter than three characters, numeric-only names, and names without letters; deduplicate after normalization.
- Document rule: keep a document when the polygon name is in the URL, or when the polygon name and `Monaco` or `Principality of Monaco` are both in the text.
- Matching: exact normalized, case-insensitive matching; URL escapes are decoded.
- Evidence: save the URL, complete FineWeb text, matched fields, context fields, and short display excerpts.

## V3

CLI flag: `--retrieval-version v3`

- Polygon profiles: keep every valid area produced from a closed way or relation in the PBF; do not find or require a Monaco boundary.
- Name cleanup: read only the main `name` tag; reject names shorter than three characters, numeric-only names, and names without letters; deduplicate after normalization.
- Document rule: require the polygon name in the URL and require the polygon name plus `Monaco` or `Principality of Monaco` in the FineWeb text.
- Matching: exact normalized, case-insensitive matching; URL escapes are decoded; the text condition must be satisfied in the text field itself.
- Document deduplication: keep one result per polygon and FineWeb document, using FineWeb `id` when present and otherwise URL plus a SHA-256 hash of the complete text.
- Evidence: save the URL, complete FineWeb text, matched fields, context fields, and short display excerpts.

## V4

CLI flag: `--retrieval-version v4`

- Polygon profiles: keep every valid area produced from a closed way or relation in the PBF; do not find or require a Monaco boundary.
- Name cleanup: read only the main `name` tag; reject names shorter than three characters, numeric-only names, and names without letters; deduplicate after normalization.
- Document rule: require the polygon name and `Monaco` or `Principality of Monaco` in the FineWeb text. The URL is not a selection condition.
- Matching: exact normalized, case-insensitive matching in text; the URL is retained as metadata but is not used to accept or reject a document.
- Document deduplication: keep one result per polygon and FineWeb document, using FineWeb `id` when present and otherwise URL plus a SHA-256 hash of the complete text.
- Evidence: save the URL, complete FineWeb text, matched fields, context fields, and short display excerpts.

## V5

CLI flag: retrieval version v5

- Polygon profiles: start with every meaningful named area from the PBF,
  including closed ways and valid polygon relations. No country boundary is
  searched for or required.
- Name cleanup: use the main name tag, reject names shorter than three
  normalized characters, numeric-only names, and names without letters, then
  deduplicate normalized names.
- Specificity filter: count each normalized name in the PBF and in the FineWeb
  shard. Keep an OSM name only when it occurs once in the PBF and in no more
  than 0.1% of FineWeb documents and is not the configured country name. The
  country name supplies context only; it is never a polygon candidate.
- Document rule: require both the selected polygon name and the configured
  country name in the same FineWeb text. The URL is not a selection condition.
- Matching: exact normalized, case-insensitive matching in text. The country
  context is one exact configured name: Monaco for Monaco or Liechtenstein for
  Liechtenstein.
- Document deduplication: keep one result per polygon and FineWeb document,
  using FineWeb id when present and otherwise URL plus a SHA-256 hash of the
  complete text.
- Evidence: save the URL, complete FineWeb text, matched fields, context
  fields, and short display excerpts.
- Resumability: save the name-frequency counts and cutoff in the run
  directory; the manifest stores the artifact hash and the selected V5
  definition.

## Reproducibility

The selected definition is copied into every run manifest under `configuration.retrieval_definition`. The configuration hash covers that definition. A run cannot resume if the saved definition or configuration no longer matches the selected version. Checkpoints, logs, raw inputs, and output artifacts remain under the Seagate data root.

## Published Hugging Face files

The public dataset exposes V1, V2, V3, V4, V5, V6, V7, and V8 as separate configs. Their retrieval rules
are comparable, but their historical evidence schemas are not identical:

- `v1/train` is the original excerpt-only release. It has `text_excerpt` and
  `url_excerpt`, but no full `text` field.
- `v2/train` is the full-text release. It has the complete FineWeb document in
  `text`, plus the excerpt fields.
- `v3/train` is the strict URL-and-text release. It has the complete FineWeb
  document in `text`, plus the excerpt fields and deduplicated evidence.
- `v4/train` is the text-only release. It has the complete FineWeb document in
  `text`, plus the excerpt fields and deduplicated evidence; its URL is not a
  selection condition. On the first shard it contains 1,282 records across 42
  polygon names, including 1,205 records selected from text alone.

V5 exposes Monaco and Liechtenstein as separate viewer splits and keeps the
complete FineWeb text in the same evidence schema as V4.

## V6

CLI flag: retrieval version `v6`

- Polygon profiles: use the same meaningful named closed ways and valid polygon
  relations as V5. No country boundary or OSM tags are used.
- Name cleanup: use the main `name` tag, reject names shorter than three
  normalized characters, numeric-only names, and labels without letters, then
  keep only names that occur once in the PBF and in no more than 0.1% of FineWeb
  documents. The configured country name is context only.
- Document rule: require the selected polygon name and the exact configured
  country name in FineWeb text. The closest name/country spans must have at most
  500 characters between them after existing normalization. The URL is not a
  selection condition.
- Evidence: keep the full `text`, `name_country_distance`,
  `polygon_name_sentence`, and `country_name_sentence`. V6 omits both
  `text_excerpt` and `url_excerpt`; the URL itself remains metadata.
- Resumability: reuse the V5 name-frequency artifact contract under each V6 run
  directory and store the 500-character limit in the manifest definition and
  configuration hash.

V6 publishes Monaco and Liechtenstein as separate viewer splits. The V6 schema
is intentionally different from V1–V5 because its sentence fields replace both
excerpt fields.

## V7

V7 is a post-processing version, so it is not passed to `scan
--retrieval-version`. It reads each V6 JSONL artifact and runs `sat-3l-sm` from
`wtpsplit` over the complete `text` field.

- It preserves every V6 field and adds `sentences`, an ordered JSON list of the
  original sentence strings.
- It uses `split_on_input_newlines=false` and `strip_whitespace=false` so web
  line breaks are not silently treated as sentence boundaries and source
  whitespace remains visible.
- It rejects any model output for which `''.join(sentences) != text`.
- It records the source and result SHA-256 values, model ID, ONNX Runtime
  providers, batch settings, row count, and sentence count in the V7 manifest.
- It writes output and manifests atomically and reuses a completed result when
  all fingerprints still match.

The public V7 files are separate viewer splits:

- `v7/monaco`: 45 rows and 4,569 sentences.
- `v7/liechtenstein`: 6 rows and 328 sentences.

V7 retains the complete `text` for auditability. It uploads neither the raw
FineWeb/OSM inputs nor the sentence model.

## V8

V8 is a post-processing version, so it is not passed to `scan
--retrieval-version`. It reads the V7 JSONL artifact and tests the complete
FineWeb `text` field against the approved [V8 topic vocabulary](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons/blob/main/metadata/v8/v8-topic-vocabulary-v1.json).

- Keep a row when at least one of the 136 vocabulary terms occurs in full
  `text`.
- Matching is case-insensitive, applies Unicode NFKC normalization, and uses
  whole-word boundaries. `url` is never searched.
- The vocabulary contains only strong standalone terms grouped into five
  topic categories. It is shared by both country splits and is not
  country-aware.
- Preserve every V7 field unchanged, including complete `text` and the ordered
  `sentences` list.
- Record source, result, and vocabulary SHA-256 values, matching settings,
  category counts, and processed/kept/filtered row counts in the V8 manifest.
- Write atomically and reuse a completed result when all fingerprints match.

The public V8 files are separate viewer splits. Their manifests and the exact
vocabulary are available under the HF dataset `metadata/v8/` directory. The
implementation is in [`v8.py`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/src/fineweb_polygons/v8.py).

On the first shard, V8 kept 39 of 45 Monaco V7 documents and filtered 6; the
kept rows cover 25 polygon names and 4,472 sentences. It kept all 6
Liechtenstein V7 documents, covering 5 polygon names and 328 sentences.
Category counts overlap because a document may match several categories.

## V9

V9 is a post-processing step over the two V8 artifacts. It uses the same
approved vocabulary, but applies it to each sentence instead of the complete
document.

- Keep a sentence when at least one whole-word V8 topic term occurs in that
  sentence.
- Require the sentence to be within two sentence positions of a sentence that
  contains the polygon name, using the same case-insensitive normalized exact
  matching as earlier versions.
- Preserve the complete `text`, the URL, the original ordered `sentences`, and
  the useful V8 evidence fields. Add `sentences_with_topic_term`, a list of
  only selected sentence strings. Add the aligned
  `relevant_sentence_metadata` list containing the original index, matched
  terms, matched categories, topic counts, and polygon/country sentence
  distances, without repeating sentence text.
- Filter a document when it has no qualifying local topic sentence.
- Keep the configured country as an audit anchor. V8 already guarantees the
  polygon and country are within 500 normalized characters in the document.
- Record source, result, vocabulary, context-window, count, and matching
  settings in the manifest. Atomic output and matching hashes make a completed
  run reusable.
- Omit these redundant legacy V6 row fields from V9: `context_fields`,
  `context_phrase`, `country_name_sentence`, `matched_fields`, `matched_name`,
  and `polygon_name_sentence`. V9 still reads the V8 `context_phrase` input
  internally as the country anchor.
- The current V9 output schema is version 4: `sentences_with_topic_term` is text-only,
  while `relevant_sentence_metadata` contains the aligned per-sentence
  metadata. Removing redundant row fields changes the published schema only,
  not selection.

The public V9 paths are:

- `v9/monaco`: `data/v9/monaco-v9-10bt-000-v1-topic-sentences.jsonl`
- `v9/liechtenstein`: `data/v9/liechtenstein-v9-10bt-000-v1-topic-sentences.jsonl`

On the first shard, V9 kept 29 of 39 Monaco V8 rows and wrote 93 relevant
sentences. It kept 4 of 6 Liechtenstein V8 rows and wrote 9 relevant sentences.
The polygon-name evidence was in the same sentence for 33 Monaco sentences and
3 Liechtenstein sentences; the remaining matches were one or two sentence
positions away.

The V1 file remains unchanged for reproducibility. A future regenerated file must
use a new artifact path and document its own schema.
