# Retrieval versions

Version IDs are immutable contracts. A future change to a selection or matching rule must be published as a new version, such as `v3`; it must not overwrite the meaning of `v1` or `v2`.

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

## Reproducibility

The selected definition is copied into every run manifest under `configuration.retrieval_definition`. The configuration hash covers that definition. A run cannot resume if the saved definition or configuration no longer matches the selected version. Checkpoints, logs, raw inputs, and output artifacts remain under the Seagate data root.

## Published Hugging Face files

The public dataset exposes V1 and V2 as separate configs. Their retrieval rules
are comparable, but their historical evidence schemas are not identical:

- `v1/train` is the original excerpt-only release. It has `text_excerpt` and
  `url_excerpt`, but no full `text` field.
- `v2/train` is the full-text release. It has the complete FineWeb document in
  `text`, plus the excerpt fields.

The V1 file remains unchanged for reproducibility. A future regenerated file must
use a new artifact path and document its own schema.
