"""Deterministic dataset-card rendering for Direction 2 results."""

from __future__ import annotations

from collections.abc import Mapping

from fineweb_polygons.direction2.models import HF_CONFIG_NAME, OUTPUT_COLUMNS


def render_dataset_card(manifest: Mapping[str, object]) -> str:
    """Render a standalone card from the measured run manifest."""
    inventory = _mapping(manifest["polygon_inventory"], "polygon_inventory")
    results = _mapping(manifest["results"], "results")
    countries = _mapping(manifest["countries"], "countries")
    lines = [
        "---",
        "config_name: " + HF_CONFIG_NAME,
        "---",
        "# Direction 2 — lexical polygon candidates",
        "",
        "This artifact is a lexical candidate-generation POC. It scans the FineWeb "
        "shard for OSM polygon names and `name:*` aliases.",
        "",
        "## Measured run",
        "",
        f"- {inventory['polygons_read']} polygon objects read",
        f"- {inventory['names_indexed']} unique normalized names indexed",
        f"- {results['fineweb_docs_scanned']} FineWeb documents scanned",
        f"- {results['matches_found']} name mentions written",
        f"- {results['unique_polygons_matched']} unique polygons matched",
        "",
        "## Rule",
        "",
        "A row is written for every boundary-aware name or alias mention in the "
        "FineWeb document text. The row contains the matching sentence and the "
        "sentence immediately before and after it when available.",
        "",
        "No URL matching, LLM, embedding, thematic filter, or geographic "
        "disambiguation is used.",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        *(
            f"| `{column}` | {_column_description(column)} |"
            for column in OUTPUT_COLUMNS
        ),
        "",
        "## Source splits",
        "",
        "| Source | Matches |",
        "| --- | ---: |",
        *(
            f"| `{key}` | {_country_matches(value)} |"
            for key, value in sorted(countries.items())
        ),
        "",
        "This card is generated deterministically from the run manifest. The "
        "full Direction 2 contract is in the [GitHub direction README]("
        "https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/directions/"
        "lexical-candidates/README.md); "
        "the frozen Direction 1 archive remains in the same repository.",
        "",
    ]
    return "\n".join(lines)


def _mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest field {key!r} must be an object")
    return value


def _country_matches(value: object) -> object:
    return _mapping(value, "country")["matches_found"]


def _column_description(column: str) -> str:
    descriptions = {
        "polygon_id": "stable source/object identifier",
        "polygon_name": "OSM main `name` value",
        "matched_alias": "name or alias value that matched",
        "osm_tags": "all OSM tags as sorted JSON",
        "centroid": "centroid as JSON with latitude and longitude",
        "fineweb_url": "FineWeb document URL",
        "sentence": "the sentence containing the match",
        "context": "the sentence plus one neighboring sentence on each side",
    }
    return descriptions[column]
