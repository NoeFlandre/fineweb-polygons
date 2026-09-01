"""Deterministic Hugging Face card rendering for Direction 2 lexical V2."""

from __future__ import annotations

from collections.abc import Mapping

from fineweb_polygons.direction2.v2_models import (
    HF_CONFIG_NAME_V2,
    OUTPUT_COLUMNS_V2,
)


def render_dataset_card(manifest: Mapping[str, object]) -> str:
    """Render the V2 card from a completed run manifest."""
    inventory = _mapping(manifest["polygon_inventory"], "polygon_inventory")
    results = _mapping(manifest["results"], "results")
    countries = _mapping(manifest["countries"], "countries")
    configuration = _mapping(manifest["configuration"], "configuration")
    country_rows = sorted(
        (key, _mapping(value, "country")) for key, value in countries.items()
    )
    lines = [
        "---",
        "config_name: " + HF_CONFIG_NAME_V2,
        "---",
        "# Direction 2 — lexical polygon candidates V2",
        "",
        "This version reduces generic-name noise while keeping the retrieval "
        "lexical and deterministic.",
        "",
        "## Measured run",
        "",
        f"- {inventory['polygons_read']} polygon objects read",
        f"- {inventory['names_considered']} normalized names considered",
        f"- {inventory['names_indexed']} names indexed",
        f"- {inventory['generic_names']} names classified as generic",
        f"- {inventory['names_discarded']} names discarded",
        f"- {results['fineweb_docs_frequency_pass']} FineWeb documents in the "
        "frequency pass",
        f"- {results['fineweb_docs_match_pass']} FineWeb documents in the "
        "matching pass",
        f"- {results['matches_found']} matches written",
        f"- {results['distinctive_matches']} distinctive-name matches",
        f"- {results['generic_matches']} generic-name matches with country",
        f"- {results['unique_polygons_matched']} unique polygons matched",
        "",
        "## Rule",
        "",
        "A name is discarded when it has no letters, fewer than three "
        "alphabetic characters, or is the exact source country name. A name "
        "is generic when it is reused by more "
        "than one OSM polygon, appears in more than 0.1% of FineWeb documents, "
        "or is one token with at most eight letters.",
        "",
        "Distinctive names are matched directly. Generic names are kept only "
        "when the source country appears independently in the same sentence. "
        "The URL is provenance only. There is no LLM, embedding, thematic "
        "filter, tag filter, deduplication, or geographic disambiguation.",
        "",
        "## Configuration",
        "",
        "- FineWeb frequency ratio: "
        f"{configuration['fineweb_document_frequency_ratio']}",
        f"- Minimum alphabetic characters: {configuration['minimum_name_letters']}",
        "- Short single-token limit: "
        f"{configuration['short_single_token_max_letters']}",
        f"- Frequency inventory reused: {configuration['frequency_pass_reused']}",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        *(
            f"| {column} | {_column_description(column)} |"
            for column in OUTPUT_COLUMNS_V2
        ),
        "",
        "## Source splits",
        "",
        "| Source | Matches | Distinctive | Generic with country |",
        "| --- | ---: | ---: | ---: |",
        *(
            "| "
            + f"{key} | {value['matches_found']} | "
            + f"{value['distinctive_matches']} | {value['generic_matches']} |"
            for key, value in country_rows
        ),
        "",
        "This card is generated deterministically from the run manifest. The "
        "full contract is in the GitHub V2 README at "
        "https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/"
        "directions/lexical-candidates/lexical-v2/README.md. The original "
        "Direction 2 V1 README remains available.",
        "",
    ]
    return "\n".join(lines)


def _mapping(value: object, key: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"manifest field {key!r} must be an object")
    return value


def _column_description(column: str) -> str:
    descriptions = {
        "polygon_id": "stable source/object identifier",
        "polygon_name": "OSM main name value",
        "matched_alias": "name or alias value that matched",
        "osm_tags": "all OSM tags as sorted JSON",
        "centroid": "centroid as JSON with latitude and longitude",
        "fineweb_url": "FineWeb document URL",
        "sentence": "the sentence containing the match",
        "context": "the sentence plus one neighboring sentence on each side",
        "name_match_class": "distinctive_name or generic_name_with_country",
        "osm_polygon_count": "number of OSM polygons using the normalized name",
        "fineweb_document_frequency": (
            "FineWeb documents containing the normalized name"
        ),
    }
    return descriptions[column]
