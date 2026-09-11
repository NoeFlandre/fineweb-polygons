"""Deterministic Hugging Face card rendering for Direction 2 lexical V3."""

from __future__ import annotations

from collections.abc import Mapping

from fineweb_polygons.directions.lexical.v3.models import (
    HF_CONFIG_NAME_V3,
    OUTPUT_COLUMNS_V3,
)


def render_dataset_card(manifest: Mapping[str, object]) -> str:
    """Render the V3 card from a completed run manifest."""
    inventory = _mapping(manifest["polygon_inventory"], "polygon_inventory")
    results = _mapping(manifest["results"], "results")
    countries = _mapping(manifest["countries"], "countries")
    configuration = _mapping(manifest["configuration"], "configuration")
    country_rows = sorted(
        (key, _mapping(value, "country")) for key, value in countries.items()
    )
    lines = [
        "---",
        "config_name: " + HF_CONFIG_NAME_V3,
        "---",
        "# Direction 2 — lexical polygon candidates V3",
        "",
        "V3 keeps the lexical recall of V2 and adds a small, deterministic "
        "evidence score for generic-name disambiguation.",
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
        f"- {results['matches_found']} candidates written",
        f"- {results['high_confidence_matches']} high-confidence candidates",
        f"- {results['possible_matches']} possible candidates",
        f"- {results['rejected_matches']} rejected candidates",
        f"- {results['unique_polygons_matched']} unique polygons matched",
        "",
        "## Decision rule",
        "",
        "The base score is 2 for a V2 distinctive name and 0 for a generic "
        "name. Add 3 when the name appears in the URL, 3 when the country "
        "appears in the matching sentence, 1 when it appears only in the "
        "context window, 2 when another alias of the same polygon is nearby, "
        "and 1 when another same-source polygon name is nearby.",
        "",
        "A score of 4 or more is `high_confidence`, 2-3 is `possible`, and "
        "below 2 is `rejected`. All tiers are retained so the decision is "
        "auditable; filter `decision_tier` to `high_confidence` for the "
        "strictest candidate view.",
        "",
        "There is no LLM, NER model, embedding, thematic filter, URL-only "
        "keep rule, deduplication, or geographic resolver in V3.",
        "",
        "## Configuration",
        "",
        f"- FineWeb frequency ratio: "
        f"{configuration['fineweb_document_frequency_ratio']}",
        f"- Minimum alphabetic characters: {configuration['minimum_name_letters']}",
        f"- Frequency inventory reused: {configuration['frequency_pass_reused']}",
        "",
        "## Columns",
        "",
        "| Column | Meaning |",
        "| --- | --- |",
        *(
            f"| {column} | {_column_description(column)} |"
            for column in OUTPUT_COLUMNS_V3
        ),
        "",
        "## Source splits",
        "",
        "| Source | Candidates | High confidence | Possible | Rejected |",
        "| --- | ---: | ---: | ---: | ---: |",
        *(
            "| "
            + f"{key} | {value['matches_found']} | "
            + f"{value['high_confidence_matches']} | "
            + f"{value['possible_matches']} | {value['rejected_matches']} |"
            for key, value in country_rows
        ),
        "",
        "This card is generated deterministically from the run manifest. The "
        "full contract is in the GitHub V3 README at "
        "https://github.com/NoeFlandre/fineweb-polygons/blob/main/docs/"
        "directions/lexical-candidates/v3/README.md. The V1 and V2 pages "
        "remain available for comparison. The deterministic V2/V3 report is "
        "published at metadata/direction-2-lexical/v3/comparison-v2-v3.md.",
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
        "name_match_class": "distinctive_name or generic_name",
        "osm_polygon_count": "number of OSM polygons using the normalized name",
        "fineweb_document_frequency": (
            "FineWeb documents containing the normalized name"
        ),
        "decision_tier": "high_confidence, possible, or rejected",
        "evidence_score": "bounded deterministic score before tiering",
        "evidence_reasons": "pipe-separated reasons contributing to the score",
        "name_in_url": "matched alias appears in the normalized URL",
        "country_in_sentence": "source country appears in the matching sentence",
        "country_in_context": "source country appears only in nearby context",
        "same_polygon_alias_nearby": "another alias of the polygon is nearby",
        "other_polygon_name_nearby": "another same-source polygon name is nearby",
    }
    return descriptions[column]
