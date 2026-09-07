"""Render the public dataset catalog and Hugging Face configs from the registry.

`metadata/catalog.json` and the `configs:` block of the dataset card are
generated, not hand-maintained. That makes `registry.py` the only place where
a direction, a version, a Hugging Face configuration, or a public path is
declared, so the three cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from fineweb_polygons.registry import DIRECTIONS, Direction, Version

SCHEMA_VERSION = 2

DATASET = {
    "id": "NoeFlandre/fineweb-polygons",
    "license": "odc-by",
    "visibility": "public",
    "github": "https://github.com/NoeFlandre/fineweb-polygons",
    "huggingface": "https://huggingface.co/datasets/NoeFlandre/fineweb-polygons",
    "fineweb_source": "HuggingFaceFW/fineweb sample/10BT/000_00000.parquet",
}

IMMUTABILITY = {
    "historical_paths_are_immutable": True,
    "rule": (
        "A changed retrieval or output contract receives a new version "
        "identifier and path."
    ),
    "raw_data_policy": (
        "Raw FineWeb, OSM PBF, model caches, manifests, logs, and run "
        "checkpoints stay on the Seagate project volume."
    ),
    "layout": (
        "Every direction publishes under data/<direction>/<version>/ and "
        "metadata/<direction>/<version>/."
    ),
}

SHARED_SOURCE = {
    "fineweb_shard": "sample/10BT/000_00000.parquet",
    "countries": ["Monaco", "Liechtenstein"],
    "data_scope": (
        "The public files contain filtered evidence records from the tiny "
        "first shard, not the raw FineWeb shard."
    ),
}


def _version_record(version: Version) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": version.id,
        "summary": version.summary,
        "countries": list(version.countries),
        "source_version": version.source_version,
        "huggingface_config": version.hf_config,
        "data_files": [dict(entry) for entry in version.data_files()],
        "metadata_files": list(version.metadata_files()),
        "card": version.card,
    }
    return record


def _direction_record(direction: Direction) -> dict[str, Any]:
    return {
        "id": direction.id,
        "name": direction.name,
        "status": direction.status,
        "package": direction.package,
        "readme": direction.documentation,
        "record": f"metadata/directions/{direction.id}.json",
        "latest_version": direction.latest_version.id,
        "versions": [_version_record(version) for version in direction.versions],
    }


def build_catalog(catalog_date: str) -> dict[str, Any]:
    """Return the complete public catalog for every declared direction."""
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_date": catalog_date,
        "dataset": DATASET,
        "immutability": IMMUTABILITY,
        "shared_source": SHARED_SOURCE,
        "directions": [_direction_record(direction) for direction in DIRECTIONS],
    }


def build_direction_record(direction: Direction, catalog_date: str) -> dict[str, Any]:
    """Return the standalone machine-readable record for one direction."""
    return {
        "schema_version": SCHEMA_VERSION,
        "record_date": catalog_date,
        "direction_id": direction.id,
        "name": direction.name,
        "status": direction.status,
        "package": direction.package,
        "readme": direction.documentation,
        "github": DATASET["github"],
        "huggingface": DATASET["huggingface"],
        "latest_version": direction.latest_version.id,
        "versions": [version.id for version in direction.versions],
        "outputs": {
            "hf_config": direction.latest_version.hf_config,
            "data_files": [dict(e) for e in direction.latest_version.data_files()],
        },
        "historical_outputs": {
            version.id: {
                "hf_config": version.hf_config,
                "data_files": [dict(e) for e in version.data_files()],
            }
            for version in direction.versions[:-1]
        },
    }


def build_hf_configs() -> list[dict[str, Any]]:
    """Return the `configs:` block for the Hugging Face dataset card."""
    return [
        {
            "config_name": version.hf_config,
            "data_files": [dict(entry) for entry in version.data_files()],
        }
        for direction in DIRECTIONS
        for version in direction.versions
    ]


def build_readme_configs() -> str:
    """Return the `configs:` YAML block for the dataset card front matter."""
    lines = ["configs:"]
    for entry in build_hf_configs():
        lines.append(f"  - config_name: {entry['config_name']}")
        lines.append("    data_files:")
        for data_file in entry["data_files"]:
            lines.append(f"      - split: {data_file['split']}")
            lines.append(f"        path: {data_file['path']}")
    return "\n".join(lines) + "\n"


def build_catalog_page() -> str:
    """Return the readable dataset-catalog page for the documentation site."""
    github = DATASET["github"]
    huggingface = DATASET["huggingface"]
    lines = [
        "# Dataset catalog",
        "",
        "<!-- Generated by scripts/build_catalog.py from"
        " fineweb_polygons.registry. Do not edit by hand. -->",
        "",
        f"The public dataset is [{DATASET['id']} on Hugging Face]({huggingface}).",
        f"The source code and runnable contracts are in the"
        f" [GitHub repository]({github}). The machine-readable catalog is",
        f"[`metadata/catalog.json`]({huggingface}/blob/main/metadata/catalog.json).",
        "",
        "Every direction publishes under `data/<direction>/<version>/`, with its",
        "manifests under the matching `metadata/<direction>/<version>/` prefix.",
        "A changed rule gets a new version and a new path; published paths are",
        "never reused for different content.",
        "",
    ]
    for direction in DIRECTIONS:
        lines += [
            f"## {direction.name}",
            "",
            f"**ID:** `{direction.id}` &middot; **Status:** {direction.status}"
            f" &middot; **Latest:** `{direction.latest_version.id}`",
            "",
            f"Documentation: [`{direction.documentation}`]"
            f"({github}/blob/main/{direction.documentation})",
            "",
            "| Version | Splits | Config | Path | Summary |",
            "| --- | --- | --- | --- | --- |",
        ]
        for version in direction.versions:
            splits = ", ".join(split for split, _ in version.files)
            lines.append(
                f"| `{version.id}` | {splits} | `{version.hf_config}` |"
                f" `{version.data_prefix}/` | {version.summary} |"
            )
        lines.append("")
    lines += [
        "## How to inspect a release",
        "",
        "1. Open the version's `README.md` beside its data for the standalone",
        "   contract.",
        "2. Open its manifest under `metadata/` for source fingerprints,",
        "   settings, counts, and the output hash.",
        "3. Load it by configuration name, for example",
        '   `load_dataset("NoeFlandre/fineweb-polygons", "v10")`.',
        "",
    ]
    return "\n".join(lines)
