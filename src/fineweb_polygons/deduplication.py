"""Deterministic final-artifact deduplication for retrieval matches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from fineweb_polygons.artifact_io import (
    atomic_text_output as _atomic_text_output,
)
from fineweb_polygons.artifact_io import (
    deterministic_temporary_path as _deterministic_temporary_path,
)


def deduplicate_matches(path: Path) -> int:
    """Keep the first match for each polygon/document identity."""
    identities: set[tuple[str, ...]] = set()
    retained = 0
    with (
        path.open("r", encoding="utf-8") as source,
        _atomic_text_output(
            path,
            temporary_factory=_deterministic_temporary_path,
        ) as output,
    ):
        for line in source:
            record = json.loads(line)
            identity = _record_identity(record)
            if identity in identities:
                continue
            identities.add(identity)
            output.write(line if line.endswith("\n") else f"{line}\n")
            retained += 1
    return retained


def _record_identity(record: Mapping[str, object]) -> tuple[str, ...]:
    polygon_id = str(record["polygon_id"])
    document_id = record.get("fineweb_document_id")
    if document_id is not None and str(document_id):
        return polygon_id, "id", str(document_id)
    url = str(record.get("url", ""))
    raw_text = record.get("text", "")
    text = raw_text if isinstance(raw_text, str) else str(raw_text)
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return polygon_id, "url-text", url, text_digest
