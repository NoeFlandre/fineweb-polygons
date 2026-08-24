"""Deterministic final-artifact deduplication for retrieval matches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def deduplicate_matches(path: Path) -> int:
    """Keep the first match for each polygon/document identity."""
    temporary_path = path.with_name(f".{path.name}.tmp")
    identities: set[tuple[str, ...]] = set()
    retained = 0
    try:
        with (
            path.open("r", encoding="utf-8") as source,
            temporary_path.open("w", encoding="utf-8") as output,
        ):
            for line in source:
                record = json.loads(line)
                identity = _record_identity(record)
                if identity in identities:
                    continue
                identities.add(identity)
                output.write(line if line.endswith("\n") else f"{line}\n")
                retained += 1
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
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
