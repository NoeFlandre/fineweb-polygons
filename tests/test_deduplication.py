import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from fineweb_polygons.deduplication import _record_identity, deduplicate_matches


def _record(
    marker: str,
    *,
    polygon_id: str = "way/1",
    document_id: str | None = "doc-1",
    url: str = "https://example.test/page",
    text: str = "Full text",
) -> str:
    return json.dumps(
        {
            "marker": marker,
            "polygon_id": polygon_id,
            "fineweb_document_id": document_id,
            "url": url,
            "text": text,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def test_deduplicate_matches_keeps_first_document_per_polygon(tmp_path: Path) -> None:
    path = tmp_path / "matches.jsonl"
    path.write_text(
        "\n".join(
            (
                _record("id-first"),
                _record("id-duplicate", url="https://other.test/page"),
                _record(
                    "fallback-first",
                    document_id=None,
                    url="https://example.test/fallback",
                    text="Fallback text",
                ),
                _record(
                    "fallback-duplicate",
                    document_id="",
                    url="https://example.test/fallback",
                    text="Fallback text",
                ),
                _record(
                    "other-polygon",
                    polygon_id="way/2",
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    retained = deduplicate_matches(path)

    assert retained == 3
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["marker"] for record in records] == [
        "id-first",
        "fallback-first",
        "other-polygon",
    ]
    assert not path.with_name(f".{path.name}.tmp").exists()


def test_deduplicate_matches_opens_both_files_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "matches.jsonl"
    path.write_text(_record("one") + "\n", encoding="utf-8")
    original_open: Any = Path.open
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def recording_open(self: Path, *args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)

    deduplicate_matches(path)

    assert calls == [
        (("r",), {"encoding": "utf-8"}),
        (("w",), {"encoding": "utf-8"}),
    ]


def test_deduplicate_matches_preserves_the_source_error_if_temp_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "missing.jsonl"
    temporary_path = path.with_name(f".{path.name}.tmp")
    original_open: Any = Path.open

    def fail_source(self: Path, *args: object, **kwargs: object) -> object:
        if self == path:
            raise RuntimeError("source failed")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_source)

    with pytest.raises(RuntimeError, match="source failed"):
        deduplicate_matches(path)

    assert not temporary_path.exists()


def test_record_identity_prefers_fineweb_id_with_a_stable_namespace() -> None:
    record = {
        "polygon_id": "way/1",
        "fineweb_document_id": 0,
        "url": "https://example.test/ignored",
        "text": "Ignored fallback text",
    }

    assert _record_identity(record) == ("way/1", "id", "0")


def test_record_identity_fallback_uses_url_and_full_text_digest() -> None:
    record = {
        "polygon_id": "relation/2",
        "url": "https://example.test/page",
        "text": "Full fallback text",
    }
    digest = hashlib.sha256(b"Full fallback text").hexdigest()

    assert _record_identity(record) == (
        "relation/2",
        "url-text",
        "https://example.test/page",
        digest,
    )


def test_record_identity_fallback_defaults_missing_fields() -> None:
    digest = hashlib.sha256(b"").hexdigest()

    assert _record_identity({"polygon_id": "way/3"}) == (
        "way/3",
        "url-text",
        "",
        digest,
    )


def test_record_identity_stringifies_non_string_text_before_hashing() -> None:
    digest = hashlib.sha256(b"42").hexdigest()

    assert _record_identity({"polygon_id": "way/5", "text": 42}) == (
        "way/5",
        "url-text",
        "",
        digest,
    )


def test_record_identity_encodes_text_with_explicit_utf8() -> None:
    encodings: list[str] = []

    class _EncodingRecorder(str):
        def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
            encodings.append(encoding)
            return super().encode(encoding, errors)

    _record_identity({"polygon_id": "way/4", "text": _EncodingRecorder("Café")})

    assert encodings == ["utf-8"]
