import pytest

from fineweb_polygons.versions import (
    available_retrieval_versions,
    get_retrieval_definition,
)


def test_retrieval_versions_have_a_stable_publication_order() -> None:
    assert available_retrieval_versions() == ("v1", "v2")


def test_v1_definition_records_its_complete_contract() -> None:
    definition = get_retrieval_definition("v1")

    assert definition.to_record() == {
        "version": "v1",
        "title": "Named polygon exact matching",
        "polygon_profile_version": "v1-all-named-closed-ways-and-polygon-relations",
        "matcher_version": "v1-exact-name-context",
        "polygon_rule": (
            "Keep every named closed way and polygon relation read from the PBF."
        ),
        "document_rule": (
            "Keep a document when a polygon name and Monaco context appear in "
            "either FineWeb text or URL."
        ),
        "evidence_rule": (
            "Use case-insensitive exact normalized matching and retain the full "
            "FineWeb text."
        ),
        "requires_text_context": False,
    }


def test_v2_definition_requires_text_context() -> None:
    definition = get_retrieval_definition("v2")

    assert definition.version == "v2"
    assert definition.polygon_profile_version == "v2-in-boundary-meaningful-names"
    assert definition.matcher_version == (
        "v2-exact-name-url-or-text-with-text-country-context"
    )
    assert definition.requires_text_context is True


def test_unknown_retrieval_version_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\Aretrieval_version must be v1 or v2\Z"):
        get_retrieval_definition("v3")
