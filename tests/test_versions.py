import pytest

from fineweb_polygons.versions import (
    available_retrieval_versions,
    get_retrieval_definition,
)


def test_retrieval_versions_have_a_stable_publication_order() -> None:
    assert available_retrieval_versions() == ("v1", "v2", "v3", "v4", "v5", "v6")


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
        "requires_url_name": False,
        "deduplicate_documents": False,
    }


def test_v2_definition_requires_text_context() -> None:
    definition = get_retrieval_definition("v2")

    assert definition.version == "v2"
    assert definition.polygon_profile_version == "v2-in-boundary-meaningful-names"
    assert definition.matcher_version == (
        "v2-exact-name-url-or-text-with-text-country-context"
    )
    assert definition.requires_text_context is True
    assert definition.requires_url_name is False
    assert definition.deduplicate_documents is False


def test_v3_definition_requires_both_fields_and_document_deduplication() -> None:
    definition = get_retrieval_definition("v3")

    assert definition.version == "v3"
    assert definition.polygon_profile_version == "v3-all-meaningful-polygon-areas"
    assert definition.matcher_version == "v3-exact-name-url-and-text-context"
    assert definition.requires_text_context is True
    assert definition.requires_url_name is True
    assert definition.deduplicate_documents is True


def test_v4_definition_requires_text_name_and_document_deduplication() -> None:
    definition = get_retrieval_definition("v4")

    assert definition.version == "v4"
    assert definition.polygon_profile_version == "v4-all-meaningful-polygon-areas"
    assert definition.matcher_version == "v4-exact-name-and-text-context"
    assert definition.requires_text_context is True
    assert definition.requires_url_name is False
    assert definition.requires_text_name is True
    assert definition.deduplicate_documents is True
    assert definition.to_record()["requires_text_name"] is True


def test_v5_definition_filters_generic_names_and_uses_text_only() -> None:
    definition = get_retrieval_definition("v5")

    assert definition.version == "v5"
    assert definition.polygon_profile_version == "v5-specific-meaningful-polygon-areas"
    assert definition.matcher_version == "v5-exact-specific-name-and-country-in-text"
    assert definition.requires_text_context is True
    assert definition.requires_url_name is False
    assert definition.requires_text_name is True
    assert definition.deduplicate_documents is True
    assert definition.to_record()["requires_name_specificity"] is True


def test_v6_definition_adds_an_inclusive_500_character_text_span_limit() -> None:
    definition = get_retrieval_definition("v6")

    assert definition.version == "v6"
    assert definition.polygon_profile_version == "v6-specific-meaningful-polygon-areas"
    assert definition.matcher_version == (
        "v6-exact-specific-name-and-country-within-500-normalized-characters"
    )
    assert definition.requires_text_context is True
    assert definition.requires_url_name is False
    assert definition.requires_text_name is True
    assert definition.requires_name_specificity is True
    assert definition.deduplicate_documents is True
    assert definition.max_name_country_distance == 500
    assert definition.to_record()["max_name_country_distance"] == 500


def test_unknown_retrieval_version_is_rejected() -> None:
    with pytest.raises(
        ValueError, match=r"\Aretrieval_version must be v1, v2, v3, v4, v5, or v6\Z"
    ):
        get_retrieval_definition("v7")
