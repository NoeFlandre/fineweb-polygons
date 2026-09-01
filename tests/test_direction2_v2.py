from fineweb_polygons.direction2.v2_specificity import classify_name


def test_name_policy_rejects_short_and_numeric_names() -> None:
    assert (
        classify_name(
            "A",
            polygon_count=1,
            document_frequency=0,
            document_count=1000,
        ).decision
        == "discard"
    )
    assert (
        classify_name(
            "2",
            polygon_count=1,
            document_frequency=0,
            document_count=1000,
        ).decision
        == "discard"
    )


def test_name_policy_marks_short_single_token_names_generic() -> None:
    result = classify_name(
        "Central",
        polygon_count=1,
        document_frequency=1,
        document_count=1000,
    )

    assert result.decision == "generic"
    assert result.reason == "short_single_token"


def test_name_policy_marks_reused_names_generic() -> None:
    result = classify_name(
        "Old Mill",
        polygon_count=2,
        document_frequency=1,
        document_count=1000,
    )

    assert result.decision == "generic"
    assert result.reason == "osm_reuse"


def test_name_policy_marks_frequent_names_generic() -> None:
    result = classify_name(
        "Distinctive Hall",
        polygon_count=1,
        document_frequency=11,
        document_count=1000,
    )

    assert result.decision == "generic"
    assert result.reason == "fineweb_frequency"


def test_name_policy_keeps_rare_specific_names() -> None:
    result = classify_name(
        "Palais du Prince",
        polygon_count=1,
        document_frequency=1,
        document_count=1000,
    )

    assert result.decision == "distinctive"


def test_name_policy_marks_source_country_name_non_indexable() -> None:
    result = classify_name(
        "Monaco",
        polygon_count=1,
        document_frequency=1,
        document_count=1000,
        country_name="Monaco",
    )

    assert result.decision == "discard"
    assert result.reason == "country_name"
