Feature: Direction 2 V3 lexical candidate generation

  Scenario: Generate auditable candidates from two OSM sources
    Given a tiny shard with named areas from Monaco and Liechtenstein
    When I run the Direction 2 V3 pipeline
    Then it writes one Parquet result for each source
    And each result uses the published V3 schema
    And the manifest marks the run complete
