# Direction 2: lexical polygon candidates

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

Direction 2 is a separate lexical candidate-generation POC. It tests whether
an efficient name matcher can connect FineWeb documents to OSM polygon areas
before any semantic or geographic filtering is added.

## Versions

Each version has its own page, code subpackage, dataset card, manifest, and
Hugging Face configuration. Earlier versions stay immutable.

| Version | Page | Code | Config |
| --- | --- | --- | --- |
| `direction-2-lexical-v1` | [Broad lexical baseline](v1/README.md) | `directions/lexical/v1/` | `direction_2_lexical_v1` |
| `direction-2-lexical-v2` | [Specificity-aware](v2/README.md) | `directions/lexical/v2/` | `direction_2_lexical_v2` |

## Shared machinery

Everything both versions agree on lives directly in
`src/fineweb_polygons/directions/lexical/`: the OSM area reader (`osm.py`),
the deterministic sentence windows (`sentences.py`), the Aho-Corasick matcher
(`matching.py`), and the shared record shapes (`models.py`). A new version
adds a subpackage; it does not add a file prefix to this one.

## Inputs

- `monaco-latest.osm.pbf`
- `liechtenstein-latest.osm.pbf`
- FineWeb `sample/10BT/000_00000.parquet`

The raw files stay on the Seagate project volume. The public results are
filtered evidence only.

## Polygon inventory

The reader uses osmium area processing and keeps every emitted OSM area. It
does not search for a country boundary or apply OSM tag filters. For each area
it records:

- a stable `source/way-or-relation/osm-id` identifier;
- the main `name` value;
- every non-empty `name:*` value as an alias;
- all OSM tags;
- an area-weighted longitude/latitude centroid.

Unnamed areas are counted but cannot produce a name match.

## Scope boundary

This direction intentionally has no LLM, embeddings, thematic vocabulary,
remote-sensing classifier, URL matching, or geographic disambiguation. Those
are separate future experiments and must not change an existing version's
meaning. V1 additionally has no frequency filter; V2 adds one and is a
separate version because of it.

The implementation is in [`src/fineweb_polygons/directions/lexical/`](https://github.com/NoeFlandre/fineweb-polygons/tree/main/src/fineweb_polygons/directions/lexical).
The frozen first approach remains [Direction 1: FineWeb polygon retrieval](../fineweb-retrieval/README.md).
