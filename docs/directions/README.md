# Research directions

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

A **direction** is a coherent line of experiments with its own question,
inputs, decision rules, outputs, and limitations. A **version** is one
immutable step inside a direction. Version numbers never cross a direction
boundary: Direction 2's first version is `direction-2-lexical-v1`, not V11.

## Registry

| Direction | Status | Versions | Purpose |
| --- | --- | --- | --- |
| [Direction 1: FineWeb polygon retrieval](fineweb-retrieval/README.md) | Frozen | V1–V10 | Lexically retrieve FineWeb evidence for OSM polygon names, then narrow it to topic sentences and local-model `yes` sentences. |
| [Direction 2: lexical polygon candidates](lexical-candidates/README.md) | Active POC | lexical-v1, lexical-v2 | Validate large-scale lexical candidate generation, then reduce generic-name noise with deterministic specificity rules. |
| [Direction 3: text to geographic footprint](text-geographic-footprint/README.md) | Planned | None yet | Start from geographic entities mentioned in text, recover their footprints, then relate topic evidence to OSM polygons. |

The authoritative machine-readable form of this table is
[`src/fineweb_polygons/registry.py`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/src/fineweb_polygons/registry.py).
Everything else — `metadata/catalog.json`, the per-direction records under
`metadata/directions/`, the [dataset catalog](../dataset-catalog.md), and the
`configs:` block of the dataset card — is generated from it by
`just catalog`, so those files cannot drift apart.

## One shape, everywhere

The same `<direction>/<version>` shape is used by the code, the
documentation, and the public dataset. That is what keeps a tenth version from
turning into a tenth special case.

| Concern | Path |
| --- | --- |
| Code | `src/fineweb_polygons/directions/<direction>/<version>/` |
| Tests | `tests/directions/<direction>/` |
| Documentation | `docs/directions/<direction>/<version>/README.md` |
| Public data | `data/<direction>/<version>/<country>.<ext>` |
| Public metadata | `metadata/<direction>/<version>/` |

Shared, direction-agnostic building blocks live in
`src/fineweb_polygons/core/`. Modules shared by every version of one direction
live directly in that direction's package. Nothing in `core/` knows which
directions exist, and no direction imports another;
[`tests/test_layering.py`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/tests/test_layering.py)
fails if either rule is broken.

## Adding a direction

1. Create `src/fineweb_polygons/directions/<name>/` with a narrow
   `__init__.py` that exposes its run interface and nothing else.
2. Put the version's contract constants — its version ID, Hugging Face
   configuration, and data prefix — in that version's `models.py`.
3. Add one `Version` and one `Command` to `registry.py`.
4. Run `just catalog` to regenerate every catalog artifact.
5. Write `docs/directions/<name>/README.md` and add it to the `mkdocs.yml`
   navigation.

Direction 3 is currently only a documented, registered foundation. It has no
public version, data file, or Hugging Face configuration yet; the first real
output will define those contracts explicitly.

`cli.py` never changes: it builds its parser from the registry.

## Separation rules

- A direction gets a stable ID, a standalone README, and a machine-readable
  record under `metadata/directions/`.
- A changed rule or output contract gets a new version and a new public path;
  it never changes the meaning of an earlier version.
- A genuinely different retrieval idea starts a new direction. It may reuse
  the FineWeb and OSM inputs, but it must not overwrite another direction's
  code, manifests, or public files.
- A Git tag marks the frozen endpoint of a direction. Direction 1 is pinned by
  `direction-1-fineweb-retrieval-v10`.
- GitHub stores code and documentation. Hugging Face stores public filtered
  evidence and its metadata. Raw data, caches, checkpoints, logs, and model
  files remain on the Seagate project volume.
