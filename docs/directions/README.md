# Research directions

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

This directory separates independent research approaches. A direction is a
coherent line of experiments with its own question, inputs, decision rules,
outputs, and limitations. A version is one immutable step inside a direction.

## Registry

| Direction | Status | Versions | Purpose |
| --- | --- | --- | --- |
| [Direction 1: FineWeb polygon retrieval](fineweb-retrieval/README.md) | Frozen | V1–V10 | Lexically retrieve FineWeb evidence for OSM polygon names, then narrow it to topic sentences and local-model `yes` sentences. |
| [Direction 2: lexical polygon candidates](lexical-candidates/README.md) | Active POC | lexical-v1, lexical-v2 | Validate large-scale lexical candidate generation, then reduce generic-name noise with deterministic specificity rules. |

Direction 1 is the complete first approach. Its public artifacts remain
available in the [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons), and its source and contracts remain in this GitHub repository.

## Separation rules

- A direction gets a stable ID, a standalone README, and a machine-readable
  record under `metadata/directions/`.
- A changed rule or output contract gets a new version and a new public path; it
  never changes the meaning of an earlier version.
- A genuinely different retrieval idea starts a new direction. It may reuse
  the FineWeb and OSM inputs, but it must not overwrite Direction 1 code,
  manifests, or HF files.
- A Git tag marks the frozen endpoint of a direction. Direction 1 is pinned by
  `direction-1-fineweb-retrieval-v10`.
- GitHub stores code and documentation. Hugging Face stores public filtered
  evidence and its metadata. Raw data, caches, checkpoints, logs, and model
  files remain on the Seagate project volume.

Direction 2 is recorded in
[`metadata/directions/direction-2-lexical-candidates.json`](https://github.com/NoeFlandre/fineweb-polygons/blob/main/metadata/directions/direction-2-lexical-candidates.json).
Its code lives under `src/fineweb_polygons/direction2/`, and its public files
use separate versioned HF configurations and paths:
`direction_2_lexical_v1` / `data/direction-2/lexical-v1/` and
`direction_2_lexical_v2` / `data/direction-2/lexical-v2/`.
