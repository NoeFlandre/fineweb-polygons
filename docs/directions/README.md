# Research directions

[GitHub repository](https://github.com/NoeFlandre/fineweb-polygons) · [Hugging Face dataset](https://huggingface.co/datasets/NoeFlandre/fineweb-polygons)

This directory separates independent research approaches. A direction is a
coherent line of experiments with its own question, inputs, decision rules,
outputs, and limitations. A version is one immutable step inside a direction.

## Registry

| Direction | Status | Versions | Purpose |
| --- | --- | --- | --- |
| [Direction 1: FineWeb polygon retrieval](fineweb-retrieval/README.md) | Frozen | V1–V10 | Lexically retrieve FineWeb evidence for OSM polygon names, then narrow it to topic sentences and local-model `yes` sentences. |

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

## Starting Direction 2

The next approach should begin with a new page under `docs/directions/`, a new
machine-readable record, and a new code/output namespace. Before it runs, record
the target relevance definition, candidate-generation rule, acceptance rule,
evaluation plan, and compute budget. Compare its results with Direction 1 V10
without modifying the frozen V1–V10 artifacts.
