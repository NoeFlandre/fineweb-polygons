# Technical debt register

This register records limitations that are known and intentional. It is kept
short so it does not become a second backlog.

| Item | Why it remains | Cleanup path |
| --- | --- | --- |
| Legacy retrieval stage files are larger than the newer lexical modules. | V1–V10 are frozen public contracts; broad mechanical splitting risks changing import paths and manifests. | Split only when a concrete behavior change creates a stable boundary, with compatibility tests first. |
| Direction 2 V3 is lexical candidate generation, not geographic disambiguation. | It is a deliberately measurable POC with no embeddings, NER, or semantic model. | Add one separately versioned disambiguation signal after candidate quality is evaluated. |
| Local model runtime is optional and supplied externally. | V10 must not download or publish model files as part of the source package or dataset. | Keep model paths explicit; add a runtime adapter only when another supported backend is required. |
| The Seagate data-root default is workstation-specific. | The project requirement is to keep raw and generated data off the SSD. | Use `FINEWEB_POLYGONS_DATA_ROOT` in CI, containers, and any other host. |
| pytest-bdd emits pytest 10 migration warnings. | The locked release still uses deprecated fixture registration APIs; acceptance scenarios currently pass. | Upgrade the dependency once it supports the new APIs and rerun acceptance tests without suppressing warnings. |

None of these items is a reason to weaken a deterministic gate or to change a
published experiment contract in place.
