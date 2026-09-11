# ADR 0001: explicit quality and side-effect boundaries

Status: accepted

## Context

This repository contains two research directions, resumable data pipelines,
and public experiment contracts. The raw shard, OSM extracts, model files, and
run artifacts are local research data; source code and small reproducibility
metadata are versioned in GitHub.

## Decision

- Keep domain transformations deterministic and in direction-specific modules.
- Keep filesystem, hashing, atomic publication, and project-root policy behind
  the `core` I/O and foundation modules.
- Keep directions independent; compose public commands through `registry.py`.
- Require explicit property, acceptance, architecture, typing, coverage/CRAP,
  mutation, packaging, documentation, and smoke checks in the quality workflow.
- Keep the Seagate path as the deliberate default data root, with
  `FINEWEB_POLYGONS_DATA_ROOT` as the override for CI and controlled runs.
  External model paths are supplied by CLI arguments or environment variables,
  never by a checkout-specific path.

## Consequences

The quality gates are visible and repeatable, and failed writes cannot publish
partial artifacts. The Seagate default is intentionally platform-specific for
this workstation; portable execution must provide an explicit data-root
override. New behavior still requires a new version or direction contract.
