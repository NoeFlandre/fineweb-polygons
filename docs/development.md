# Development guide

## Setup

```bash
uv sync
uv run pre-commit install
```

Keep raw and generated data on `/Volumes/Seagate M3/projects/fineweb-polygons`. Do not copy PBF, Parquet, JSONL, database, or run-output files into this checkout.

## Red-green-refactor

New behavior follows TDD:

1. Write one focused failing test.
2. Run it and verify the expected RED failure.
3. Add the smallest implementation that makes it pass.
4. Run the focused test and the full suite.
5. Refactor only while the suite remains green.

The foundation package currently exposes only storage-path behavior. No OSM or FineWeb processing should be added until its design is agreed.

## Quality commands

```bash
just format-check
just lint
just typecheck
just test
just crap
just docs
just mutation
just qa
```

The CRAP gate rejects any measured function with a score of 6 or higher. Mutation testing is serialized with one worker to keep Mac resource use bounded.
