# Foundation architecture

The foundation has one deliberate job: make future processing safe to design and run without committing to a polygon-to-document matching method.

## Storage boundary

The repository contains code, tests, configuration, documentation, and small metadata. Data belongs under:

```text
/Volumes/Seagate M3/projects/fineweb-polygons/
├── raw/          # immutable source inputs, including Monaco OSM extracts
├── runs/         # run-specific manifests and checkpoint state
├── logs/         # run logs and diagnostics
└── artifacts/    # generated, reviewable outputs
```

`ProjectPaths` is the only current code surface for this layout. It accepts an environment override for tests and controlled deployments, while defaulting to the Seagate path. The repository must never silently fall back to a local data directory.

## Future resumability contract

The later pipeline design must give each run a stable identifier and write a manifest before processing. The manifest is expected to record the raw-input checksum, source revision or snapshot, configuration, dependency lock, code revision, and output schema version. Work units must have deterministic identities so a restart can skip verified completed units and re-run only missing or invalid outputs.

The current foundation does not implement checkpoint storage, because the unit of work and output format have not been chosen.

## Future logging contract

Each run must have a dedicated log path on the Seagate volume. Logs should include timestamps, severity, run ID, work-unit identity, input reference, event name, and enough exception context to diagnose a failed restart. Human-readable console output may be derived from the same events; it must not be the only record.

## Deferred decisions

The following remain intentionally open:

- how OSM polygons are normalized and identified;
- how FineWeb is accessed, filtered, and partitioned;
- what “directly tied” and “high confidence” mean operationally;
- whether retrieval uses lexical, semantic, geospatial, or hybrid signals;
- the published schema, split strategy, and acceptance sample.
