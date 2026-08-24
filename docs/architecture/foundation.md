# Foundation architecture

The foundation keeps the V1 lexical baseline safe to run while leaving later retrieval methods open.

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

## V1 resumability contract

Each run has a stable identifier and writes a manifest before processing. The manifest records raw-input checksums, configuration, polygon-profile fingerprints, output schema version, and deterministic chunk identities. A chunk covers up to 32 contiguous Parquet row groups, writes its JSONL output atomically, and can be skipped on restart after its checkpoint is complete.

The result is merged atomically only after all chunks have completed. Raw inputs, checkpoints, logs, and results remain below the configured Seagate data root.

## Future logging contract

Each run must have a dedicated log path on the Seagate volume. Logs should include timestamps, severity, run ID, work-unit identity, input reference, event name, and enough exception context to diagnose a failed restart. Human-readable console output may be derived from the same events; it must not be the only record.

## Deferred decisions

The following remain intentionally open:

- how OSM polygons are normalized and identified;
- how FineWeb is accessed and scaled beyond the first shard;
- whether the V1 exact lexical rule is sufficient for “directly tied” and “high confidence”;
- whether retrieval uses lexical, semantic, geospatial, or hybrid signals;
- the published schema, split strategy, and acceptance sample.
