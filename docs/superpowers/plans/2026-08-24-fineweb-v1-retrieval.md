# FineWeb V1 Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scan one FineWeb 10BT Parquet shard for high-confidence Monaco polygon matches using exact case-insensitive normalized names and Monaco context in either `text` or `url`.

**Architecture:** Read named closed OSM ways and named multipolygon/boundary relations into immutable polygon profiles. Build one Aho–Corasick matcher for all normalized names and one for the two context phrases, then stream Parquet row groups in bounded batches. Store deterministic per-row-group JSONL checkpoints, a manifest, structured logs, and a merged result only on the external Seagate data root.

**Tech Stack:** Python 3.12, uv, PyArrow, PyOsmium, pyahocorasick, pytest, Ruff, ty, MkDocs, CRAP, and mutmut.

---

### Task 1: Add runtime dependencies and normalization contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` via `uv lock`
- Create: `src/fineweb_polygons/models.py`
- Create: `src/fineweb_polygons/normalization.py`
- Create: `tests/test_normalization.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Add the runtime dependencies.**

Add these entries to `[project].dependencies`:

```toml
dependencies = [
    "pyarrow>=18.0",
    "pyosmium>=4.0",
    "pyahocorasick>=2.1",
]
```

- [ ] **Step 2: Write the failing normalization tests.**

Create focused tests for Unicode compatibility normalization, case folding,
separator normalization, and empty values:

```python
from fineweb_polygons.normalization import normalize_for_search


def test_normalization_ignores_case_and_repeated_separators() -> None:
    assert normalize_for_search("  Stade  Louis-II\n") == "stade louis ii"


def test_normalization_decodes_url_escapes() -> None:
    assert normalize_for_search("https://example.test/Palais%20Monaco") == (
        "https example test palais monaco"
    )


def test_normalization_returns_empty_for_none() -> None:
    assert normalize_for_search(None) == ""
```

Add model construction coverage:

```python
from fineweb_polygons.models import PolygonProfile


def test_polygon_profile_retains_original_and_normalized_name() -> None:
    profile = PolygonProfile.create("way/7", "Palais  MONACO")

    assert profile.polygon_id == "way/7"
    assert profile.name == "Palais  MONACO"
    assert profile.normalized_name == "palais monaco"
```

- [ ] **Step 3: Run the focused tests and verify RED.**

Run:

```bash
uv run pytest tests/test_normalization.py tests/test_models.py -q
```

Expected: collection fails because the new modules and `PolygonProfile` do not
exist yet.

- [ ] **Step 4: Implement the minimal contracts.**

`normalization.py` should expose `NORMALIZATION_VERSION` and:

```python
def normalize_for_search(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = unquote(text)
    return " ".join(_SEPARATOR_RE.split(text)).strip()
```

Use `re.compile(r"[\W_]+", re.UNICODE)` for `_SEPARATOR_RE`. `models.py` should
define frozen, slotted dataclasses for `PolygonProfile`, `FineWebDocument`, and
`MatchEvidence`; `PolygonProfile.create()` must call the shared normalizer.

- [ ] **Step 5: Run the focused tests and verify GREEN.**

Run:

```bash
uv run pytest tests/test_normalization.py tests/test_models.py -q
```

Expected: all focused tests pass.

- [ ] **Step 6: Lock the dependency graph.**

Run:

```bash
uv lock
```

Expected: `uv.lock` records the three runtime packages and their transitive
dependencies without adding data files to the repository.

- [ ] **Step 7: Commit the contract.**

```bash
git add pyproject.toml uv.lock src/fineweb_polygons/models.py src/fineweb_polygons/normalization.py tests/test_models.py tests/test_normalization.py
git commit -m "feat: add V1 data and normalization contracts"
```

### Task 2: Read named Monaco polygon profiles

**Files:**
- Create: `src/fineweb_polygons/polygons.py`
- Create: `tests/test_polygons.py`

- [ ] **Step 1: Write the failing parser test.**

Create a temporary OSM XML fixture containing one named closed way, one named
multipolygon relation, one unnamed closed way, and one open named way. Assert
that only the two polygon records are returned and that identifiers distinguish
ways from relations:

```python
def test_read_named_polygon_profiles_keeps_only_named_polygon_entities(
    tmp_path: Path,
) -> None:
    pbf = tmp_path / "mini.osm"
    pbf.write_text(MINI_OSM_XML, encoding="utf-8")

    result = read_named_polygon_profiles(pbf)

    assert [profile.polygon_id for profile in result.profiles] == [
        "relation/20",
        "way/10",
    ]
    assert result.named_count == 2
    assert result.unnamed_count == 1
```

- [ ] **Step 2: Run the parser test and verify RED.**

```bash
uv run pytest tests/test_polygons.py::test_read_named_polygon_profiles_keeps_only_named_polygon_entities -q
```

Expected: collection fails because `polygons.py` does not exist.

- [ ] **Step 3: Implement the minimal PyOsmium reader.**

Define `PolygonReadResult` with `profiles`, `named_count`, and `unnamed_count`.
The handler must count closed ways and relations whose `type` is
`multipolygon` or `boundary`; it must emit only records with a non-empty
`name`. Use stable IDs in the form `way/{entity.id}` and `relation/{entity.id}`.
Sort profiles by `polygon_id` before returning them so runs are deterministic.

- [ ] **Step 4: Run the parser test and verify GREEN.**

```bash
uv run pytest tests/test_polygons.py -q
```

Expected: all parser tests pass.

- [ ] **Step 5: Commit the polygon reader.**

```bash
git add src/fineweb_polygons/polygons.py tests/test_polygons.py
git commit -m "feat: read named OSM polygon profiles"
```

### Task 3: Implement exact multi-pattern matching

**Files:**
- Create: `src/fineweb_polygons/matching.py`
- Create: `tests/test_matching.py`

- [ ] **Step 1: Write the failing matching tests.**

Cover the two independent fields and the high-confidence rule:

```python
def test_name_in_url_and_context_in_text_is_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Casino de Monaco")])
    document = FineWebDocument(
        row_index=4,
        document_id="doc-4",
        text="A report from Monaco describes the venue.",
        url="https://example.test/casino-de-monaco",
    )

    matches = matcher.match(document)

    assert len(matches) == 1
    assert matches[0].matched_fields == ("url",)
    assert matches[0].context_fields == ("text",)


def test_name_in_text_and_context_in_url_is_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(
        row_index=5,
        document_id=None,
        text="Fontvieille has a new report.",
        url="https://monaco.example.test/report",
    )

    assert matcher.match(document)[0].polygon_id == "way/1"


def test_name_without_context_is_not_high_confidence() -> None:
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])
    document = FineWebDocument(6, "doc-6", "Fontvieille has a report.", "")

    assert matcher.match(document) == ()
```

- [ ] **Step 2: Run the matching tests and verify RED.**

```bash
uv run pytest tests/test_matching.py -q
```

Expected: collection fails because `matching.py` is absent.

- [ ] **Step 3: Implement the Aho–Corasick matcher.**

Build one automaton from unique padded normalized names and one from the two
padded context phrases. Scan normalized `text` and URL values separately. Emit
a `MatchEvidence` record only when a name is found in at least one field and a
context phrase is found in at least one field. Map duplicate normalized names
back to every matching polygon profile. Record sorted field tuples and short
original-field excerpts.

- [ ] **Step 4: Run the matching tests and verify GREEN.**

```bash
uv run pytest tests/test_matching.py -q
```

Expected: all matching tests pass.

- [ ] **Step 5: Commit the matcher.**

```bash
git add src/fineweb_polygons/matching.py tests/test_matching.py
git commit -m "feat: add exact text and URL evidence matching"
```

### Task 4: Stream Parquet row groups with atomic partition outputs

**Files:**
- Create: `src/fineweb_polygons/scanning.py`
- Create: `tests/test_scanning.py`

- [ ] **Step 1: Write the failing streaming tests.**

Create a tiny Parquet file with `id`, `text`, and `url` columns. Test that the
scanner projects only those columns, uses global row indexes, writes JSONL, and
replaces a temporary output atomically:

```python
def test_scan_row_group_writes_matching_evidence(tmp_path: Path) -> None:
    shard = write_fixture_shard(tmp_path / "shard.parquet")
    output = tmp_path / "partition.jsonl"
    matcher = EvidenceMatcher([PolygonProfile.create("way/1", "Fontvieille")])

    stats = scan_row_group(shard, row_group_index=0, matcher=matcher, output_path=output)

    assert stats.rows_scanned == 2
    assert stats.matches_written == 1
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record["fineweb_row_index"] == 1
    assert record["polygon_id"] == "way/1"
```

Add a schema test that raises `ValueError` when `text` or `url` is missing.

- [ ] **Step 2: Run the scanning tests and verify RED.**

```bash
uv run pytest tests/test_scanning.py -q
```

Expected: collection fails because `scanning.py` is absent.

- [ ] **Step 3: Implement bounded row-group scanning.**

Use `pyarrow.parquet.ParquetFile`. Validate `text` and `url`, include `id` when
available, and iterate with `iter_batches(batch_size=8192, row_groups=[index])`.
Track the row-group start from Parquet metadata. Convert only the projected
columns needed for each batch, pass `FineWebDocument` values to the matcher,
and write sorted JSON objects to `output_path.with_suffix(".tmp")` before
calling `Path.replace()`.

- [ ] **Step 4: Run the scanning tests and verify GREEN.**

```bash
uv run pytest tests/test_scanning.py -q
```

Expected: all scanning tests pass.

- [ ] **Step 5: Commit the scanner.**

```bash
git add src/fineweb_polygons/scanning.py tests/test_scanning.py
git commit -m "feat: stream FineWeb row groups with checkpoints"
```

### Task 5: Add resumable runs, manifests, and structured logs

**Files:**
- Create: `src/fineweb_polygons/runs.py`
- Create: `tests/test_runs.py`
- Modify: `src/fineweb_polygons/foundation.py`

- [ ] **Step 1: Write the failing run-resumption test.**

Use two Parquet row groups and a supplied profile tuple. Execute the same run
twice. The second execution must reuse completed partition files, preserve the
same result bytes, and report skipped partitions:

```python
def test_run_resumes_completed_row_groups(tmp_path: Path) -> None:
    paths = ProjectPaths.from_environment(
        tmp_path / "repo",
        environ={DATA_ROOT_ENVIRONMENT_VARIABLE: str(tmp_path / "external")},
    )
    config = ScanRunConfig(paths=paths, pbf_path=pbf, shard_path=shard, run_id="case")

    first = execute_run(config, profiles=profiles)
    first_bytes = first.result_path.read_bytes()
    second = execute_run(config, profiles=profiles)

    assert second.partitions_skipped == first.partitions_completed
    assert second.result_path.read_bytes() == first_bytes
```

Add tests for changed input fingerprints and repository-local data roots being
rejected.

- [ ] **Step 2: Run the run tests and verify RED.**

```bash
uv run pytest tests/test_runs.py -q
```

Expected: collection fails because `runs.py` has not been created.

- [ ] **Step 3: Implement the run coordinator.**

Create `ScanRunConfig`, `RunSummary`, and a deterministic `RunLayout`. Compute
SHA-256 fingerprints for the PBF and shard, hash the sorted configuration, and
write `manifest.json` atomically. Use one partition per Parquet row group. A
partition is complete only when its JSONL file has been atomically replaced and
the manifest has been updated. Log JSON records such as:

```json
{"event":"partition_complete","partition":0,"rows_scanned":2,"matches":1}
```

On restart, require matching input/configuration fingerprints, skip complete
partitions, merge partition files in row-group order, and atomically replace the
final result. Reject paths outside `ProjectPaths.data_root` and reject a data
root equal to or below the repository root. Keep all run files under `runs/`,
`logs/`, and `artifacts/` on the configured external root.

- [ ] **Step 4: Run the run tests and verify GREEN.**

```bash
uv run pytest tests/test_runs.py -q
```

Expected: all run and resumption tests pass.

- [ ] **Step 5: Commit the run coordinator.**

```bash
git add src/fineweb_polygons/foundation.py src/fineweb_polygons/runs.py tests/test_runs.py
git commit -m "feat: add resumable FineWeb scan runs"
```

### Task 6: Expose the scan command and document the external-data workflow

**Files:**
- Modify: `src/fineweb_polygons/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`
- Modify: `docs/development.md`
- Modify: `docs/index.md`
- Modify: `justfile`

- [ ] **Step 1: Write the failing CLI tests.**

Test that an empty invocation retains the foundation message and that the scan
parser passes explicit paths/configuration to the run coordinator. Keep the
coordinator injection point as a small callable so the CLI test does not scan a
real shard.

- [ ] **Step 2: Run the CLI tests and verify RED.**

```bash
uv run pytest tests/test_cli.py -q
```

Expected: the new scan invocation test fails because the subcommand is absent.

- [ ] **Step 3: Implement the CLI and docs.**

Add:

```bash
uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/10BT/000_00000.parquet" \
  --run-id v1-10bt-000
```

The command defaults to the Seagate data root, `8192` rows per batch, and a
stable run ID. It prints a compact JSON summary and returns non-zero on missing
inputs or fingerprint conflicts. Document the official first-shard download:

```bash
export HF_HOME="/Volumes/Seagate M3/projects/fineweb-polygons/.hf"
hf download HuggingFaceFW/fineweb \
  --repo-type dataset \
  --include "sample/10BT/000_00000.parquet" \
  --local-dir "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/10BT"
```

Document the output locations and the exact V1 rule. Do not add data paths to
the repository or upload the shard/results to the public Hugging Face dataset.

- [ ] **Step 4: Run the CLI tests and verify GREEN.**

```bash
uv run pytest tests/test_cli.py -q
```

Expected: all CLI tests pass.

- [ ] **Step 5: Commit the user-facing command.**

```bash
git add src/fineweb_polygons/cli.py tests/test_cli.py README.md docs/development.md docs/index.md justfile
git commit -m "feat: expose resumable FineWeb V1 scan"
```

### Task 7: Run the real one-shard acceptance test and quality gates

**Files:**
- Modify: `docs/superpowers/plans/2026-08-24-fineweb-v1-retrieval.md`

- [ ] **Step 1: Download only the first 10BT shard to Seagate storage.**

Use the external Hugging Face cache and local directory shown in Task 6. Verify
the expected path exists and compute its SHA-256 without copying it into the
repository.

- [ ] **Step 2: Run the real resumable scan.**

```bash
uv run fineweb-polygons scan \
  --pbf "/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" \
  --shard "/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/10BT/000_00000.parquet" \
  --run-id v1-10bt-000
```

Run the same command again and verify the JSON summary reports skipped
completed partitions and that the final artifact checksum is unchanged.

- [ ] **Step 3: Run the complete local quality gate.**

```bash
just qa
just mutation
```

Expected: Ruff, ty, tests, CRAP below 6, strict MkDocs, and serialized mutation
testing pass. If Docker Desktop is available, also run:

```bash
docker build -t fineweb-polygons:v1 .
```

- [ ] **Step 4: Inspect repository and external-data boundaries.**

```bash
git status --short --branch
find /Users/noeflandre/fineweb-polygons -maxdepth 2 -type f \( -name '*.parquet' -o -name '*.osm.pbf' -o -name '*.jsonl' \)
```

Expected: no raw or generated data files appear in the repository checkout;
run artifacts exist only under `/Volumes/Seagate M3/projects/fineweb-polygons`.

- [ ] **Step 5: Commit the verified V1 release.**

```bash
git add docs/superpowers/plans/2026-08-24-fineweb-v1-retrieval.md
git commit -m "docs: record FineWeb V1 implementation verification"
```
