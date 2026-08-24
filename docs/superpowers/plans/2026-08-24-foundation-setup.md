# FineWeb Polygons Foundation Setup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish a public-repository-ready, Seagate-backed Python foundation for the future OSM-polygon-to-FineWeb retrieval project without implementing data processing or retrieval decisions.

**Architecture:** Keep source code, tests, configuration, and documentation in the Git checkout. Keep raw OSM input, run manifests, checkpoints, logs, and generated artifacts below `/Volumes/Seagate M3/projects/fineweb-polygons`. Expose only a small `ProjectPaths` module now; defer the polygon extraction, FineWeb access, relevance definition, confidence threshold, and matching strategy to a later design phase.

**Tech Stack:** `uv`, Python 3.12, Ruff, ty, pytest, pytest-cov, mutmut, radon-based CRAP checking, pre-commit, Just, Docker, MkDocs Material, GitHub Actions, and the Hugging Face Hub CLI.

---

### Task 1: Bootstrap repository policy and quality-tool configuration

**Files:**
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `pyproject.toml`
- Create: `ty.toml`
- Create: `.pre-commit-config.yaml`
- Create: `justfile`
- Create: `Dockerfile`
- Create: `.github/workflows/quality.yml`
- Create: `scripts/check_crap.py`

- [ ] **Step 1: Add repository-wide storage and generated-file boundaries**

  Ignore `.venv`, Python caches, coverage/mutation reports, MkDocs output, local `.env` files, macOS metadata, and all local data extensions including `*.osm.pbf`, `*.parquet`, and `*.jsonl`. Document that raw and generated data belong on the Seagate project root.

- [ ] **Step 2: Add `pyproject.toml` with the package and development tools**

  Define the `fineweb-polygons` package under `src/`, require Python `>=3.12,<3.15`, add Ruff, ty, pytest, pytest-cov, mutmut, radon, MkDocs Material, and pre-commit as development dependencies, configure Ruff and pytest, and configure mutmut to mutate only `src/` with one worker.

- [ ] **Step 3: Add runnable quality commands**

  Define Just recipes for `test`, `lint`, `format-check`, `typecheck`, `docs`, `crap`, `mutation`, and an aggregate `qa`. The CRAP recipe runs coverage first and fails when any measured function has a CRAP score greater than or equal to 6. Mutation testing remains an explicit command because it is intentionally more expensive than the normal quality gate.

- [ ] **Step 4: Add container and CI definitions**

  Make the Docker image install the locked project with uv and run the smoke CLI. Make GitHub Actions run the locked dependency sync, Ruff, ty, tests with coverage, CRAP checking, and strict MkDocs validation on supported pushes and pull requests.

- [ ] **Step 5: Generate and inspect the uv lockfile**

  Run `uv lock` and inspect the resulting lockfile for the declared project and development dependencies. Do not add raw data or a local virtual environment to Git.

### Task 2: Establish the Seagate-backed path contract with TDD

**Files:**
- Create: `tests/test_foundation.py`
- Create: `src/fineweb_polygons/__init__.py`
- Create: `src/fineweb_polygons/foundation.py`

- [ ] **Step 1: Write the failing default-root test**

  Assert that `ProjectPaths.from_environment(repository_root)` uses `/Volumes/Seagate M3/projects/fineweb-polygons` when no override is present.

- [ ] **Step 2: Run the focused test and verify the expected RED failure**

  Run `uv run pytest tests/test_foundation.py::test_default_data_root_is_on_seagate -q`. It must fail because `ProjectPaths` does not yet exist.

- [ ] **Step 3: Write the failing override-and-layout tests**

  Assert that `FINEWEB_POLYGONS_DATA_ROOT` overrides the default and that `ensure_data_layout()` creates only `raw`, `runs`, `logs`, and `artifacts` below the selected external root.

- [ ] **Step 4: Implement the minimal path module**

  Add a frozen dataclass with `repository_root`, `data_root`, `raw_dir`, `runs_dir`, `logs_dir`, `artifacts_dir`, a mapping-aware environment constructor for testability, and an idempotent layout initializer. Do not add pipeline, matching, checkpoint, or network behavior.

- [ ] **Step 5: Run the focused tests and verify GREEN**

  Run `uv run pytest tests/test_foundation.py -q`, then refactor only if all focused tests remain green.

### Task 3: Add public project and dataset metadata

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `CITATION.cff`
- Create: `mkdocs.yml`
- Create: `docs/index.md`
- Create: `docs/architecture/foundation.md`
- Create: `docs/development.md`

- [ ] **Step 1: Document the scope and deferred decisions**

  Explain the future goal, Monaco-only starting input, Seagate data boundary, resumability/logging/reproducibility requirements, deep-module intent, and the explicit decision not to implement retrieval or relevance logic in this foundation.

- [ ] **Step 2: Add the full ODC-By 1.0 license and citation metadata**

  Use the Open Data Commons Attribution 1.0 text, identify the dataset as `odc-by` in the Hugging Face card front matter, and provide a valid `CITATION.cff` for the public GitHub software/data project.

- [ ] **Step 3: Add strict MkDocs Material configuration**

  Configure a minimal documentation site with navigation for overview, foundation architecture, and development commands. `mkdocs build --strict` must work from a clean locked environment.

### Task 4: Prepare the public remotes without publishing data

**Files:**
- No raw-data files are added to Git or Hugging Face.

- [ ] **Step 1: Verify the Seagate source and local exclusion**

  Confirm the relocated raw file exists at `/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf`, the local checkout contains no PBF, and the SHA-256 remains `c51ff78facdd222d77a15172471e7c5c77995bc78d9dc0d2f6017287dc0eb188`.

- [ ] **Step 2: Create the public Hugging Face dataset repository**

  Create `NoeFlandre/fineweb-polygons` with type `dataset` and without `--private`, then upload only the dataset card, license, and citation metadata.

- [ ] **Step 3: Create and push the public GitHub repository**

  Create `NoeFlandre/fineweb-polygons` with `gh repo create --public`, push `main`, and verify the remote is public. If GitHub authentication remains invalid, stop at the exact login boundary and report it without weakening the public visibility requirement.

### Task 5: Run the full foundation verification and hand off

**Files:**
- Modify only files required by verification failures.

- [ ] **Step 1: Run the complete local quality gate**

  Run `just qa` and inspect every command's exit status.

- [ ] **Step 2: Run mutation testing serially**

  Run `just mutation` with one worker and report the exact mutation result; do not treat an unrun or interrupted mutation report as evidence.

- [ ] **Step 3: Verify Git and remote state**

  Check `git status --short --branch`, `git log --oneline`, GitHub visibility, Hugging Face dataset visibility, and the absence of raw-data artifacts in both repositories.

- [ ] **Step 4: Commit the verified foundation**

  Use a Conventional Commit such as `chore: bootstrap fineweb polygons foundation` only after the local verification commands have produced fresh evidence.
