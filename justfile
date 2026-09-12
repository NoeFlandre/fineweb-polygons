set dotenv-load := false
set export

data_root := env_var_or_default("FINEWEB_POLYGONS_DATA_ROOT", "/Volumes/Seagate M3/projects/fineweb-polygons")
UV_CACHE_DIR := data_root + "/cache/uv"
UV_PROJECT_ENVIRONMENT := data_root + "/.venv"
COVERAGE_FILE := data_root + "/.coverage"
COVERAGE_JSON := data_root + "/coverage.json"
TMPDIR := data_root + "/tmp/qa"
PRE_COMMIT_HOME := data_root + "/cache/pre-commit"

default: qa

prepare:
    mkdir -p "{{ TMPDIR }}"

lock-check: prepare
    uv lock --check

sync: prepare
    uv sync --locked

format: prepare
    uv run ruff format src tests scripts

format-check: prepare
    uv run ruff format --check src tests scripts

lint: prepare
    uv run ruff check src tests scripts

typecheck: prepare
    uv run ty check src tests scripts

test: prepare
    uv run pytest

property: prepare
    uv run pytest --no-cov -m property

acceptance: prepare
    uv run pytest --no-cov -m acceptance

architecture: prepare
    uv run pytest --no-cov -m architecture

docs: prepare
    uv run mkdocs build --strict --site-dir "{{ data_root }}/site"

catalog:
    uv run python scripts/build_catalog.py

catalog-check:
    uv run python scripts/build_catalog.py --check

crap: test
    uv run python scripts/check_crap.py --source src --coverage "{{ COVERAGE_JSON }}" --max-crap 6

mutation: prepare
    uv run mutmut run --max-children 1
    uv run python scripts/check_mutation.py

scan shard pbf="{{data_root}}/raw/monaco-latest.osm.pbf":
    uv run fineweb-polygons scan --data-root "{{ data_root }}" --pbf "{{ pbf }}" --shard "{{ shard }}" --run-id v1-10bt-000-v3

scan-v2 shard pbf="{{data_root}}/raw/monaco-latest.osm.pbf":
    uv run fineweb-polygons scan --data-root "{{ data_root }}" --pbf "{{ pbf }}" --shard "{{ shard }}" --run-id v2-10bt-000-v2 --retrieval-version v2

scan-v3 shard pbf="{{data_root}}/raw/monaco-latest.osm.pbf":
    uv run fineweb-polygons scan --data-root "{{ data_root }}" --pbf "{{ pbf }}" --shard "{{ shard }}" --run-id v3-10bt-000-v1 --retrieval-version v3

scan-v4 shard pbf="{{data_root}}/raw/monaco-latest.osm.pbf":
    uv run fineweb-polygons scan --data-root "{{ data_root }}" --pbf "{{ pbf }}" --shard "{{ shard }}" --run-id v4-10bt-000-v1 --retrieval-version v4

scan-v5 shard pbf="{{data_root}}/raw/monaco-latest.osm.pbf" country="Monaco" run_id="v5-monaco-10bt-000-v3":
    uv run fineweb-polygons scan --data-root "{{ data_root }}" --pbf "{{ pbf }}" --shard "{{ shard }}" --run-id "{{ run_id }}" --retrieval-version v5 --country-name "{{ country }}"

scan-v6 shard pbf="{{data_root}}/raw/monaco-latest.osm.pbf" country="Monaco" run_id="v6-monaco-10bt-000-v1":
    uv run fineweb-polygons scan --data-root "{{ data_root }}" --pbf "{{ pbf }}" --shard "{{ shard }}" --run-id "{{ run_id }}" --retrieval-version v6 --country-name "{{ country }}"

lexical-v1 shard="{{data_root}}/raw/fineweb/sample/10BT/000_00000.parquet":
    uv run fineweb-polygons direction2-lexical-v1 --data-root "{{ data_root }}" --shard "{{ shard }}"

lexical-v2 shard="{{data_root}}/raw/fineweb/sample/10BT/000_00000.parquet":
    uv run fineweb-polygons direction2-lexical-v2 --data-root "{{ data_root }}" --shard "{{ shard }}"

package: prepare
    uv build --out-dir "{{ data_root }}/dist"

smoke: prepare
    uv run fineweb-polygons

qa: lock-check format-check lint typecheck catalog-check test property acceptance architecture crap docs package mutation smoke
