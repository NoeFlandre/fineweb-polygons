set dotenv-load := false
set export

data_root := env_var_or_default("FINEWEB_POLYGONS_DATA_ROOT", "/Volumes/Seagate M3/projects/fineweb-polygons")
UV_CACHE_DIR := data_root + "/cache/uv"
UV_PROJECT_ENVIRONMENT := data_root + "/.venv"
COVERAGE_FILE := data_root + "/.coverage"
COVERAGE_JSON := data_root + "/coverage.json"

default: qa

sync:
    uv sync --locked

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

typecheck:
    uv run ty check src tests scripts

test:
    uv run pytest

property:
    uv run pytest --no-cov -m property

acceptance:
    uv run pytest --no-cov -m acceptance

architecture:
    uv run pytest --no-cov -m architecture

docs:
    uv run mkdocs build --strict --site-dir "{{ data_root }}/site"

catalog:
    uv run python scripts/build_catalog.py

catalog-check:
    uv run python scripts/build_catalog.py --check

crap: test
    uv run python scripts/check_crap.py --source src --coverage "{{ COVERAGE_JSON }}" --max-crap 6

mutation:
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

package:
    uv build --out-dir "{{ data_root }}/dist"

smoke:
    uv run fineweb-polygons

qa: format-check lint typecheck catalog-check test property acceptance architecture crap docs package mutation smoke
