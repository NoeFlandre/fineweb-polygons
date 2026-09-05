set dotenv-load := false

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
    uv run ty check src tests

test:
    uv run pytest

docs:
    uv run mkdocs build --strict --site-dir "/Volumes/Seagate M3/projects/fineweb-polygons/site"

crap: test
    uv run python scripts/check_crap.py --source src --coverage "/Volumes/Seagate M3/projects/fineweb-polygons/coverage.json" --max-crap 6

mutation:
    uv run mutmut run --max-children 1
    uv run python scripts/check_mutation.py

scan shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v1-10bt-000-v3

scan-v2 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v2-10bt-000-v2 --retrieval-version v2

scan-v3 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v3-10bt-000-v1 --retrieval-version v3

scan-v4 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v4-10bt-000-v1 --retrieval-version v4

scan-v5 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" country="Monaco" run_id="v5-monaco-10bt-000-v3":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id "{{run_id}}" --retrieval-version v5 --country-name "{{country}}"

scan-v6 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" country="Monaco" run_id="v6-monaco-10bt-000-v1":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-cleanup" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v8" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id "{{run_id}}" --retrieval-version v6 --country-name "{{country}}"

direction2 shard="/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-direction2" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-direction2" uv run fineweb-polygons direction2-lexical-v1 --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" --shard "{{shard}}"

direction2-v2 shard="/Volumes/Seagate M3/projects/fineweb-polygons/raw/fineweb/sample/10BT/000_00000.parquet":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/cache/uv-direction2-v2" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-direction2" uv run fineweb-polygons direction2-lexical-v2 --data-root "/Volumes/Seagate M3/projects/fineweb-polygons" --shard "{{shard}}"

qa: format-check lint typecheck test crap docs
