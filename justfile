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

crap:
    uv run pytest
    uv run python scripts/check_crap.py --source src --coverage "/Volumes/Seagate M3/projects/fineweb-polygons/coverage.json" --max-crap 6

mutation:
    uv run mutmut run --max-children 1
    uv run python scripts/check_mutation.py

scan shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v1-10bt-000-v2

scan-v2 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v2-10bt-000-v1 --retrieval-version v2

scan-v3 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache-v3" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v3" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v3-10bt-000-v1 --retrieval-version v3

scan-v4 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache-v4" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v4" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id v4-10bt-000-v1 --retrieval-version v4

scan-v5 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" country="Monaco" run_id="v5-monaco-10bt-000-v3":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache-v5" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v5" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id "{{run_id}}" --retrieval-version v5 --country-name "{{country}}"

scan-v6 shard pbf="/Volumes/Seagate M3/projects/fineweb-polygons/raw/monaco-latest.osm.pbf" country="Monaco" run_id="v6-monaco-10bt-000-v1":
    UV_CACHE_DIR="/Volumes/Seagate M3/projects/fineweb-polygons/.uv-cache-v6" UV_PROJECT_ENVIRONMENT="/Volumes/Seagate M3/projects/fineweb-polygons/.venvs/fineweb-polygons-v5" uv run fineweb-polygons scan --pbf "{{pbf}}" --shard "{{shard}}" --run-id "{{run_id}}" --retrieval-version v6 --country-name "{{country}}"

qa: format-check lint typecheck test crap docs
