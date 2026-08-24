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
    uv run mkdocs build --strict

crap:
    uv run pytest
    uv run python scripts/check_crap.py --source src --coverage coverage.json --max-crap 6

mutation:
    uv run mutmut run --max-children 1

qa: format-check lint typecheck test crap docs
