"""Regenerate every catalog artifact from the registry.

Run `just catalog` after adding or changing a version in
`fineweb_polygons.registry`. `just qa` fails if the committed files do not
match what this script would write.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from fineweb_polygons.catalog import (
    build_catalog,
    build_catalog_page,
    build_direction_record,
    build_hf_configs,
    build_readme_configs,
)
from fineweb_polygons.registry import DIRECTIONS

CATALOG_DATE = "2026-09-07"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _render(root: Path) -> dict[Path, str]:
    """Return every generated file and its exact expected content."""
    generated = {
        root / "metadata" / "catalog.json": json.dumps(
            build_catalog(CATALOG_DATE), indent=2, ensure_ascii=False
        )
        + "\n"
    }
    for direction in DIRECTIONS:
        path = root / "metadata" / "directions" / f"{direction.id}.json"
        generated[path] = (
            json.dumps(
                build_direction_record(direction, CATALOG_DATE),
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
    generated[root / "metadata" / "huggingface-configs.json"] = (
        json.dumps(build_hf_configs(), indent=2, ensure_ascii=False) + "\n"
    )
    generated[root / "docs" / "dataset-catalog.md"] = build_catalog_page()
    generated[root / "README.md"] = _readme_with_configs(root / "README.md")
    return generated


def _readme_with_configs(path: Path) -> str:
    """Return the dataset card with its `configs:` block regenerated."""
    text = path.read_text(encoding="utf-8")
    start = text.index("configs:\n")
    end = text.index("---\n", start)
    return text[:start] + build_readme_configs() + text[end:]


def main(argv: list[str] | None = None) -> int:
    """Write or verify the generated catalog artifacts."""
    parser = argparse.ArgumentParser(prog="build_catalog")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail instead of writing when a file is out of date",
    )
    parser.add_argument("--root", type=Path, default=_REPOSITORY_ROOT)
    parsed = parser.parse_args(argv)

    stale: list[Path] = []
    for path, content in _render(parsed.root).items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        if parsed.check:
            stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(parsed.root)}")
    if stale:
        for path in stale:
            print(f"stale: {path.relative_to(parsed.root)}", file=sys.stderr)
        print("run `just catalog` to regenerate", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
