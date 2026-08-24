"""Fail when a measured function reaches the project's CRAP threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radon.complexity import cc_visit


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--max-crap", type=float, required=True)
    return parser.parse_args()


def _coverage_by_file(path: Path) -> dict[str, dict[str, Any]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload["files"]


def _coverage_entry(
    filename: str, source_root: Path, coverage: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    candidates = [filename, str(Path(filename).resolve())]
    source_path = Path(filename)
    if not source_path.is_absolute():
        candidates.append(str((Path.cwd() / source_path).resolve()))
        candidates.append(str((source_root / source_path.name).resolve()))
    for candidate in candidates:
        if candidate in coverage:
            return coverage[candidate]
    return {}


def _blocks(source: str) -> list[Any]:
    unique: dict[tuple[int, int, str], Any] = {}
    for block in cc_visit(source):
        methods = getattr(block, "methods", None)
        for candidate in methods if methods is not None else [block]:
            key = (candidate.lineno, candidate.endline, candidate.fullname)
            unique[key] = candidate
    return sorted(unique.values(), key=lambda block: block.lineno)


def _crap_score(complexity: int, coverage_percent: float) -> float:
    uncovered = 1.0 - (coverage_percent / 100.0)
    return complexity**2 * uncovered**3 + complexity


def _report(
    source_root: Path, coverage: dict[str, dict[str, Any]]
) -> list[tuple[str, float]]:
    reports: list[tuple[str, float]] = []
    for path in sorted(source_root.rglob("*.py")):
        entry = _coverage_entry(str(path), source_root, coverage)
        functions = entry.get("functions", {})
        source = path.read_text(encoding="utf-8")
        for block in _blocks(source):
            function = functions.get(block.fullname, {})
            covered = float(function.get("summary", {}).get("percent_covered", 0.0))
            score = _crap_score(block.complexity, covered)
            reports.append((f"{path}:{block.lineno} {block.fullname}", score))
    return reports


def main() -> int:
    args = _arguments()
    reports = _report(args.source, _coverage_by_file(args.coverage))
    failures = 0
    for name, score in reports:
        status = "FAIL" if score >= args.max_crap else "PASS"
        print(f"{status} {name}: CRAP={score:.2f}")
        failures += score >= args.max_crap
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
