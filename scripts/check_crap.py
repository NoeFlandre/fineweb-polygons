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


def _coverage_by_file(path: Path) -> dict[str, set[int]]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {
        filename: set(details["executed_lines"])
        for filename, details in payload["files"].items()
    }


def _coverage_entry(
    filename: str, source_root: Path, coverage: dict[str, set[int]]
) -> set[int]:
    candidates = [filename, str(Path(filename).resolve())]
    source_path = Path(filename)
    if not source_path.is_absolute():
        candidates.append(str((Path.cwd() / source_path).resolve()))
        candidates.append(str((source_root / source_path.name).resolve()))
    for candidate in candidates:
        if candidate in coverage:
            return coverage[candidate]
    return set()


def _blocks(source: str) -> list[Any]:
    unique: dict[tuple[int, int, str], Any] = {}
    for block in cc_visit(source):
        candidates = [block, *getattr(block, "methods", [])]
        for candidate in candidates:
            key = (candidate.lineno, candidate.endline, candidate.name)
            unique[key] = candidate
    return list(unique.values())


def _crap_score(complexity: int, coverage_percent: float) -> float:
    uncovered = 1.0 - (coverage_percent / 100.0)
    return complexity**2 * uncovered**3 + complexity


def _report(
    source_root: Path, coverage: dict[str, set[int]]
) -> list[tuple[str, float]]:
    reports: list[tuple[str, float]] = []
    for path in sorted(source_root.rglob("*.py")):
        executed = _coverage_entry(str(path), source_root, coverage)
        source = path.read_text(encoding="utf-8")
        for block in _blocks(source):
            lines = set(range(block.lineno, block.endline + 1))
            covered = 100.0 * len(lines & executed) / len(lines)
            score = _crap_score(block.complexity, covered)
            reports.append((f"{path}:{block.lineno} {block.name}", score))
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
