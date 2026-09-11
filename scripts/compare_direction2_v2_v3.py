"""Create a deterministic V2-versus-V3 report from Direction 2 Parquet files."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pyarrow.parquet as pq

_KEY_COLUMNS = ("polygon_id", "matched_alias", "fineweb_url", "sentence")
_V3_COLUMNS = (
    *_KEY_COLUMNS,
    "polygon_name",
    "name_match_class",
    "decision_tier",
    "evidence_score",
    "evidence_reasons",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-dir", type=Path, required=True)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_v2(path: Path) -> list[tuple[object, ...]]:
    return [
        tuple(row[column] for column in _KEY_COLUMNS)
        for row in pq.read_table(path, columns=list(_KEY_COLUMNS)).to_pylist()
    ]


def _read_v3(path: Path) -> list[dict[str, object]]:
    return [
        {column: row[column] for column in _V3_COLUMNS}
        for row in pq.read_table(path, columns=list(_V3_COLUMNS)).to_pylist()
    ]


def _counts(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    tiers = Counter(str(row["decision_tier"]) for row in rows)
    classes = Counter(str(row["name_match_class"]) for row in rows)
    names = Counter(str(row["polygon_name"]) for row in rows)
    high = [row for row in rows if row["decision_tier"] == "high_confidence"]
    high_names = Counter(str(row["polygon_name"]) for row in high)
    high_reasons = Counter(str(row["evidence_reasons"]) for row in high)
    generic_high = sum(row["name_match_class"] == "generic_name" for row in high)
    return {
        "rows": len(rows),
        "unique_polygons": len({row["polygon_id"] for row in rows}),
        "duplicate_rows": len(rows) - len({_row_key(row) for row in rows}),
        "tiers": tiers,
        "classes": classes,
        "top_names": names.most_common(10),
        "high_names": high_names.most_common(10),
        "high_reasons": high_reasons.most_common(8),
        "high_generic_rows": generic_high,
        "high_rows": len(high),
        "sample": _sample(high),
    }


def _row_key(row: dict[str, object]) -> tuple[object, ...]:
    return tuple(row[column] for column in _KEY_COLUMNS)


def _sample(
    rows: Sequence[dict[str, object]], limit: int = 5
) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -int(row["evidence_score"]),
            str(row["polygon_name"]),
            str(row["fineweb_url"]),
            str(row["sentence"]),
        ),
    )
    return [
        {
            "name": row["polygon_name"],
            "url": row["fineweb_url"],
            "sentence": _shorten(str(row["sentence"])),
            "score": row["evidence_score"],
            "reasons": row["evidence_reasons"],
        }
        for row in ordered[:limit]
    ]


def _shorten(value: str, limit: int = 240) -> str:
    return value if len(value) <= limit else value[: limit - 1].rstrip() + "…"


def _format_counter(value: Counter[str]) -> str:
    return ", ".join(f"{key}: {count}" for key, count in value.most_common())


def _format_pairs(value: list[tuple[str, int]]) -> str:
    return ", ".join(f"{name}: {count}" for name, count in value)


def _render(
    summaries: dict[str, dict[str, object]],
    *,
    multiset_equal: bool,
) -> str:
    total_rows = sum(int(summary["rows"]) for summary in summaries.values())
    total_high = sum(int(summary["high_rows"]) for summary in summaries.values())
    total_generic_high = sum(
        int(summary["high_generic_rows"]) for summary in summaries.values()
    )
    lines = [
        "# Direction 2 V2 → V3 comparison",
        "",
        "Generated deterministically from the local Parquet artifacts.",
        "",
        "## Recall contract",
        "",
        f"- V3 and V2 candidate occurrence multisets equal: `{multiset_equal}`",
        "- V3 keeps every V2 occurrence and adds evidence fields and tiers.",
        "",
        "## Counts",
        "",
        "| Source | Rows | Unique polygons | High confidence | Possible | Rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source, summary in summaries.items():
        tiers = summary["tiers"]
        assert isinstance(tiers, Counter)
        lines.append(
            f"| {source} | {summary['rows']} | {summary['unique_polygons']} | "
            f"{tiers['high_confidence']} | {tiers['possible']} | "
            f"{tiers['rejected']} |"
        )
    lines.extend(
        [
            "",
            f"Across both sources V3 writes {total_rows} rows; "
            f"{total_high} ({100 * total_high / total_rows:.2f}%) are high confidence. "
            f"{total_generic_high} high-confidence rows still use a generic name.",
            "",
            "## Qualitative assessment",
            "",
            "V3 is better than V2 as a ranking and filtering layer because it "
            "preserves recall while exposing a deterministic confidence tier. "
            "It is not yet a clean relevance set: URL matches and nearby same-source "
            "names can promote generic names, so high confidence still needs review "
            "or a later geographic/entity disambiguation step.",
            "",
        ]
    )
    for source, summary in summaries.items():
        lines.extend(
            [
                f"## {source}",
                "",
                f"- V2/V3 top names: {_format_pairs(summary['top_names'])}",
                f"- V3 high-confidence names: {_format_pairs(summary['high_names'])}",
                f"- High-confidence evidence patterns: "
                f"{_format_pairs(summary['high_reasons'])}",
                f"- Name classes: {_format_counter(summary['classes'])}",
                "- Duplicate rows retained for compatibility: "
                f"{summary['duplicate_rows']}",
                "",
                "Stable high-confidence examples:",
                "",
            ]
        )
        for example in summary["sample"]:
            lines.extend(
                [
                    f"- `{example['name']}` (score {example['score']}; "
                    f"{example['reasons']}) — {example['sentence']}",
                    f"  URL: {example['url']}",
                ]
            )
        lines.append("")
    return "\n".join(lines)


def _compare(v2_dir: Path, v3_dir: Path) -> tuple[dict[str, dict[str, object]], bool]:
    summaries: dict[str, dict[str, object]] = {}
    multiset_equal = True
    for source in ("monaco", "liechtenstein"):
        v2_rows = _read_v2(v2_dir / f"{source}.parquet")
        v3_rows = _read_v3(v3_dir / f"{source}.parquet")
        multiset_equal = multiset_equal and Counter(v2_rows) == Counter(
            _row_key(row) for row in v3_rows
        )
        summaries[source] = _counts(v3_rows)
    return summaries, multiset_equal


def main() -> int:
    arguments = _arguments()
    summaries, multiset_equal = _compare(arguments.v2_dir, arguments.v3_dir)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        _render(summaries, multiset_equal=multiset_equal), encoding="utf-8"
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
