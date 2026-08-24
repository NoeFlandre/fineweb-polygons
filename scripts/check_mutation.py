"""Fail when mutmut reports anything that was not killed."""

from __future__ import annotations

import subprocess
import sys

BAD_STATUSES = ("survived", "timeout", "not checked")


def main() -> int:
    result = subprocess.run(
        ["mutmut", "results"],
        check=False,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        print(output, file=sys.stderr, end="\n" if output else "")
        return result.returncode

    failures = [
        line.strip()
        for line in output.splitlines()
        if any(line.rstrip().endswith(status) for status in BAD_STATUSES)
    ]
    if failures:
        print("mutation gate failed:", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1

    print("mutation gate passed: no survivors, timeouts, or unchecked mutants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
