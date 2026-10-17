#!/usr/bin/env python3
"""Fail the gate for a test whose call stays over the limit when run alone."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HEADER = re.compile(r"^=+ slowest(?: \d+)? durations =+$")
PHASE = re.compile(r"^(\d+(?:\.\d+)?)s (setup|call|teardown)\s+(\S.*?)\s*$")


def phases(text: str) -> list[tuple[float, str, str]]:
    found: list[tuple[float, str, str]] = []
    inside = False
    for line in text.splitlines():
        if HEADER.match(line):
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("="):
            break
        match = PHASE.match(line)
        if match:
            found.append((float(match.group(1)), match.group(2), match.group(3)))
    return found


def has_durations(text: str) -> bool:
    return any(HEADER.match(line) for line in text.splitlines())


def alone(nodeid: str) -> float | None:
    env = {key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-q", "--durations=0", "--durations-min=0"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    calls = [
        seconds
        for seconds, when, node in phases(result.stdout)
        if when == "call" and node == nodeid
    ]
    if result.returncode != 0 or not calls:
        return None
    return calls[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--limit", type=float, default=2.0)
    parser.add_argument("--list-from", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.log, encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    if not has_durations(text):
        print("the package suite printed no durations section, so the call limit was not checked")
        return 1

    rows = phases(text)
    shared = sorted(
        ((seconds, when, node) for seconds, when, node in rows
         if when != "call" and seconds >= args.list_from),
        reverse=True,
    )
    over = sorted(
        ((seconds, node) for seconds, when, node in rows
         if when == "call" and seconds > args.limit),
        reverse=True,
    )

    if shared:
        print(f"fixture setup and teardown of {args.list_from:g}s or more (shared loads, not limited):")
        for seconds, when, node in shared:
            print(f"  {seconds:6.2f}s {when:<8} {node}")
    else:
        print(f"no fixture setup or teardown took {args.list_from:g}s or more")

    if not over:
        print(f"no test call took more than {args.limit:g}s in the suite")
        return 0

    print(f"{len(over)} test call(s) took more than {args.limit:g}s in the suite; re-timing each alone:")
    failed = 0
    for suite_seconds, node in over:
        alone_seconds = alone(node)
        if alone_seconds is None:
            failed += 1
            print(f"  FAIL  suite {suite_seconds:6.2f}s  alone   n/a    {node} (did not pass alone)")
        elif alone_seconds > args.limit:
            failed += 1
            print(f"  FAIL  suite {suite_seconds:6.2f}s  alone {alone_seconds:6.2f}s  {node}")
        else:
            print(f"  ok    suite {suite_seconds:6.2f}s  alone {alone_seconds:6.2f}s  {node}")

    if failed:
        print(f"{failed} test(s) still take more than {args.limit:g}s for their call when run alone")
        print(
            "shrink the input, move the wait onto the jumping clock (tests/_jumping_clock.py), "
            "or move work several tests share into one module-scoped fixture"
        )
        return 1
    print(f"every test within the {args.limit:g}s limit when run alone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
