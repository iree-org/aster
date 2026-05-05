#!/usr/bin/env python3
"""Migrate old amdgcn.branch/amdgcn.cbranch ops to new ISA-first s_branch/s_cbranch_* ops.

Old format:
  amdgcn.branch s_branch ^DEST
  amdgcn.cbranch s_cbranch_VARIANT %COND ^DEST fallthrough (^FALLTHRU) : TYPE

New format:
  amdgcn.s_branch ^DEST
  amdgcn.s_cbranch_VARIANT %COND, true(^DEST) false(^FALLTHRU) : TYPE

The old `dest` maps to new `true_dest` and old `fallthrough` maps to new `false_dest`.
"""

import argparse
import re
import sys
from pathlib import Path

# Matches: amdgcn.branch s_branch ^DEST (with optional trailing spaces/newline)
_BRANCH_RE = re.compile(r"amdgcn\.branch\s+s_branch\s+(\^[^\s,;]+)")

# Matches: amdgcn.cbranch s_cbranch_VARIANT %COND ^DEST fallthrough (^FALLTHRU)
# The space between "fallthrough" and "(" is optional.
_CBRANCH_RE = re.compile(
    r"amdgcn\.cbranch\s+(s_cbranch_\w+)\s+(%\S+)\s+(\^\S+)\s+fallthrough\s*\((\^[^\)]+)\)"
)


def migrate(text: str) -> str:
    text = _BRANCH_RE.sub(r"amdgcn.s_branch \1", text)
    text = _CBRANCH_RE.sub(r"amdgcn.\1 \2, true(\3) false(\4)", text)
    return text


def process_file(path: Path, dry_run: bool) -> bool:
    """Return True if the file was (or would be) modified."""
    original = path.read_text(encoding="utf-8")
    updated = migrate(original)
    if updated == original:
        return False
    if dry_run:
        print(f"[dry-run] would update: {path}")
    else:
        path.write_text(updated, encoding="utf-8")
        print(f"updated: {path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to process (recursively for directories).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    parser.add_argument(
        "--extensions",
        default=".mlir,.py",
        help="Comma-separated list of file extensions to process (default: .mlir,.py).",
    )
    args = parser.parse_args()

    extensions = tuple(e.strip() for e in args.extensions.split(","))

    changed = 0
    checked = 0
    for root in args.paths:
        root = root.resolve()
        if root.is_file():
            candidates = [root]
        else:
            candidates = [
                p for p in root.rglob("*") if p.is_file() and p.suffix in extensions
            ]
        for path in candidates:
            checked += 1
            if process_file(path, args.dry_run):
                changed += 1

    print(
        f"\n{'[dry-run] ' if args.dry_run else ''}checked {checked} file(s), "
        f"{'would change' if args.dry_run else 'changed'} {changed}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
