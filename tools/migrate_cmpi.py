#!/usr/bin/env python3
"""Migrate amdgcn.cmpi old-format instructions to new per-instruction ops.

Old format:
    amdgcn.cmpi s_cmp_eq_i32 outs %scc ins %a, %b : outs(!amdgcn.scc<0>) ins(i32, i32)
    %0 = amdgcn.cmpi v_cmp_eq_i32_e64 outs %dst ins %a, %v1 : dps(type) ins(...)

New format:
    amdgcn.s_cmp_eq_i32 outs(%scc) ins(%a, %b) : outs(!amdgcn.scc<0>) ins(i32, i32)
    amdgcn.v_cmp_eq_i32 outs(%dst) ins(%a, %v1) : outs(type) ins(...)

Usage:
    python tools/migrate_cmpi.py file1.mlir file2.mlir ...
    python tools/migrate_cmpi.py --all  # find and migrate all .mlir files under aster/
"""

import argparse
import re
import sys
from pathlib import Path

E64_TO_BASE = {
    "v_cmp_eq_i32_e64": "v_cmp_eq_i32",
}

ALL_MNEMONICS = [
    "s_cmp_eq_i32",
    "s_cmp_lg_i32",
    "s_cmp_gt_i32",
    "s_cmp_ge_i32",
    "s_cmp_lt_i32",
    "s_cmp_le_i32",
    "s_cmp_eq_u32",
    "s_cmp_lg_u32",
    "s_cmp_gt_u32",
    "s_cmp_ge_u32",
    "s_cmp_lt_u32",
    "s_cmp_le_u32",
    "v_cmp_eq_i32",
    "v_cmp_ne_i32",
    "v_cmp_lt_i32",
    "v_cmp_le_i32",
    "v_cmp_gt_i32",
    "v_cmp_ge_i32",
    "v_cmp_lt_u32",
    "v_cmp_le_u32",
    "v_cmp_gt_u32",
    "v_cmp_ge_u32",
    "v_cmp_eq_i32_e64",
]


def migrate_mlir_content(content: str) -> str:
    """Migrate all amdgcn.cmpi occurrences in MLIR content."""
    # Join continuation lines: if a line ends with cmpi operands and the next
    # line starts with `:`, join them so the regex can match the full statement.
    # We need to handle multiline statements where `: outs(...)` is on the next line.
    lines = content.split("\n")
    joined_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if next line is a continuation (starts with whitespace + `:`)
        if i + 1 < len(lines) and re.match(r"^\s+:", lines[i + 1]):
            # Check if current line has an amdgcn.cmpi that hasn't been closed
            if "amdgcn.cmpi" in line or "cmpi " in line:
                joined_lines.append(line + "\n" + lines[i + 1])
                i += 2
                continue
        joined_lines.append(line)
        i += 1

    result_lines = []
    for line in joined_lines:
        line = migrate_line(line)
        result_lines.append(line)

    return "\n".join(result_lines)


def migrate_line(line: str) -> str:
    """Migrate a single line (possibly joined multiline) containing amdgcn.cmpi."""
    # Pattern 1: Full IR line with `amdgcn.cmpi`
    # Handles: `amdgcn.cmpi <mnemonic> outs <outs> ins <ins> : <types>`
    # Also: `%0 = amdgcn.cmpi <mnemonic> outs <outs> ins <ins> : dps(<type>) ins(<types>)`
    # Also: CHECK lines like `// CHECK: cmpi s_cmp_eq_i32 outs ...`

    # Handle full amdgcn.cmpi with result assignment
    # %0 = amdgcn.cmpi v_cmp_eq_i32_e64 outs %dst ins %a, %v1 : dps(type) ins(types)
    line = re.sub(
        r"(%\S+\s*=\s*)amdgcn\.cmpi\s+(\S+)\s+outs\s+(\S+)\s+ins\s+([^:]+?):\s*dps\(([^)]+)\)\s+ins\(([^)]+)\)",
        lambda m: _rewrite_cmpi_with_result(m),
        line,
    )

    # Handle amdgcn.cmpi without result (standard form)
    # amdgcn.cmpi <mnemonic> outs <dest> ins <lhs>, <rhs>\n    : outs(<types>) ins(<types>)
    # amdgcn.cmpi <mnemonic> outs <dest> ins <lhs>, <rhs> : outs(<types>) ins(<types>)
    line = re.sub(
        r"amdgcn\.cmpi\s+(\S+)\s+outs\s+(\S+)\s+ins\s+([^:]+?):\s*outs\(([^)]+)\)\s+ins\(([^)]+)\)",
        lambda m: _rewrite_cmpi_standard(m),
        line,
    )

    # Handle CHECK lines that match the printed form (no `amdgcn.` prefix in CHECK)
    # e.g. `// CHECK: cmpi s_cmp_eq_i32 outs %{{.*}} ins %{{.*}}, %{{.*}} : outs(...) ins(...)`
    line = re.sub(
        r"(//\s*CHECK[^:]*:\s*(?:.*?))cmpi\s+(\S+)\s+outs\s+(\S+)\s+ins\s+([^:]+?):\s*outs\(([^)]+)\)\s+ins\(([^)]+)\)",
        lambda m: _rewrite_check_cmpi(m),
        line,
    )

    # Handle short CHECK lines like `// CHECK: cmpi s_cmp_eq_i32` (no types)
    line = re.sub(
        r"(//\s*CHECK[^:]*:\s*(?:.*?))cmpi\s+((?:s_cmp_|v_cmp_)\S+)\b(?!\s+outs)",
        lambda m: _rewrite_check_cmpi_short(m),
        line,
    )

    # Handle CHECK-NOT lines like `// CHECK-NOT: cmpi s_cmp_eq_i32`
    line = re.sub(
        r"(//\s*CHECK-NOT:\s*(?:.*?))cmpi\s+((?:s_cmp_|v_cmp_)\S+)\b",
        lambda m: _rewrite_check_cmpi_short(m),
        line,
    )

    # Handle CHECK-NOT: cmpi (bare, no mnemonic)
    line = re.sub(
        r"(//\s*CHECK-NOT:\s+)cmpi\b",
        lambda m: m.group(1) + "s_cmp_",
        line,
    )

    # Handle CHECK: cmpi (bare, used as "any cmpi" match) - match CHECK lines
    # that have just `cmpi` with no mnemonic following
    # Be careful not to match `cmpi` that was already converted.

    return line


def _resolve_mnemonic(mnemonic: str) -> str:
    """Resolve E64 mnemonics to their base form."""
    return E64_TO_BASE.get(mnemonic, mnemonic)


def _rewrite_cmpi_with_result(m: re.Match) -> str:
    """Rewrite `%0 = amdgcn.cmpi <mnemonic> outs <dest> ins <ins> : dps(<type>) ins(<types>)`."""
    _ = m.group(1)  # `%0 = ` — dropped in new format.
    mnemonic = _resolve_mnemonic(m.group(2))
    dest = m.group(3)
    ins = m.group(4).strip().rstrip(",").strip()
    dest_type = m.group(5)
    in_types = m.group(6)
    # New format has no SSA result for DPS ops that become outs-style.
    # Drop the result assignment.
    return (
        f"amdgcn.{mnemonic} outs({dest}) ins({ins}) : outs({dest_type}) ins({in_types})"
    )


def _rewrite_cmpi_standard(m: re.Match) -> str:
    """Rewrite `amdgcn.cmpi <mnemonic> outs <dest> ins <ins> : outs(<types>) ins(<types>)`."""
    mnemonic = _resolve_mnemonic(m.group(1))
    dest = m.group(2)
    ins = m.group(3).strip().rstrip(",").strip()
    out_types = m.group(4)
    in_types = m.group(5)
    return (
        f"amdgcn.{mnemonic} outs({dest}) ins({ins}) : outs({out_types}) ins({in_types})"
    )


def _rewrite_check_cmpi(m: re.Match) -> str:
    """Rewrite CHECK line with full cmpi pattern."""
    prefix = m.group(1)
    mnemonic = _resolve_mnemonic(m.group(2))
    dest = m.group(3)
    ins = m.group(4).strip().rstrip(",").strip()
    out_types = m.group(5)
    in_types = m.group(6)
    return f"{prefix}{mnemonic} outs({dest}) ins({ins}) : outs({out_types}) ins({in_types})"


def _rewrite_check_cmpi_short(m: re.Match) -> str:
    """Rewrite short CHECK line like `// CHECK: cmpi s_cmp_eq_i32`."""
    prefix = m.group(1)
    mnemonic = _resolve_mnemonic(m.group(2))
    return f"{prefix}{mnemonic}"


def migrate_file(filepath: Path, dry_run: bool = False) -> bool:
    """Migrate a single file.

    Returns True if changes were made.
    """
    content = filepath.read_text()
    new_content = migrate_mlir_content(content)
    if content != new_content:
        if dry_run:
            print(f"Would modify: {filepath}")
        else:
            filepath.write_text(new_content)
            print(f"Modified: {filepath}")
        return True
    return False


def find_mlir_files(root: Path) -> list:
    """Find all .mlir files that contain amdgcn.cmpi or cmpi s_cmp/v_cmp."""
    result = []
    for f in root.rglob("*.mlir"):
        if "build" in f.parts:
            continue
        content = f.read_text()
        if "cmpi" in content and ("s_cmp_" in content or "v_cmp_" in content):
            result.append(f)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description="Migrate amdgcn.cmpi to new format")
    parser.add_argument("files", nargs="*", help="Files to migrate")
    parser.add_argument(
        "--all", action="store_true", help="Find and migrate all .mlir files"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would change")
    parser.add_argument("--root", default=".", help="Root directory for --all")
    args = parser.parse_args()

    if args.all:
        files = find_mlir_files(Path(args.root))
    else:
        files = [Path(f) for f in args.files]

    if not files:
        print("No files to process.", file=sys.stderr)
        return 1

    changed = 0
    for f in files:
        if migrate_file(f, dry_run=args.dry_run):
            changed += 1

    print(
        f"\n{changed}/{len(files)} files {'would be ' if args.dry_run else ''}modified."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
