#!/usr/bin/env python3
"""Migrate old SOP1/SOP2/SOPP/SWaitcnt instruction formats to the new format.

The old format uses wrapper ops (amdgcn.sop1, amdgcn.sop2, amdgcn.sopp.sopp,
amdgcn.sopp.s_waitcnt) with an opcode attribute. The new format uses individual
ops per instruction (amdgcn.s_add_u32, amdgcn.s_mov_b32, etc.) with explicit
outs(...) / ins(...) sections.

Usage:
    python migrate_sop.py [--phase PHASE] [--dry-run] [FILES...]

If no files are given, searches under the script's parent directory for .mlir
and .py files.
"""

import argparse
import glob
import os
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Instruction category definitions
# ---------------------------------------------------------------------------

# SOP2Out2In2: 2 outs (dst0 + scc_dst), 2 ins (src0, src1)
SOP2_OUT2_IN2 = [
    "s_add_u32",
    "s_sub_u32",
    "s_add_i32",
    "s_sub_i32",
    "s_min_i32",
    "s_min_u32",
    "s_max_i32",
    "s_max_u32",
    "s_and_b32",
    "s_and_b64",
    "s_or_b32",
    "s_or_b64",
    "s_xor_b32",
    "s_xor_b64",
    "s_andn2_b32",
    "s_orn2_b32",
    "s_nand_b32",
    "s_nor_b32",
    "s_xnor_b32",
    "s_lshl_b32",
    "s_lshl_b64",
    "s_lshr_b32",
    "s_lshr_b64",
    "s_ashr_i32",
    "s_ashr_i64",
    "s_bfe_u32",
    "s_bfe_i32",
    "s_lshl1_add_u32",
    "s_lshl2_add_u32",
    "s_lshl3_add_u32",
    "s_lshl4_add_u32",
    "s_absdiff_i32",
]

# SOP2Out2In3: 2 outs (dst0 + scc_dst), 3 ins (src0, src1, scc_src)
SOP2_OUT2_IN3 = ["s_addc_u32", "s_subb_u32"]

# SOP2Out1In2: 1 out (dst0), 2 ins (src0, src1) -- no SCC
SOP2_OUT1_IN2 = ["s_bfm_b32", "s_mul_hi_i32", "s_mul_hi_u32", "s_mul_i32"]

# SOP2Out1In3: 1 out (dst0), 3 ins (src0, src1, scc_src)
SOP2_OUT1_IN3 = ["s_cselect_b32", "s_cselect_b64"]

ALL_SOP2 = SOP2_OUT2_IN2 + SOP2_OUT2_IN3 + SOP2_OUT1_IN2 + SOP2_OUT1_IN3

# An SSA value in MLIR, also FileCheck pattern %[[NAME:.*]] or %[[NAME]]#N.
VAL = r"%(?:[\w#$.]+|\[\[[^\]]+\]\](?:#\d+)?)"

# An MLIR type (non-greedy, balanced angle brackets).
TYP = r"[!i][\w.]+(?:<[^>]*>)?"


# ---------------------------------------------------------------------------
# Phase 1: s_waitcnt
# ---------------------------------------------------------------------------


def migrate_s_waitcnt_mlir(text: str) -> str:
    """Old: [amdgcn.]sopp.s_waitcnt ... -> New: [amdgcn.]s_waitcnt ..."""

    def _replace_waitcnt(m):
        ns = m.group(1) or ""
        return f"{ns}s_waitcnt "

    text = re.sub(
        r"(amdgcn\.)?sopp\.s_waitcnt\s+(?:<s_waitcnt>|#amdgcn\.inst<s_waitcnt>)\s*",
        _replace_waitcnt,
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Phase 2: SOPP (s_barrier, s_nop)
# ---------------------------------------------------------------------------


def migrate_s_barrier_mlir(text: str) -> str:
    """Old: [amdgcn.]sopp.sopp #amdgcn.inst<s_barrier> -> New: [amdgcn.]s_barrier"""

    def _replace_barrier(m):
        ns = m.group(1) or ""
        return f"{ns}s_barrier"

    text = re.sub(
        r"(amdgcn\.)?sopp\.sopp\s+(?:#amdgcn\.inst<s_barrier>|<s_barrier>)",
        _replace_barrier,
        text,
    )
    return text


def migrate_s_nop_mlir(text: str) -> str:
    """Old: [amdgcn.]sopp.sopp #amdgcn.inst<s_nop>[, imm = N] -> New: [amdgcn.]s_nop N"""

    def _replace_nop(m):
        ns = m.group("ns") or ""
        imm = m.group("imm") or "0"
        return f"{ns}s_nop {imm}"

    text = re.sub(
        r"(?P<ns>amdgcn\.)?sopp\.sopp\s+(?:#amdgcn\.inst<s_nop>|<s_nop>)"
        r"(?:\s*,\s*imm\s*=\s*(?P<imm>\d+))?",
        _replace_nop,
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Phase 3: SOP1 s_mov_b32
# ---------------------------------------------------------------------------


def migrate_s_mov_b32_mlir(text: str) -> str:
    r"""Migrate s_mov_b32 from old SOP1 format.

    Old: [%r = ] [amdgcn.]sop1 s_mov_b32 outs %dst ins %src [attr-dict] : dst_ty, src_ty
    New: [%r = ] [amdgcn.]s_mov_b32 outs(%dst) ins(%src) [attr-dict] : outs(dst_ty) ins(src_ty)
    """
    pattern = (
        r"(?P<prefix>(?:(?P<result>" + VAL + r")\s*=\s*)?)"
        r"(?P<ns>amdgcn\.)?sop1\s+s_mov_b32\s+outs\s+(?P<dst>" + VAL + r")"
        r"\s+ins\s+(?P<src>" + VAL + r")"
        r"(?P<attr>\s*\{[^}]*\})?"
        r"\s*\n?\s*:\s*(?P<dst_ty>" + TYP + r")\s*,\s*(?P<src_ty>" + TYP + r")"
    )

    def _replace(m):
        prefix = m.group("prefix").rstrip()
        ns = m.group("ns") or ""
        dst = m.group("dst")
        src = m.group("src")
        attr = m.group("attr") or ""
        dst_ty = m.group("dst_ty").strip()
        src_ty = m.group("src_ty").strip()
        result_prefix = f"{prefix} " if prefix else ""
        return (
            f"{result_prefix}{ns}s_mov_b32 outs({dst}) ins({src})"
            f"{attr} : outs({dst_ty}) ins({src_ty})"
        )

    text = re.sub(pattern, _replace, text)
    return text


# ---------------------------------------------------------------------------
# Phase 4-7: SOP2 instructions
# ---------------------------------------------------------------------------


def migrate_sop2_mlir(text: str, mnemonics: list, category: str) -> str:
    r"""Migrate SOP2 instructions from old to new format."""
    mnemonic_pat = "|".join(re.escape(m) for m in mnemonics)

    pattern = (
        r"(?P<prefix>(?:(?P<result>" + VAL + r")\s*=\s*)?)"
        r"(?P<ns>amdgcn\.)?sop2\s+(?P<mnemonic>" + mnemonic_pat + r")"
        r"\s+outs\s+(?P<dst>" + VAL + r")"
        r"\s+ins\s+(?P<src0>" + VAL + r")\s*,\s*(?P<src1>" + VAL + r")"
        r"(?P<attr>\s*\{[^}]*\})?"
        r"\s*\n?\s*:\s*(?P<types>[^\n]+?)\s*$"
    )

    def _replace(m):
        prefix = m.group("prefix").rstrip()
        ns = m.group("ns") or ""
        mnemonic = m.group("mnemonic")
        dst = m.group("dst")
        src0 = m.group("src0")
        src1 = m.group("src1")
        attr = m.group("attr") or ""
        types_str = m.group("types").strip()
        types = [t.strip() for t in types_str.split(",")]

        if category == "out1_in2":
            dst_ty = types[0] if len(types) > 0 else "!amdgcn.sgpr"
            src0_ty = types[1] if len(types) > 1 else "!amdgcn.sgpr"
            src1_ty = types[2] if len(types) > 2 else "!amdgcn.sgpr"
            result_prefix = f"{prefix} " if prefix else ""
            return (
                f"{result_prefix}{ns}{mnemonic} outs({dst}) ins({src0}, {src1})"
                f"{attr} : outs({dst_ty}) ins({src0_ty}, {src1_ty})"
            )

        elif category == "out1_in3":
            dst_ty = types[0] if len(types) > 0 else "!amdgcn.sgpr"
            src0_ty = types[1] if len(types) > 1 else "!amdgcn.sgpr"
            src1_ty = types[2] if len(types) > 2 else "!amdgcn.sgpr"
            result_prefix = f"{prefix} " if prefix else ""
            scc_name = f"%_scc_src_{mnemonic.replace('s_', '')}"
            alloca_line = f"{scc_name} = {ns}alloca : !amdgcn.scc\n    "
            return (
                f"{alloca_line}{result_prefix}{ns}{mnemonic} outs({dst}) "
                f"ins({src0}, {src1}, {scc_name})"
                f"{attr} : outs({dst_ty}) ins({src0_ty}, {src1_ty}, !amdgcn.scc<0>)"
            )

        elif category == "out2_in2":
            dst_ty = types[0] if len(types) > 0 else "!amdgcn.sgpr"
            src0_ty = types[1] if len(types) > 1 else "!amdgcn.sgpr"
            src1_ty = types[2] if len(types) > 2 else "!amdgcn.sgpr"
            scc_name = f"%_scc_dst_{mnemonic.replace('s_', '')}"
            alloca_line = f"{scc_name} = {ns}alloca : !amdgcn.scc\n    "
            if prefix:
                result_name = m.group("result")
                return (
                    f"{alloca_line}{result_name}, {scc_name}_res = "
                    f"{ns}{mnemonic} outs({dst}, {scc_name}) "
                    f"ins({src0}, {src1})"
                    f"{attr} : outs({dst_ty}, !amdgcn.scc<0>) ins({src0_ty}, {src1_ty})"
                )
            else:
                return (
                    f"{alloca_line}{ns}{mnemonic} outs({dst}, {scc_name}) "
                    f"ins({src0}, {src1})"
                    f"{attr} : outs({dst_ty}, !amdgcn.scc<0>) ins({src0_ty}, {src1_ty})"
                )

        elif category == "out2_in3":
            dst_ty = types[0] if len(types) > 0 else "!amdgcn.sgpr"
            src0_ty = types[1] if len(types) > 1 else "!amdgcn.sgpr"
            src1_ty = types[2] if len(types) > 2 else "!amdgcn.sgpr"
            scc_out = f"%_scc_dst_{mnemonic.replace('s_', '')}"
            scc_in = f"%_scc_src_{mnemonic.replace('s_', '')}"
            alloca_lines = (
                f"{scc_out} = {ns}alloca : !amdgcn.scc\n    "
                f"{scc_in} = {ns}alloca : !amdgcn.scc\n    "
            )
            if prefix:
                result_name = m.group("result")
                return (
                    f"{alloca_lines}{result_name}, {scc_out}_res = "
                    f"{ns}{mnemonic} outs({dst}, {scc_out}) "
                    f"ins({src0}, {src1}, {scc_in})"
                    f"{attr} : outs({dst_ty}, !amdgcn.scc<0>) "
                    f"ins({src0_ty}, {src1_ty}, !amdgcn.scc<0>)"
                )
            else:
                return (
                    f"{alloca_lines}{ns}{mnemonic} outs({dst}, {scc_out}) "
                    f"ins({src0}, {src1}, {scc_in})"
                    f"{attr} : outs({dst_ty}, !amdgcn.scc<0>) "
                    f"ins({src0_ty}, {src1_ty}, !amdgcn.scc<0>)"
                )

        return m.group(0)

    text = re.sub(pattern, _replace, text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# Phase 8: Drop s_cbranch_g_fork
# ---------------------------------------------------------------------------


def drop_cbranch_g_fork(text: str) -> str:
    """Remove s_cbranch_g_fork lines from MLIR text."""
    pattern = (
        r"^\s*amdgcn\.sop2\s+s_cbranch_g_fork\s+ins\s+"
        + VAL
        + r"\s*,\s*"
        + VAL
        + r"\s*:\s*[^\n]+\n?"
    )
    text = re.sub(pattern, "", text, flags=re.MULTILINE)
    return text


# ---------------------------------------------------------------------------
# Apply all migrations
# ---------------------------------------------------------------------------


def migrate_mlir(text: str, phase: int | None = None) -> str:
    """Apply MLIR migrations up to the given phase (or all if None)."""
    if phase is None or phase >= 1:
        text = migrate_s_waitcnt_mlir(text)
    if phase is None or phase >= 2:
        text = migrate_s_barrier_mlir(text)
        text = migrate_s_nop_mlir(text)
    if phase is None or phase >= 3:
        text = migrate_s_mov_b32_mlir(text)
    if phase is None or phase >= 4:
        text = migrate_sop2_mlir(text, SOP2_OUT1_IN2, "out1_in2")
    if phase is None or phase >= 5:
        text = migrate_sop2_mlir(text, SOP2_OUT1_IN3, "out1_in3")
    if phase is None or phase >= 6:
        text = migrate_sop2_mlir(text, SOP2_OUT2_IN2, "out2_in2")
    if phase is None or phase >= 7:
        text = migrate_sop2_mlir(text, SOP2_OUT2_IN3, "out2_in3")
    if phase is None or phase >= 8:
        text = drop_cbranch_g_fork(text)
    return text


def find_files(root: str) -> list[str]:
    """Find all .mlir and .py files under root in relevant directories."""
    search_dirs = [
        "test",
        "examples",
        "mlir_kernels",
        "contrib",
        "python",
    ]
    files = []
    for d in search_dirs:
        dirpath = os.path.join(root, d)
        if not os.path.isdir(dirpath):
            continue
        for ext in ("*.mlir", "*.py"):
            files.extend(glob.glob(os.path.join(dirpath, "**", ext), recursive=True))
    # Exclude build directories.
    files = [f for f in files if "/build/" not in f]
    return sorted(files)


def process_file(path: str, phase: int | None, dry_run: bool) -> bool:
    """Process a single file.

    Returns True if changes were made.
    """
    with open(path, "r") as f:
        original = f.read()

    modified = migrate_mlir(original, phase)

    if modified != original:
        if dry_run:
            print(f"[DRY RUN] Would modify: {path}")
        else:
            with open(path, "w") as f:
                f.write(modified)
            print(f"Modified: {path}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        type=int,
        default=None,
        help="Run migrations up to this phase (1-8). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without modifying files.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Files to process. If empty, searches under the repo root.",
    )
    args = parser.parse_args()

    if args.files:
        files = args.files
    else:
        script_dir = Path(__file__).resolve().parent
        repo_root = script_dir.parent
        files = find_files(str(repo_root))

    changed = 0
    for f in files:
        if process_file(f, args.phase, args.dry_run):
            changed += 1

    print(f"\n{'Would modify' if args.dry_run else 'Modified'} {changed} file(s).")


if __name__ == "__main__":
    main()
