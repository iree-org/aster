#!/usr/bin/env python3
"""Migrate VOP instructions from old format to new format in .mlir and .py files.

Old VOP2/VOP3 format:
  [%res [, %res2] =] amdgcn.vop2 <mnemonic> outs %vdst [dst1 = %dst1]
      ins %src0, %src1 [src2 = %src2] [attr-dict] : type, type, ...

New format:
  amdgcn.<mnemonic> outs(%dst0 [, %dst1]) ins(%src0, %src1 [, %src2])
      : outs(type [, type]) ins(type, type [, type])

Old VOP1 format:
  %res = amdgcn.vop1.vop1 #amdgcn.inst<mnemonic> %dst, %src0
      : (dstType, srcType) -> resultType

New format:
  amdgcn.<mnemonic> outs(%dst) ins(%src0) : outs(dstType) ins(srcType)
"""

import argparse
import os
import re
import sys


def parse_type(text, pos):
    """Parse a single MLIR type starting at pos, return (type_str, end_pos).

    Handles nested angle brackets like !amdgcn.sgpr<[? + 2]>.
    """
    start = pos
    depth = 0
    while pos < len(text):
        ch = text[pos]
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif ch in (",", ")", "\n") and depth == 0:
            break
        pos += 1
    return text[start:pos].strip(), pos


def parse_type_list(text, pos):
    """Parse a comma-separated list of types starting at pos."""
    types = []
    while pos < len(text):
        text_stripped = text[pos:].lstrip()
        pos = pos + len(text[pos:]) - len(text_stripped)
        if not text_stripped or text_stripped[0] in (")", "\n"):
            break
        ty, pos = parse_type(text, pos)
        if ty:
            types.append(ty)
        text_stripped = text[pos:].lstrip()
        pos = pos + len(text[pos:]) - len(text_stripped)
        if pos < len(text) and text[pos] == ",":
            pos += 1
        else:
            break
    return types, pos


def _operand_re():
    """Regex fragment matching an MLIR operand including FileCheck variables.

    Matches: %name, %name#0, %[[NAME]], %[[NAME:.*]], %[[NAME:.*]]#0
    """
    return r"%(?:\[\[[\w:.* ]+\]\](?:#\d+)?|[\w#:.]+)"


def migrate_vop2_vop3_line(line, old_mnemonic, new_mnemonic, old_op):
    """Migrate a single VOP2/VOP3 instruction occurrence in a line.

    old_op is 'vop2' or 'vop3'.
    """
    op_re = _operand_re()
    pattern_str = (
        r"((?:" + op_re + r"(?:\s*,\s*" + op_re + r")*\s*=\s*)?)"
        r"(?:amdgcn\.)?"
        + re.escape(old_op)
        + r"\s+"
        + re.escape(old_mnemonic)
        + r"\s+outs\s+"
    )
    match = re.search(pattern_str, line)
    if not match:
        return line

    result_assign = match.group(1).strip()
    rest = line[match.end() :]

    vdst_match = re.match(r"(" + op_re + r")", rest)
    if not vdst_match:
        return line
    vdst0 = vdst_match.group(1)
    rest = rest[vdst_match.end() :]

    dst1 = None
    dst1_match = re.match(r"\s+dst1\s*=\s*(" + op_re + r")", rest)
    if dst1_match:
        dst1 = dst1_match.group(1)
        rest = rest[dst1_match.end() :]

    ins_match = re.match(r"\s+ins\s+", rest)
    if not ins_match:
        return line
    rest = rest[ins_match.end() :]

    src0_match = re.match(r"(" + op_re + r")\s*,\s*", rest)
    if not src0_match:
        return line
    src0 = src0_match.group(1).strip()
    rest = rest[src0_match.end() :]

    src1_match = re.match(r"(" + op_re + r")", rest)
    if not src1_match:
        return line
    src1 = src1_match.group(1).strip()
    rest = rest[src1_match.end() :]

    src2 = None
    src2_match = re.match(r"\s+src2\s*=\s*(" + op_re + r")", rest)
    if src2_match:
        src2 = src2_match.group(1)
        rest = rest[src2_match.end() :]

    # Parse colon and type list.
    # The types are: vdst_type [, dst1_type], src0_type, src1_type [, src2_type]
    # They may span to the next line, so we need to handle multi-line.
    colon_match = re.match(r"\s*:", rest)
    if not colon_match:
        return line
    rest = rest[colon_match.end() :]

    # Collect all types.
    types, _ = parse_type_list(rest, 0)
    if not types:
        return line

    # Separate out types into outs and ins.
    idx = 0
    vdst_type = types[idx]
    idx += 1
    dst1_type = None
    if dst1:
        dst1_type = types[idx]
        idx += 1
    src0_type = types[idx]
    idx += 1
    src1_type = types[idx]
    idx += 1
    src2_type = None
    if src2:
        src2_type = types[idx]
        idx += 1

    # Build new format.
    outs_operands = vdst0
    if dst1:
        outs_operands += ", " + dst1
    ins_operands = src0 + ", " + src1
    if src2:
        ins_operands += ", " + src2

    outs_types = vdst_type
    if dst1_type:
        outs_types += ", " + dst1_type
    ins_types = src0_type + ", " + src1_type
    if src2_type:
        ins_types += ", " + src2_type

    # Preserve everything before the match (includes leading whitespace,
    # comment prefixes like '// CHECK:', etc.).
    before_match = line[: match.start()]

    result_part = ""
    if result_assign:
        result_part = result_assign + " "

    new_line = (
        before_match
        + result_part
        + "amdgcn."
        + new_mnemonic
        + " outs("
        + outs_operands
        + ")"
        + " ins("
        + ins_operands
        + ")"
        + " : outs("
        + outs_types
        + ")"
        + " ins("
        + ins_types
        + ")"
    )
    return new_line


def migrate_vop1_line(line, old_mnemonic, new_mnemonic):
    """Migrate a VOP1 instruction occurrence in a line.

    Old: %result = amdgcn.vop1.vop1 #amdgcn.inst<mnemonic> %dst, %src0
             : (dstType, srcType) -> resultType
    New: amdgcn.<new_mnemonic> outs(%dst) ins(%src0)
             : outs(dstType) ins(srcType)
    """
    op_re = _operand_re()
    pattern_str = (
        r"((?:" + op_re + r"\s*=\s*)?)"
        r"(?:amdgcn\.)?vop1\.vop1\s+(?:#amdgcn\.inst)?<"
        + re.escape(old_mnemonic)
        + r">\s+"
    )
    match = re.search(pattern_str, line)
    if not match:
        return line

    result_assign = match.group(1).strip()
    rest = line[match.end() :]

    dst_match = re.match(r"(" + op_re + r")\s*,\s*(" + op_re + r")", rest)
    if not dst_match:
        return line
    dst = dst_match.group(1)
    src0 = dst_match.group(2)
    rest = rest[dst_match.end() :]

    # Parse optional attr-dict and the type section.
    attr_match = re.match(r"\s*(\{[^}]*\})", rest)
    attr_dict = ""
    if attr_match:
        attr_dict = attr_match.group(1)
        rest = rest[attr_match.end() :]

    type_match = re.match(r"\s*:\s*\(([^)]+)\)\s*->\s*(\S+)", rest)

    before_match = line[: match.start()]

    result_part = ""
    if result_assign:
        result_part = result_assign + " "

    type_section = ""
    if type_match:
        inner_types = type_match.group(1)
        type_parts = [t.strip() for t in inner_types.split(",")]
        if len(type_parts) >= 2:
            dst_type = type_parts[0]
            src_type = type_parts[1]
            type_section = " : outs(" + dst_type + ") ins(" + src_type + ")"

    attr_section = ""
    if attr_dict:
        attr_section = " " + attr_dict

    new_line = (
        before_match
        + result_part
        + "amdgcn."
        + new_mnemonic
        + " outs("
        + dst
        + ")"
        + " ins("
        + src0
        + ")"
        + attr_section
        + type_section
    )
    return new_line


def migrate_vop1_lane_line(line, old_mnemonic, new_mnemonic):
    """Migrate a VOP1 lane instruction occurrence in a line.

    Old: [%result =] amdgcn.vop1.lane #amdgcn.inst<mnemonic> %dst, %src0
             : (dstType, srcType) -> resultType
    New: amdgcn.<new_mnemonic> outs(%dst) ins(%src0)
             : outs(dstType) ins(srcType)
    """
    op_re = _operand_re()
    pattern_str = (
        r"((?:" + op_re + r"\s*=\s*)?)"
        r"(?:amdgcn\.)?vop1\.lane\s+(?:#amdgcn\.inst)?<"
        + re.escape(old_mnemonic)
        + r">\s+"
    )
    match = re.search(pattern_str, line)
    if not match:
        return line

    result_assign = match.group(1).strip()
    rest = line[match.end() :]

    dst_match = re.match(r"(" + op_re + r")\s*,\s*(" + op_re + r")", rest)
    if not dst_match:
        return line
    dst = dst_match.group(1)
    src0 = dst_match.group(2)
    rest = rest[dst_match.end() :]

    type_match = re.match(r"\s*(\{[^}]*\}\s*)?:\s*\(([^)]+)\)\s*->\s*(\S+)", rest)
    if not type_match:
        return line

    attr_dict = type_match.group(1) or ""
    inner_types = type_match.group(2)
    type_parts = [t.strip() for t in inner_types.split(",")]
    if len(type_parts) < 2:
        return line
    dst_type = type_parts[0]
    src_type = type_parts[1]

    before_match = line[: match.start()]

    result_part = ""
    if result_assign:
        result_part = result_assign + " "

    new_line = (
        before_match
        + result_part
        + "amdgcn."
        + new_mnemonic
        + " outs("
        + dst
        + ")"
        + " ins("
        + src0
        + ")"
        + (" " + attr_dict.strip() + " " if attr_dict.strip() else " ")
        + ": outs("
        + dst_type
        + ")"
        + " ins("
        + src_type
        + ")"
    )
    return new_line


def migrate_check_shorthand(line, old_mnemonic, new_mnemonic, old_op):
    """Migrate CHECK-line shorthand patterns.

    These are substring-match CHECK lines that omit the `amdgcn.` prefix, e.g.:
    // CHECK: v_add_f32 outs(%[[DST]]) ins(%[[LHS]], %[[RHS]])
    ->
    // CHECK: v_add_f32 outs(%[[DST]]) ins(%[[LHS]], %[[RHS]])

    They also often lack the type section entirely.
    """
    op_re = _operand_re()

    # Match pattern: <check-prefix> <vop2|vop3> <mnemonic> outs ...
    pattern_str = (
        r"(//\s*CHECK[^:]*:\s*)"
        r"((?:"
        + op_re
        + r"(?:\s*,\s*"
        + op_re
        + r")*\s*=\s*)?)"
        + re.escape(old_op)
        + r"\s+"
        + re.escape(old_mnemonic)
        + r"\s+outs\s+"
    )
    match = re.search(pattern_str, line)
    if not match:
        return line

    check_prefix = match.group(1)
    result_assign = match.group(2).strip()
    rest = line[match.end() :]

    # Parse vdst0.
    vdst_match = re.match(r"(" + op_re + r")", rest)
    if not vdst_match:
        return line
    vdst0 = vdst_match.group(1)
    rest = rest[vdst_match.end() :]

    # Parse optional dst1.
    dst1 = None
    dst1_match = re.match(r"\s+dst1\s*=\s*(" + op_re + r")", rest)
    if dst1_match:
        dst1 = dst1_match.group(1)
        rest = rest[dst1_match.end() :]

    # Parse ins.
    ins_match = re.match(r"\s+ins\s+", rest)
    if not ins_match:
        return line
    rest = rest[ins_match.end() :]

    # Parse src0, src1.
    src0_match = re.match(r"(" + op_re + r")\s*,\s*", rest)
    if not src0_match:
        return line
    src0 = src0_match.group(1).strip()
    rest = rest[src0_match.end() :]

    src1_match = re.match(r"(" + op_re + r")", rest)
    if not src1_match:
        return line
    src1 = src1_match.group(1).strip()
    rest = rest[src1_match.end() :]

    # Parse optional src2.
    src2 = None
    src2_match = re.match(r"\s+src2\s*=\s*(" + op_re + r")", rest)
    if src2_match:
        src2 = src2_match.group(1)
        rest = rest[src2_match.end() :]

    # Build new format outs/ins.
    outs_operands = vdst0
    if dst1:
        outs_operands += ", " + dst1
    ins_operands = src0 + ", " + src1
    if src2:
        ins_operands += ", " + src2

    result_part = ""
    if result_assign:
        result_part = result_assign + " "

    # Handle optional type section.
    type_section = ""
    colon_match = re.match(r"\s*:\s*", rest)
    if colon_match:
        type_rest = rest[colon_match.end() :]
        types, _ = parse_type_list(type_rest, 0)
        if types:
            idx = 0
            vdst_type = types[idx]
            idx += 1
            dst1_type = None
            if dst1:
                dst1_type = types[idx]
                idx += 1
            src0_type = types[idx]
            idx += 1
            src1_type = types[idx]
            idx += 1
            src2_type = None
            if src2 and idx < len(types):
                src2_type = types[idx]
                idx += 1

            outs_types = vdst_type
            if dst1_type:
                outs_types += ", " + dst1_type
            ins_types = src0_type + ", " + src1_type
            if src2_type:
                ins_types += ", " + src2_type
            type_section = " : outs(" + outs_types + ") ins(" + ins_types + ")"

    before_match = line[: match.start()]
    new_line = (
        before_match
        + check_prefix
        + result_part
        + new_mnemonic
        + " outs("
        + outs_operands
        + ")"
        + " ins("
        + ins_operands
        + ")"
        + type_section
    )
    return new_line


def migrate_check_shorthand_vop1(
    line, old_mnemonic, new_mnemonic, vop_prefix="vop1.vop1"
):
    """Migrate CHECK-line shorthand patterns for VOP1/VOP1-lane.

    // CHECK: vop1.vop1 <mnemonic> %dst, %src0 : (dstType, srcType) ->
    resultType
    """
    op_re = _operand_re()
    escaped_prefix = re.escape(vop_prefix)
    pattern_str = (
        r"(//\s*CHECK[^:]*:\s*)"
        r"((?:"
        + op_re
        + r"\s*=\s*)?)"
        + escaped_prefix
        + r"\s+(?:#amdgcn\.inst)?<?"
        + re.escape(old_mnemonic)
        + r">?\s+"
    )
    match = re.search(pattern_str, line)
    if not match:
        return line

    check_prefix = match.group(1)
    result_assign = match.group(2).strip()
    rest = line[match.end() :]

    dst_match = re.match(r"(" + op_re + r")\s*,\s*(" + op_re + r")", rest)
    if not dst_match:
        return line
    dst = dst_match.group(1)
    src0 = dst_match.group(2)
    rest = rest[dst_match.end() :]

    type_section = ""
    type_match = re.match(r"\s*:\s*\(([^)]+)\)\s*->\s*(\S+)", rest)
    if type_match:
        inner_types = type_match.group(1)
        type_parts = [t.strip() for t in inner_types.split(",")]
        if len(type_parts) >= 2:
            dst_type = type_parts[0]
            src_type = type_parts[1]
            type_section = " : outs(" + dst_type + ") ins(" + src_type + ")"

    before_match = line[: match.start()]
    result_part = ""
    if result_assign:
        result_part = result_assign + " "

    new_line = (
        before_match
        + check_prefix
        + result_part
        + new_mnemonic
        + " outs("
        + dst
        + ")"
        + " ins("
        + src0
        + ")"
        + type_section
    )
    return new_line


def migrate_file(filepath, old_mnemonic, new_mnemonic, old_op, dry_run=False):
    """Migrate all occurrences in a file.

    Returns True if file was modified.
    """
    with open(filepath, "r") as f:
        content = f.read()

    if old_mnemonic not in content:
        return False

    lines = content.split("\n")
    new_lines = []
    modified = False
    i = 0
    while i < len(lines):
        line = lines[i]

        if old_op in ("vop2", "vop3"):
            # Handle multi-line: if a line has `amdgcn.vop2 <mnemonic>` but the
            # type list is on the next line, join them.
            if (
                old_op + " " + old_mnemonic in line
                or old_op + " " + old_mnemonic in line.replace("  ", " ")
            ):
                # Check if this line ends with a colon but types are on next line.
                joined = line
                lookahead = i + 1
                # Keep joining lines until we have the full type list.
                while lookahead < len(lines):
                    next_line = lines[lookahead]
                    stripped = next_line.strip()
                    # Types continuation: starts with ! or i (like i32) or is
                    # the : followed by types.
                    if (
                        not stripped
                        or stripped.startswith("//")
                        or stripped.startswith("func")
                        or stripped.startswith("return")
                        or stripped.startswith("%")
                        or stripped.startswith("amdgcn")
                        or stripped.startswith("lsir")
                        or stripped.startswith("arith")
                        or stripped.startswith("}")
                        or stripped.startswith("end_kernel")
                    ):
                        break
                    # This looks like a continuation of types.
                    joined += "\n" + next_line
                    lookahead += 1
                    # If we now have a balanced set of types, stop.
                    if re.search(r":\s*\S", joined) and not joined.rstrip().endswith(
                        ","
                    ):
                        break

                new_joined = migrate_vop2_vop3_line(
                    joined, old_mnemonic, new_mnemonic, old_op
                )

                if new_joined != joined:
                    modified = True
                    # Split back into multiple lines if needed.
                    new_parts = new_joined.split("\n")
                    new_lines.extend(new_parts)
                    i = lookahead
                    continue
                else:
                    new_lines.append(line)
                    i += 1
                    continue

        if old_op in ("vop1", "vop1.lane"):
            vop_prefix = "vop1.vop1" if old_op == "vop1" else "vop1.lane"
            if vop_prefix in line and old_mnemonic in line:
                # Handle multi-line VOP1/VOP1-lane.
                joined = line
                lookahead = i + 1
                # Only look ahead if the type section is not already on this
                # line (indicated by the presence of "->").
                if "->" not in line:
                    while lookahead < len(lines):
                        next_line = lines[lookahead]
                        stripped = next_line.strip()
                        if stripped.startswith(":"):
                            joined += "\n" + next_line
                            lookahead += 1
                            break
                        elif not stripped or stripped.startswith("//"):
                            break
                        else:
                            joined += "\n" + next_line
                            lookahead += 1
                            if "->" in joined:
                                break

                if old_op == "vop1":
                    new_joined = migrate_vop1_line(joined, old_mnemonic, new_mnemonic)
                else:
                    new_joined = migrate_vop1_lane_line(
                        joined, old_mnemonic, new_mnemonic
                    )
                if new_joined != joined:
                    modified = True
                    new_parts = new_joined.split("\n")
                    new_lines.extend(new_parts)
                    i = lookahead
                    continue

        # Try CHECK shorthand migration for VOP2/VOP3.
        if old_op in ("vop2", "vop3") and old_mnemonic in line and "//" in line:
            new_line = migrate_check_shorthand(line, old_mnemonic, new_mnemonic, old_op)
            if new_line != line:
                modified = True
                new_lines.append(new_line)
                i += 1
                continue

        # Try CHECK shorthand migration for VOP1 / VOP1 lane.
        if old_op in ("vop1", "vop1.lane") and old_mnemonic in line and "//" in line:
            vop_prefix = "vop1.vop1" if old_op == "vop1" else "vop1.lane"
            new_line = migrate_check_shorthand_vop1(
                line, old_mnemonic, new_mnemonic, vop_prefix
            )
            if new_line != line:
                modified = True
                new_lines.append(new_line)
                i += 1
                continue

        new_lines.append(line)
        i += 1

    if modified and not dry_run:
        with open(filepath, "w") as f:
            f.write("\n".join(new_lines))

    return modified


def find_files(root_dir, extensions):
    """Find all files with given extensions under root_dir."""
    skip_dirs = {"sandbox", "build", ".git", "__pycache__", "node_modules"}
    results = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for fn in filenames:
            if any(fn.endswith(ext) for ext in extensions):
                results.append(os.path.join(dirpath, fn))
    return sorted(results)


# Mapping from old mnemonic to (new_mnemonic, old_op_type).
# old_op_type is 'vop2', 'vop3', or 'vop1'.
VOP2_SIMPLE = [
    "v_add_f32",
    "v_sub_f32",
    "v_mul_f32",
    "v_min_f32",
    "v_max_f32",
    "v_add_f16",
    "v_sub_f16",
    "v_mul_f16",
    "v_add_u16",
    "v_sub_u16",
    "v_mul_lo_u16",
    "v_lshlrev_b16",
    "v_lshrrev_b16",
    "v_ashrrev_i16",
    "v_lshrrev_b32",
    "v_ashrrev_i32",
    "v_and_b32",
    "v_or_b32",
    "v_xor_b32",
    "v_add_u32",
    "v_sub_u32",
    "v_lshrrev_b32_e32",
    "v_lshlrev_b32_e32",
]

# VOP2 _e32 mnemonics that map to base names (without _e32).
VOP2_E32 = {
    "v_lshrrev_b32_e32": "v_lshrrev_b32",
    "v_lshlrev_b32_e32": "v_lshlrev_b32",
}

VOP2_CARRY = [
    "v_add_co_u32",
    "v_sub_co_u32",
    "v_addc_co_u32",
    "v_subb_co_u32",
    "v_add_i16",
    "v_add_i32",
    "v_cndmask_b32",
]

VOP3_ONLY = [
    "v_mul_lo_u32",
    "v_mul_hi_u32",
    "v_mul_hi_i32",
    "v_mad_u64_u32",
    "v_add3_u32",
    "v_lshl_add_u64",
    "v_pack_b32_f16",
    "v_cvt_pk_fp8_f32",
    "v_cvt_pk_bf8_f32",
    "v_lshlrev_b64",
    "v_lshrrev_b64",
    "v_ashrrev_i64",
]

# VOP3 E64 variants that map to base mnemonics.
VOP3_E64 = {
    "v_add_f32_e64": "v_add_f32",
    "v_sub_f32_e64": "v_sub_f32",
    "v_mul_f32_e64": "v_mul_f32",
    "v_min_f32_e64": "v_min_f32",
    "v_max_f32_e64": "v_max_f32",
    "v_add_f16_e64": "v_add_f16",
    "v_sub_f16_e64": "v_sub_f16",
    "v_mul_f16_e64": "v_mul_f16",
    "v_add_u16_e64": "v_add_u16",
    "v_sub_u16_e64": "v_sub_u16",
    "v_mul_lo_u16_e64": "v_mul_lo_u16",
    "v_lshlrev_b16_e64": "v_lshlrev_b16",
    "v_lshrrev_b16_e64": "v_lshrrev_b16",
    "v_ashrrev_i16_e64": "v_ashrrev_i16",
    "v_lshrrev_b32_e64": "v_lshrrev_b32",
    "v_ashrrev_i32_e64": "v_ashrrev_i32",
    "v_and_b32_e64": "v_and_b32",
    "v_or_b32_e64": "v_or_b32",
    "v_xor_b32_e64": "v_xor_b32",
    "v_add_co_u32_e64": "v_add_co_u32",
    "v_sub_co_u32_e64": "v_sub_co_u32",
    "v_addc_co_u32_e64": "v_addc_co_u32",
    "v_subb_co_u32_e64": "v_subb_co_u32",
    "v_add_u32_e64": "v_add_u32",
    "v_sub_u32_e64": "v_sub_u32",
    "v_subrev_f16_e64": "v_subrev_f16",
    "v_lshlrev_b32_e64": "v_lshlrev_b32",
    "v_subrev_co_u32_e64": "v_subrev_co_u32",
    "v_subbrev_co_u32_e64": "v_subbrev_co_u32",
    "v_subrev_u32_e64": "v_subrev_u32",
}

VOP1_MNEMONICS = [
    "v_cvt_f32_f16",
    "v_cvt_f16_f32",
    "v_cvt_f32_u32",
    "v_cvt_f32_i32",
    "v_cvt_u32_f32",
    "v_cvt_i32_f32",
    "v_mov_b32_e32",
]

# VOP1 mnemonics that map to a different name (strip _e32 suffix).
VOP1_E32 = {
    "v_mov_b32_e32": "v_mov_b32",
}

# VOP1 lane ops use amdgcn.vop1.lane format.
VOP1_LANE_MNEMONICS = [
    "v_readfirstlane_b32",
]


def build_migration_entries():
    """Build the full list of (old_mnemonic, new_mnemonic, old_op)."""
    entries = []
    for m in VOP2_SIMPLE:
        new_m = VOP2_E32.get(m, m)
        entries.append((m, new_m, "vop2"))
    for m in VOP2_CARRY:
        entries.append((m, m, "vop2"))
    for m in VOP3_ONLY:
        entries.append((m, m, "vop3"))
    for old, new in VOP3_E64.items():
        entries.append((old, new, "vop3"))
    for m in VOP1_MNEMONICS:
        new_m = VOP1_E32.get(m, m)
        entries.append((m, new_m, "vop1"))
    for m in VOP1_LANE_MNEMONICS:
        entries.append((m, m, "vop1.lane"))
    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Migrate VOP instructions to new format."
    )
    parser.add_argument(
        "--mnemonic", type=str, help="Migrate only this mnemonic (old name)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        help="Root directory to search for files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be changed without modifying files.",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all migration entries."
    )
    args = parser.parse_args()

    entries = build_migration_entries()

    if args.list:
        for old, new, op in entries:
            print(f"  {old} -> amdgcn.{new} (was amdgcn.{op})")
        return

    if args.mnemonic:
        entries = [(o, n, t) for o, n, t in entries if o == args.mnemonic]
        if not entries:
            print(f"Unknown mnemonic: {args.mnemonic}", file=sys.stderr)
            sys.exit(1)

    files = find_files(args.root, [".mlir", ".py"])

    for old_mnemonic, new_mnemonic, old_op in entries:
        for filepath in files:
            changed = migrate_file(
                filepath, old_mnemonic, new_mnemonic, old_op, dry_run=args.dry_run
            )
            if changed:
                prefix = "[DRY RUN] " if args.dry_run else ""
                print(
                    f"{prefix}Migrated {old_mnemonic} -> amdgcn.{new_mnemonic}"
                    f" in {filepath}"
                )


if __name__ == "__main__":
    main()
