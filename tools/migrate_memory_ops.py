#!/usr/bin/env python3
"""Migrate old Memory.td instruction format to new per-instruction ops.

Old format (amdgcn.load / amdgcn.store / amdgcn.load_lds) uses a generic op
with an opcode attribute. New format uses individual per-instruction ops
defined in VMem.td, DS.td, and SMem.td.

Usage:
    python migrate_memory_ops.py --category ds_read [--dry-run] [--files FILE ...]
    python migrate_memory_ops.py --category all [--dry-run]

Categories: ds_read, ds_write, ds_perm, global_load, global_store,
            buffer_load, buffer_store, buffer_lds, s_load, s_store
"""

import argparse
import re
from pathlib import Path


ASTER_ROOT = Path(__file__).resolve().parent.parent


def find_mlir_files():
    """Find all .mlir files under the aster source tree."""
    dirs = [
        ASTER_ROOT / "test",
        ASTER_ROOT / "examples",
        ASTER_ROOT / "mlir_kernels",
        ASTER_ROOT / "contrib",
    ]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.mlir"))
    return sorted(files)


def find_py_files():
    """Find all .py files under the aster source tree."""
    dirs = [
        ASTER_ROOT / "python",
        ASTER_ROOT / "contrib",
    ]
    files = []
    for d in dirs:
        if d.exists():
            files.extend(d.rglob("*.py"))
    return sorted(files)


def join_continuation_lines(text):
    """Join multi-line MLIR instructions into single logical lines.

    Old format instructions can span multiple lines. We join lines that
    are continuations (indented lines that aren't new statements).
    Returns (joined_text, line_map) where line_map maps logical line
    indices back to original line numbers.
    """
    return text


# ---------------------------------------------------------------------------
# Helpers for parsing the old format
# ---------------------------------------------------------------------------


def parse_balanced(s, open_ch="(", close_ch=")"):
    """Parse a balanced expression starting at s[0] == open_ch.

    Returns the content between the delimiters and the rest of the
    string.
    """
    assert s[0] == open_ch, f"Expected '{open_ch}' but got '{s[0]}'"
    depth = 0
    for i, ch in enumerate(s):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return s[1:i], s[i + 1 :]
    raise ValueError(f"Unbalanced {open_ch}{close_ch} in: {s[:80]}")


# ---------------------------------------------------------------------------
# The core regex that matches old-format load/store/load_lds instructions.
# These can span multiple lines. We use re.DOTALL to match across lines.
# ---------------------------------------------------------------------------

# Pattern for matching an old-format load instruction.
# Captures:
#   1. result_binding: e.g. "%result, %tok = " or "%tok = "  (the SSA bindings)
#   2. op_kind: "load" or "store" or "load_lds"
#   3. opcode: e.g. "buffer_load_dword", "ds_read_b32", etc.
#   4. rest: everything after the opcode until the token type at the end
#
# The full instruction ends with "-> !amdgcn.(read|write)_token<...>"

OLD_LOAD_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"((?:%[\w.]+(?:,\s*%[\w.]+)*\s*=\s*)?)"  # optional result bindings
    r"(?:amdgcn\.)?load\s+"  # the op name (with or without dialect prefix)
    r"([\w]+)"  # opcode (e.g. buffer_load_dword)
    r"\s+dest\s+"  # 'dest' keyword
    r"(%[\w.]+)"  # dest operand
    r"\s+addr\s+"  # 'addr' keyword
    r"(%[\w.]+)"  # addr operand
    r"(.*?)"  # offset clause + attr-dict + types
    r"(->\s*!amdgcn\.\w+_token<\w+>)",  # trailing token type
    re.DOTALL,
)

OLD_STORE_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"((?:%[\w.]+\s*=\s*)?)"  # optional result binding (token)
    r"(?:amdgcn\.)?store\s+"  # the op name (with or without dialect prefix)
    r"([\w]+)"  # opcode
    r"\s+data\s+"  # 'data' keyword
    r"(%[\w.]+)"  # data operand
    r"\s+addr\s+"  # 'addr' keyword
    r"(%[\w.]+)"  # addr operand
    r"(.*?)"  # offset clause + attr-dict + types
    r"(->\s*!amdgcn\.\w+_token<\w+>)",  # trailing token type
    re.DOTALL,
)

OLD_LOAD_LDS_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"((?:%[\w.]+\s*=\s*)?)"  # optional result binding (token)
    r"(?:amdgcn\.)?load_lds\s+"  # the op name (with or without dialect prefix)
    r"([\w]+)"  # opcode (e.g. buffer_load_dword_lds)
    r"\s+m0\s+"  # 'm0' keyword
    r"(%[\w.]+)"  # m0 operand
    r"\s+addr\s+"  # 'addr' keyword
    r"(%[\w.]+)"  # addr operand
    r"(.*?)"  # offset clause + attr-dict + types
    r"(->\s*!amdgcn\.\w+_token<\w+>)",  # trailing token type
    re.DOTALL,
)


def parse_offset_clause(text):
    """Parse the offset clause: offset u(%x) + d(%y) + c(%z)

    Returns (uniform, dynamic, constant, remaining_text).
    Each can be None if not present.
    """
    text = text.strip()
    uniform = None
    dynamic = None
    constant = None

    if not text.startswith("offset"):
        return uniform, dynamic, constant, text

    text = text[len("offset") :].strip()

    ssa = r"(?:%[\w.]+|%\[\[[^\]]+\]\])"
    parts_re = re.compile(
        r"(?:u\((" + ssa + r")\)\s*\+?\s*)?"
        r"(?:d\((" + ssa + r")\)\s*\+?\s*)?"
        r"(?:c\((" + ssa + r")\))?"
    )
    m = parts_re.match(text)
    if m:
        uniform = m.group(1)
        dynamic = m.group(2)
        constant = m.group(3)
        text = text[m.end() :].strip()

    return uniform, dynamic, constant, text


def parse_attr_dict(text):
    """Strip an optional attr-dict ({...}) from the beginning of text.

    Returns (attr_dict_str, remaining_text). attr_dict_str is empty if
    no attr-dict is present.
    """
    text = text.strip()
    if not text.startswith("{"):
        return "", text
    content, rest = parse_balanced(text, "{", "}")
    return "{" + content + "}", rest.strip()


def merge_flags_and_attr_dict(flags, attr_dict):
    """Merge new instruction flags (offen, idxen) with an existing attr-dict.

    Returns a combined attr-dict string like ' {offen, sched.stage = 1 :
    i32}'.
    """
    all_entries = list(flags)
    if attr_dict:
        inner = attr_dict.strip()
        if inner.startswith("{") and inner.endswith("}"):
            inner = inner[1:-1].strip()
        if inner:
            all_entries.append(inner)
    if not all_entries:
        return ""
    return " {" + ", ".join(all_entries) + "}"


def parse_type_section(text):
    """Parse the type section after ':'.

    Old format: ': dps(out_type) ins(in_types...) -> token_type'
    or:         ': ins(in_types...) -> token_type'

    Returns (out_type_str, in_types_str, remaining).
    """
    text = text.strip()

    # Skip attr-dict if present before the colon.
    if text.startswith("{"):
        _, text = parse_attr_dict(text)
        text = text.strip()

    if text.startswith(":"):
        text = text[1:].strip()

    out_type = None
    in_types = None

    if text.startswith("dps"):
        text = text[3:].strip()
        content, text = parse_balanced(text)
        out_type = content.strip()
        text = text.strip()

    if text.startswith("ins"):
        text = text[3:].strip()
        content, text = parse_balanced(text)
        in_types = content.strip()
        text = text.strip()

    return out_type, in_types, text


def extract_instruction_block(lines, start_idx):
    """Extract a multi-line instruction starting at start_idx.

    Returns (full_text, end_idx) where end_idx is the last line
    consumed. Instructions end when we find the token type pattern.
    """
    token_end_re = re.compile(r"->\s*!amdgcn\.\w+_token<\w+>")
    result = []
    idx = start_idx
    while idx < len(lines):
        result.append(lines[idx])
        if token_end_re.search(lines[idx]):
            return "\n".join(result), idx
        idx += 1
    return "\n".join(result), idx - 1


# ---------------------------------------------------------------------------
# Migration functions for each category
# ---------------------------------------------------------------------------


def get_opcode_category(opcode):
    """Determine which category an opcode belongs to."""
    if opcode.startswith("ds_read_b"):
        return "ds_read"
    if opcode.startswith("ds_write_b"):
        return "ds_write"
    if opcode in ("ds_permute_b32", "ds_bpermute_b32"):
        return "ds_perm"
    if opcode.startswith("global_load_dword"):
        return "global_load"
    if opcode.startswith("global_store_dword"):
        return "global_store"
    if opcode.startswith("buffer_load_dword") and "_idxen" in opcode:
        return "buffer_load"
    if opcode.startswith("buffer_load_dword"):
        return "buffer_load"
    if opcode.startswith("buffer_store_dword") and "_idxen" in opcode:
        return "buffer_store"
    if opcode.startswith("buffer_store_dword"):
        return "buffer_store"
    if opcode.startswith("buffer_load_dword") and "_lds" in opcode:
        return "buffer_lds"
    if opcode.startswith("s_load_dword"):
        return "s_load"
    if opcode.startswith("s_store_dword"):
        return "s_store"
    return None


def split_in_types(in_types_str):
    """Split a comma-separated list of types, respecting angle brackets."""
    result = []
    depth = 0
    current = []
    for ch in in_types_str:
        if ch in "<([":
            depth += 1
            current.append(ch)
        elif ch in ">)]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        s = "".join(current).strip()
        if s:
            result.append(s)
    return result


# ---------------------------------------------------------------------------
# DS Read migration
# ---------------------------------------------------------------------------


def ensure_const_offset(constant, in_types, has_const_in_types, indent, counter_holder):
    """Ensure a constant offset is available.

    Returns (constant_ssa, const_type, prefix_lines). prefix_lines is
    empty if the constant already existed.
    """
    if constant:
        const_type = in_types[-1] if has_const_in_types else "i32"
        return constant, const_type, ""

    counter_holder[0] += 1
    name = f"%c0_i32_mig{counter_holder[0]}"
    prefix = f"{indent}{name} = arith.constant 0 : i32\n"
    return name, "i32", prefix


def migrate_ds_read(
    indent,
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.load ds_read_bXX to amdgcn.ds_read_bXX."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    addr_type = in_types[0] if in_types else ""
    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, len(in_types) > 1, indent, counter_holder
    )

    body = f"{indent}amdgcn.{opcode} outs({dest}) ins({addr}) args({constant})"
    if attr_dict:
        body += f" {attr_dict}"
    body += f"\n{indent}    : outs({out_type}) ins({addr_type}) args({const_type})"
    body += f" {token_type}"

    return prefix, body


# ---------------------------------------------------------------------------
# DS Write migration
# ---------------------------------------------------------------------------


def migrate_ds_write(
    indent,
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.store ds_write_bXX to amdgcn.ds_write_bXX."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    data_type = in_types[0] if len(in_types) > 0 else ""
    addr_type = in_types[1] if len(in_types) > 1 else ""
    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, len(in_types) > 2, indent, counter_holder
    )

    body = f"{indent}amdgcn.{opcode} ins({addr}, {data}) args({constant})"
    if attr_dict:
        body += f" {attr_dict}"
    body += f"\n{indent}    : ins({addr_type}, {data_type}) args({const_type})"
    body += f" {token_type}"

    return prefix, body


# ---------------------------------------------------------------------------
# DS Permute migration
# ---------------------------------------------------------------------------


def migrate_ds_perm(
    indent,
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.load ds_(b)permute_b32 to amdgcn.ds_(b)permute_b32."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    addr_type = in_types[0] if len(in_types) > 0 else ""
    data_type = in_types[1] if len(in_types) > 1 else ""
    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, len(in_types) > 2, indent, counter_holder
    )

    new_line = (
        f"{indent}amdgcn.{opcode} outs({dest}) ins({addr}, {dynamic}) args({constant})"
    )
    if attr_dict:
        new_line += f" {attr_dict}"
    new_line += f"\n{indent}    : outs({out_type}) ins({addr_type}, {data_type}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# Global Load migration
# ---------------------------------------------------------------------------


def migrate_global_load(
    indent,
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.load global_load_dwordXX."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    addr_type = in_types[0] if len(in_types) > 0 else ""

    ins_vals = [addr]
    ins_types = [addr_type]

    if dynamic:
        ins_vals.append(f"offset = {dynamic}")
        if len(in_types) > 1 and not in_types[1].startswith("i"):
            ins_types.append(f"offset = {in_types[1]}")
            offset_type_idx = 2
        else:
            offset_type_idx = 1
    else:
        offset_type_idx = 1

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, offset_type_idx < len(in_types), indent, counter_holder
    )

    new_line = f"{indent}amdgcn.{opcode} outs({dest}) ins({', '.join(ins_vals)}) args({constant})"
    if attr_dict:
        new_line += f" {attr_dict}"
    new_line += f"\n{indent}    : outs({out_type}) ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# Global Store migration
# ---------------------------------------------------------------------------


def migrate_global_store(
    indent,
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.store global_store_dwordXX."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    data_type = in_types[0] if len(in_types) > 0 else ""
    addr_type = in_types[1] if len(in_types) > 1 else ""

    ins_vals = [data, addr]
    ins_types = [data_type, addr_type]

    if dynamic:
        ins_vals.append(f"offset = {dynamic}")
        if len(in_types) > 2 and not in_types[2].startswith("i"):
            ins_types.append(f"offset = {in_types[2]}")
            offset_type_idx = 3
        else:
            offset_type_idx = 2
    else:
        offset_type_idx = 2

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, offset_type_idx < len(in_types), indent, counter_holder
    )

    new_line = f"{indent}amdgcn.{opcode} ins({', '.join(ins_vals)}) args({constant})"
    if attr_dict:
        new_line += f" {attr_dict}"
    new_line += f"\n{indent}    : ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# Buffer Load migration
# ---------------------------------------------------------------------------


def migrate_buffer_load(
    indent,
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.load buffer_load_dwordXX[_idxen]."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    is_idxen = opcode.endswith("_idxen")
    base_opcode = opcode.replace("_idxen", "") if is_idxen else opcode

    rsrc_type = in_types[0] if len(in_types) > 0 else ""

    flags = []
    ins_vals = [addr]
    ins_types = [rsrc_type]

    type_idx = 1
    if dynamic:
        ins_vals.append(f"off_or_idx = {dynamic}")
        if uniform:
            uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
            dynamic_type = (
                in_types[type_idx + 1] if type_idx + 1 < len(in_types) else ""
            )
            ins_types.append(f"off_or_idx = {dynamic_type}")
            type_idx += 2
        else:
            dynamic_type = in_types[type_idx] if type_idx < len(in_types) else ""
            ins_types.append(f"off_or_idx = {dynamic_type}")
            type_idx += 1
    if uniform:
        ins_vals.append(uniform)
        if not dynamic:
            uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
            type_idx += 1
        ins_types.append(uniform_type)

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, type_idx < len(in_types), indent, counter_holder
    )

    if is_idxen:
        flags.append("idxen")
    elif dynamic:
        flags.append("offen")

    flag_str = merge_flags_and_attr_dict(flags, attr_dict)

    new_line = f"{indent}amdgcn.{base_opcode} outs({dest}) ins({', '.join(ins_vals)}) args({constant})"
    new_line += flag_str
    new_line += f"\n{indent}    : outs({out_type}) ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# Buffer Store migration
# ---------------------------------------------------------------------------


def migrate_buffer_store(
    indent,
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.store buffer_store_dwordXX[_idxen]."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    is_idxen = opcode.endswith("_idxen")
    base_opcode = opcode.replace("_idxen", "") if is_idxen else opcode

    data_type = in_types[0] if len(in_types) > 0 else ""
    rsrc_type = in_types[1] if len(in_types) > 1 else ""

    flags = []
    ins_vals = [data, addr]
    ins_types = [data_type, rsrc_type]

    type_idx = 2
    if dynamic:
        ins_vals.append(f"off_or_idx = {dynamic}")
        if uniform:
            uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
            dynamic_type = (
                in_types[type_idx + 1] if type_idx + 1 < len(in_types) else ""
            )
            ins_types.append(f"off_or_idx = {dynamic_type}")
            type_idx += 2
        else:
            dynamic_type = in_types[type_idx] if type_idx < len(in_types) else ""
            ins_types.append(f"off_or_idx = {dynamic_type}")
            type_idx += 1
    if uniform:
        ins_vals.append(uniform)
        if not dynamic:
            uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
            type_idx += 1
        ins_types.append(uniform_type)

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, type_idx < len(in_types), indent, counter_holder
    )

    if is_idxen:
        flags.append("idxen")
    elif dynamic:
        flags.append("offen")

    flag_str = merge_flags_and_attr_dict(flags, attr_dict)

    new_line = (
        f"{indent}amdgcn.{base_opcode} ins({', '.join(ins_vals)}) args({constant})"
    )
    new_line += flag_str
    new_line += f"\n{indent}    : ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# Buffer Load LDS migration
# ---------------------------------------------------------------------------


def migrate_buffer_lds(
    indent,
    result_binding,
    opcode,
    m0,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.load_lds buffer_load_dword(x4)_lds."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    base = opcode.replace("_lds", "")
    parts = base.split("_", 2)
    new_opcode = f"{parts[0]}_{parts[1]}_lds_{parts[2]}"

    m0_type = in_types[0] if len(in_types) > 0 else ""
    rsrc_type = in_types[1] if len(in_types) > 1 else ""

    flags = []
    ins_vals = [addr]
    ins_types = [rsrc_type]

    type_idx = 2
    if dynamic:
        ins_vals.append(f"off_or_idx = {dynamic}")
        if uniform:
            uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
            dynamic_type = (
                in_types[type_idx + 1] if type_idx + 1 < len(in_types) else ""
            )
            ins_types.append(f"off_or_idx = {dynamic_type}")
            type_idx += 2
        else:
            dynamic_type = in_types[type_idx] if type_idx < len(in_types) else ""
            ins_types.append(f"off_or_idx = {dynamic_type}")
            type_idx += 1
        flags.append("offen")

    if uniform:
        ins_vals.append(uniform)
        if not dynamic:
            uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
            type_idx += 1
        ins_types.append(uniform_type)

    ins_vals.append(m0)
    ins_types.append(m0_type)

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, type_idx < len(in_types), indent, counter_holder
    )

    flag_str = merge_flags_and_attr_dict(flags, attr_dict)

    new_line = (
        f"{indent}amdgcn.{new_opcode} ins({', '.join(ins_vals)}) args({constant})"
    )
    new_line += flag_str
    new_line += f"\n{indent}    : ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# S_Load migration
# ---------------------------------------------------------------------------


def migrate_s_load(
    indent,
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.load s_load_dwordXX."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    addr_type = in_types[0] if len(in_types) > 0 else ""

    ins_vals = [addr]
    ins_types = [addr_type]

    type_idx = 1
    if uniform:
        ins_vals.append(uniform)
        uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
        ins_types.append(uniform_type)
        type_idx += 1

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, type_idx < len(in_types), indent, counter_holder
    )

    new_line = f"{indent}amdgcn.{opcode} outs({dest}) ins({', '.join(ins_vals)}) args({constant})"
    if attr_dict:
        new_line += f" {attr_dict}"
    new_line += f"\n{indent}    : outs({out_type}) ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# S_Store migration
# ---------------------------------------------------------------------------


def migrate_s_store(
    indent,
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
    counter_holder,
):
    """Migrate amdgcn.store s_store_dwordXX."""
    in_types = split_in_types(in_types_str) if in_types_str else []

    data_type = in_types[0] if len(in_types) > 0 else ""
    addr_type = in_types[1] if len(in_types) > 1 else ""

    ins_vals = [data, addr]
    ins_types = [data_type, addr_type]

    type_idx = 2
    if uniform:
        ins_vals.append(uniform)
        uniform_type = in_types[type_idx] if type_idx < len(in_types) else ""
        ins_types.append(uniform_type)
        type_idx += 1

    constant, const_type, prefix = ensure_const_offset(
        constant, in_types, type_idx < len(in_types), indent, counter_holder
    )

    new_line = f"{indent}amdgcn.{opcode} ins({', '.join(ins_vals)}) args({constant})"
    if attr_dict:
        new_line += f" {attr_dict}"
    new_line += f"\n{indent}    : ins({', '.join(ins_types)}) args({const_type})"
    new_line += f" {token_type}"

    return prefix, new_line


# ---------------------------------------------------------------------------
# Main line-by-line processing
# ---------------------------------------------------------------------------


def parse_result_bindings(binding_str):
    """Parse result bindings like '%result, %tok = ' or '%tok = '.

    Returns a list of SSA names (without the leading %).
    """
    binding_str = binding_str.strip()
    if not binding_str:
        return []
    if binding_str.endswith("="):
        binding_str = binding_str[:-1].strip()
    parts = [p.strip() for p in binding_str.split(",")]
    return parts


SSA_PAT = r"(?:%[\w.]+|%\[\[[^\]]+\]\])"


def process_load_instruction(full_text, category, counter_holder):
    """Process an old-format amdgcn.load instruction.

    Returns (new_text, replacements) where replacements is a dict
    mapping old SSA names to their replacement SSA names.
    """
    indent_match = re.match(r"^(\s*)", full_text)
    indent = indent_match.group(1) if indent_match else ""

    normalized = re.sub(r"\s+", " ", full_text).strip()

    load_re = re.compile(
        r"((?:" + SSA_PAT + r"(?:,\s*" + SSA_PAT + r")*\s*=\s*)?)"
        r"(?:amdgcn\.)?load\s+"
        r"([\w]+)\s+"
        r"dest\s+(" + SSA_PAT + r")\s+"
        r"addr\s+(" + SSA_PAT + r")\s*"
        r"(.*)"
    )
    m = load_re.match(normalized)
    if not m:
        return None

    result_binding = m.group(1).strip()
    if result_binding and not result_binding.endswith(" "):
        result_binding += " "
    opcode = m.group(2)
    dest = m.group(3)
    addr = m.group(4)
    rest = m.group(5).strip()

    uniform, dynamic, constant, rest = parse_offset_clause(rest)
    attr_dict, rest = parse_attr_dict(rest)
    out_type, in_types_str, rest = parse_type_section(rest)
    token_match = re.search(r"(->\s*!amdgcn\.\w+_token<\w+>)", rest)
    token_type = token_match.group(1) if token_match else ""

    op_cat = get_opcode_category(opcode)
    if op_cat != category:
        return None

    if category == "ds_read":
        body = migrate_ds_read(
            indent,
            result_binding,
            opcode,
            dest,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            out_type,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "ds_perm":
        body = migrate_ds_perm(
            indent,
            result_binding,
            opcode,
            dest,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            out_type,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "global_load":
        body = migrate_global_load(
            indent,
            result_binding,
            opcode,
            dest,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            out_type,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "buffer_load":
        body = migrate_buffer_load(
            indent,
            result_binding,
            opcode,
            dest,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            out_type,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "s_load":
        body = migrate_s_load(
            indent,
            result_binding,
            opcode,
            dest,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            out_type,
            in_types_str,
            token_type,
            counter_holder,
        )
    else:
        return None

    prefix, body = body
    body_stripped = body.lstrip()
    new_text = f"{prefix}{indent}{result_binding}{body_stripped}"

    return new_text, {}


def process_store_instruction(full_text, category, counter_holder):
    """Process an old-format amdgcn.store instruction.

    Returns (new_text, replacements).
    """
    indent_match = re.match(r"^(\s*)", full_text)
    indent = indent_match.group(1) if indent_match else ""

    normalized = re.sub(r"\s+", " ", full_text).strip()

    store_re = re.compile(
        r"((?:" + SSA_PAT + r"\s*=\s*)?)"
        r"(?:amdgcn\.)?store\s+"
        r"([\w]+)\s+"
        r"data\s+(" + SSA_PAT + r")\s+"
        r"addr\s+(" + SSA_PAT + r")\s*"
        r"(.*)"
    )
    m = store_re.match(normalized)
    if not m:
        return None

    result_binding = m.group(1).strip()
    if result_binding and not result_binding.endswith(" "):
        result_binding += " "
    opcode = m.group(2)
    data = m.group(3)
    addr = m.group(4)
    rest = m.group(5).strip()

    uniform, dynamic, constant, rest = parse_offset_clause(rest)
    attr_dict, rest = parse_attr_dict(rest)
    out_type, in_types_str, rest = parse_type_section(rest)
    token_match = re.search(r"(->\s*!amdgcn\.\w+_token<\w+>)", rest)
    token_type = token_match.group(1) if token_match else ""

    op_cat = get_opcode_category(opcode)
    if op_cat != category:
        return None

    if category == "ds_write":
        body = migrate_ds_write(
            indent,
            result_binding,
            opcode,
            data,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "global_store":
        body = migrate_global_store(
            indent,
            result_binding,
            opcode,
            data,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "buffer_store":
        body = migrate_buffer_store(
            indent,
            result_binding,
            opcode,
            data,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            in_types_str,
            token_type,
            counter_holder,
        )
    elif category == "s_store":
        body = migrate_s_store(
            indent,
            result_binding,
            opcode,
            data,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            in_types_str,
            token_type,
            counter_holder,
        )
    else:
        return None

    prefix, body = body
    body_stripped = body.lstrip()
    new_text = f"{prefix}{indent}{result_binding}{body_stripped}"

    return new_text, {}


def process_load_lds_instruction(full_text, category, counter_holder):
    """Process an old-format amdgcn.load_lds instruction.

    Returns (new_text, replacements).
    """
    if category != "buffer_lds":
        return None

    indent_match = re.match(r"^(\s*)", full_text)
    indent = indent_match.group(1) if indent_match else ""

    normalized = re.sub(r"\s+", " ", full_text).strip()

    lds_re = re.compile(
        r"((?:" + SSA_PAT + r"\s*=\s*)?)"
        r"(?:amdgcn\.)?load_lds\s+"
        r"([\w]+)\s+"
        r"m0\s+(" + SSA_PAT + r")\s+"
        r"addr\s+(" + SSA_PAT + r")\s*"
        r"(.*)"
    )
    m = lds_re.match(normalized)
    if not m:
        return None

    result_binding = m.group(1).strip()
    if result_binding and not result_binding.endswith(" "):
        result_binding += " "
    opcode = m.group(2)
    m0 = m.group(3)
    addr = m.group(4)
    rest = m.group(5).strip()

    uniform, dynamic, constant, rest = parse_offset_clause(rest)
    attr_dict, rest = parse_attr_dict(rest)
    out_type, in_types_str, rest = parse_type_section(rest)
    token_match = re.search(r"(->\s*!amdgcn\.\w+_token<\w+>)", rest)
    token_type = token_match.group(1) if token_match else ""

    prefix, body = migrate_buffer_lds(
        indent,
        result_binding,
        opcode,
        m0,
        addr,
        uniform,
        dynamic,
        constant,
        attr_dict,
        in_types_str,
        token_type,
        counter_holder,
    )

    body_stripped = body.lstrip()
    new_text = f"{prefix}{indent}{result_binding}{body_stripped}"

    return new_text, {}


CHECK_PREFIX_RE = re.compile(
    r"^(\s*//\s*CHECK(?:-NEXT|-SAME|-NOT|-DAG|-LABEL)?(?:\([^)]*\))?\s*:\s*)(.*)"
)

# Map from old opcode keyword patterns to new op names for simple CHECK lines.
# The old format "amdgcn.load <opcode>" becomes "amdgcn.<opcode>",
# "amdgcn.store <opcode>" becomes "amdgcn.<opcode>",
# and "_idxen" variants lose the suffix.
SIMPLE_CHECK_RENAMES = {}

# Build rename tables for both qualified (amdgcn.load) and unqualified (load) forms.
_RENAME_ENTRIES = [
    ("load ds_read_b32", "amdgcn.ds_read_b32"),
    ("load ds_read_b64", "amdgcn.ds_read_b64"),
    ("load ds_read_b96", "amdgcn.ds_read_b96"),
    ("load ds_read_b128", "amdgcn.ds_read_b128"),
    ("store ds_write_b32", "amdgcn.ds_write_b32"),
    ("store ds_write_b64", "amdgcn.ds_write_b64"),
    ("store ds_write_b96", "amdgcn.ds_write_b96"),
    ("store ds_write_b128", "amdgcn.ds_write_b128"),
    ("load ds_permute_b32", "amdgcn.ds_permute_b32"),
    ("load ds_bpermute_b32", "amdgcn.ds_bpermute_b32"),
    ("load global_load_dword ", "amdgcn.global_load_dword "),
    ("load global_load_dwordx2 ", "amdgcn.global_load_dwordx2 "),
    ("load global_load_dwordx3 ", "amdgcn.global_load_dwordx3 "),
    ("load global_load_dwordx4 ", "amdgcn.global_load_dwordx4 "),
    ("store global_store_dword ", "amdgcn.global_store_dword "),
    ("store global_store_dwordx2 ", "amdgcn.global_store_dwordx2 "),
    ("store global_store_dwordx3 ", "amdgcn.global_store_dwordx3 "),
    ("store global_store_dwordx4 ", "amdgcn.global_store_dwordx4 "),
    ("load buffer_load_dword_idxen", "amdgcn.buffer_load_dword"),
    ("load buffer_load_dwordx2_idxen", "amdgcn.buffer_load_dwordx2"),
    ("load buffer_load_dwordx3_idxen", "amdgcn.buffer_load_dwordx3"),
    ("load buffer_load_dwordx4_idxen", "amdgcn.buffer_load_dwordx4"),
    ("load buffer_load_dword ", "amdgcn.buffer_load_dword "),
    ("load buffer_load_dwordx2 ", "amdgcn.buffer_load_dwordx2 "),
    ("load buffer_load_dwordx3 ", "amdgcn.buffer_load_dwordx3 "),
    ("load buffer_load_dwordx4 ", "amdgcn.buffer_load_dwordx4 "),
    ("store buffer_store_dword_idxen", "amdgcn.buffer_store_dword"),
    ("store buffer_store_dwordx2_idxen", "amdgcn.buffer_store_dwordx2"),
    ("store buffer_store_dwordx3_idxen", "amdgcn.buffer_store_dwordx3"),
    ("store buffer_store_dwordx4_idxen", "amdgcn.buffer_store_dwordx4"),
    ("store buffer_store_dword ", "amdgcn.buffer_store_dword "),
    ("store buffer_store_dwordx2 ", "amdgcn.buffer_store_dwordx2 "),
    ("store buffer_store_dwordx3 ", "amdgcn.buffer_store_dwordx3 "),
    ("store buffer_store_dwordx4 ", "amdgcn.buffer_store_dwordx4 "),
    ("load_lds buffer_load_dword_lds", "amdgcn.buffer_load_lds_dword"),
    ("load_lds buffer_load_dwordx4_lds", "amdgcn.buffer_load_lds_dwordx4"),
    ("load s_load_dword ", "amdgcn.s_load_dword "),
    ("load s_load_dwordx2 ", "amdgcn.s_load_dwordx2 "),
    ("load s_load_dwordx4 ", "amdgcn.s_load_dwordx4 "),
    ("load s_load_dwordx8 ", "amdgcn.s_load_dwordx8 "),
    ("load s_load_dwordx16 ", "amdgcn.s_load_dwordx16 "),
    ("store s_store_dword ", "amdgcn.s_store_dword "),
    ("store s_store_dwordx2 ", "amdgcn.s_store_dwordx2 "),
    ("store s_store_dwordx4 ", "amdgcn.s_store_dwordx4 "),
]
for _old_suffix, _new in _RENAME_ENTRIES:
    SIMPLE_CHECK_RENAMES["amdgcn." + _old_suffix] = _new
    SIMPLE_CHECK_RENAMES[_old_suffix] = _new

# Map category to list of old patterns that belong to this category.
# Include both qualified (amdgcn.load) and unqualified (load) forms.
CATEGORY_PATTERNS = {
    "ds_read": ["amdgcn.load ds_read_b", "load ds_read_b"],
    "ds_write": ["amdgcn.store ds_write_b", "store ds_write_b"],
    "ds_perm": [
        "amdgcn.load ds_permute_b32",
        "amdgcn.load ds_bpermute_b32",
        "load ds_permute_b32",
        "load ds_bpermute_b32",
    ],
    "global_load": ["amdgcn.load global_load_dword", "load global_load_dword"],
    "global_store": ["amdgcn.store global_store_dword", "store global_store_dword"],
    "buffer_load": ["amdgcn.load buffer_load_dword", "load buffer_load_dword"],
    "buffer_store": ["amdgcn.store buffer_store_dword", "store buffer_store_dword"],
    "buffer_lds": ["amdgcn.load_lds buffer_load_dword", "load_lds buffer_load_dword"],
    "s_load": ["amdgcn.load s_load_dword", "load s_load_dword"],
    "s_store": ["amdgcn.store s_store_dword", "store s_store_dword"],
}


def extract_check_instruction_block(lines, start_idx):
    """Extract a multi-line CHECK instruction.

    CHECK instructions can span multiple CHECK-NEXT lines for multi-line
    ops. Returns (check_prefix, full_mlir_text, end_idx). Only joins
    subsequent CHECK-NEXT lines if they contain continuation of the type
    section (starting with ':', 'dps', 'ins', or '->').
    """
    m = CHECK_PREFIX_RE.match(lines[start_idx])
    if not m:
        return None, None, start_idx

    first_prefix = m.group(1)
    parts = [m.group(2)]
    token_end_re = re.compile(r"->\s*!amdgcn\.\w+_token<\w+>")

    if token_end_re.search(m.group(2)):
        return first_prefix, m.group(2), start_idx

    # Only join continuation lines that look like type sections.
    continuation_re = re.compile(r"^\s*(?::|dps\(|ins\(|outs\(|args\(|->)")
    idx = start_idx + 1
    while idx < len(lines):
        cm = CHECK_PREFIX_RE.match(lines[idx])
        if not cm:
            break
        content = cm.group(2).strip()
        if not continuation_re.match(content):
            break
        parts.append(content)
        if token_end_re.search(content):
            return first_prefix, " ".join(parts), idx
        idx += 1

    # No token type found, return what we have (partial CHECK).
    return first_prefix, " ".join(parts).strip(), start_idx


def ensure_const_offset_check(constant, in_types, has_const_in_types):
    """Ensure a constant offset for CHECK lines.

    Unlike code lines, CHECK lines use FileCheck wildcards instead of
    generating arith.constant definitions. Returns (constant_ssa,
    const_type).
    """
    if constant:
        const_type = in_types[-1] if has_const_in_types else "i32"
        return constant, const_type
    return "%{{.*}}", "i32"


def migrate_check_ds_read(
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for ds_read."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(out_type or in_types_str or token_type)

    constant, const_type = ensure_const_offset_check(
        constant, in_types, len(in_types) > 1
    )

    if has_types:
        addr_type = in_types[0] if in_types else ""
        body = (
            f"{result_binding}amdgcn.{opcode} outs({dest}) ins({addr}) args({constant})"
        )
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : outs({out_type}) ins({addr_type}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{opcode} outs({dest}) ins({addr})"
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_ds_write(
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for ds_write."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(in_types_str or token_type)

    constant, const_type = ensure_const_offset_check(
        constant, in_types, len(in_types) > 2
    )

    if has_types:
        data_type = in_types[0] if len(in_types) > 0 else ""
        addr_type = in_types[1] if len(in_types) > 1 else ""
        body = f"{result_binding}amdgcn.{opcode} ins({addr}, {data}) args({constant})"
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : ins({addr_type}, {data_type}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{opcode} ins({addr}, {data})"
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_ds_perm(
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for ds_permute."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(out_type or in_types_str or token_type)

    constant, const_type = ensure_const_offset_check(
        constant, in_types, len(in_types) > 2
    )

    if has_types:
        addr_type = in_types[0] if len(in_types) > 0 else ""
        data_type = in_types[1] if len(in_types) > 1 else ""
        body = f"{result_binding}amdgcn.{opcode} outs({dest}) ins({addr}, {dynamic}) args({constant})"
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : outs({out_type}) ins({addr_type}, {data_type}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{opcode} outs({dest}) ins({addr}, {dynamic})"
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_global_load(
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for global_load."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(out_type or in_types_str or token_type)

    ins_vals = [addr]
    offset_type_idx = 1

    if dynamic:
        ins_vals.append(f"offset = {dynamic}")
        if len(in_types) > 1 and not in_types[1].startswith("i"):
            offset_type_idx = 2
        else:
            offset_type_idx = 1

    constant, const_type = ensure_const_offset_check(
        constant, in_types, offset_type_idx < len(in_types)
    )

    if has_types:
        addr_type = in_types[0] if len(in_types) > 0 else ""
        ins_types = [addr_type]
        if dynamic and len(in_types) > 1 and not in_types[1].startswith("i"):
            ins_types.append(f"offset = {in_types[1]}")

        body = f"{result_binding}amdgcn.{opcode} outs({dest}) ins({', '.join(ins_vals)}) args({constant})"
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : outs({out_type}) ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = (
            f"{result_binding}amdgcn.{opcode} outs({dest}) ins({', '.join(ins_vals)})"
        )
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_global_store(
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for global_store."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(in_types_str or token_type)

    ins_vals = [data, addr]
    offset_type_idx = 2

    if dynamic:
        ins_vals.append(f"offset = {dynamic}")
        if len(in_types) > 2 and not in_types[2].startswith("i"):
            offset_type_idx = 3
        else:
            offset_type_idx = 2

    constant, const_type = ensure_const_offset_check(
        constant, in_types, offset_type_idx < len(in_types)
    )

    if has_types:
        data_type = in_types[0] if len(in_types) > 0 else ""
        addr_type = in_types[1] if len(in_types) > 1 else ""
        ins_types = [data_type, addr_type]
        if dynamic and len(in_types) > 2 and not in_types[2].startswith("i"):
            ins_types.append(f"offset = {in_types[2]}")

        body = f"{result_binding}amdgcn.{opcode} ins({', '.join(ins_vals)}) args({constant})"
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{opcode} ins({', '.join(ins_vals)})"
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_buffer_load(
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for buffer_load."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(out_type or in_types_str or token_type)
    is_idxen = opcode.endswith("_idxen")
    base_opcode = opcode.replace("_idxen", "") if is_idxen else opcode

    flags = []
    ins_vals = [addr]
    type_idx = 1

    if dynamic:
        ins_vals.append(f"off_or_idx = {dynamic}")
        if uniform:
            type_idx += 2
        else:
            type_idx += 1
    if uniform:
        ins_vals.append(uniform)
        if not dynamic:
            type_idx += 1

    constant, const_type = ensure_const_offset_check(
        constant, in_types, type_idx < len(in_types)
    )

    if is_idxen:
        flags.append("idxen")
    elif dynamic:
        flags.append("offen")

    flag_str = merge_flags_and_attr_dict(flags, attr_dict)

    if has_types:
        rsrc_type = in_types[0] if len(in_types) > 0 else ""
        ins_types = [rsrc_type]
        t_idx = 1
        if dynamic:
            if uniform:
                dynamic_type = in_types[t_idx + 1] if t_idx + 1 < len(in_types) else ""
                ins_types.append(f"off_or_idx = {dynamic_type}")
                t_idx += 2
            else:
                dynamic_type = in_types[t_idx] if t_idx < len(in_types) else ""
                ins_types.append(f"off_or_idx = {dynamic_type}")
                t_idx += 1
        if uniform:
            if not dynamic:
                uniform_type = in_types[t_idx] if t_idx < len(in_types) else ""
                t_idx += 1
            else:
                uniform_type = in_types[t_idx - 2] if dynamic else ""
            ins_types.append(uniform_type)

        body = f"{result_binding}amdgcn.{base_opcode} outs({dest}) ins({', '.join(ins_vals)}) args({constant})"
        body += flag_str
        body += f" : outs({out_type}) ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{base_opcode} outs({dest}) ins({', '.join(ins_vals)})"
        body += flag_str
    return body


def migrate_check_buffer_store(
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for buffer_store."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(in_types_str or token_type)
    is_idxen = opcode.endswith("_idxen")
    base_opcode = opcode.replace("_idxen", "") if is_idxen else opcode

    flags = []
    ins_vals = [data, addr]

    if dynamic:
        ins_vals.append(f"off_or_idx = {dynamic}")
    if uniform:
        ins_vals.append(uniform)

    constant, const_type = ensure_const_offset_check(constant, in_types, False)

    if is_idxen:
        flags.append("idxen")
    elif dynamic:
        flags.append("offen")

    flag_str = merge_flags_and_attr_dict(flags, attr_dict)

    if has_types:
        data_type = in_types[0] if len(in_types) > 0 else ""
        rsrc_type = in_types[1] if len(in_types) > 1 else ""
        ins_types = [data_type, rsrc_type]
        t_idx = 2
        if dynamic:
            if uniform:
                dynamic_type = in_types[t_idx + 1] if t_idx + 1 < len(in_types) else ""
                ins_types.append(f"off_or_idx = {dynamic_type}")
                t_idx += 2
            else:
                dynamic_type = in_types[t_idx] if t_idx < len(in_types) else ""
                ins_types.append(f"off_or_idx = {dynamic_type}")
                t_idx += 1
        if uniform:
            if not dynamic:
                uniform_type = in_types[t_idx] if t_idx < len(in_types) else ""
                t_idx += 1
            else:
                uniform_type = in_types[t_idx - 2] if dynamic else ""
            ins_types.append(uniform_type)

        body = f"{result_binding}amdgcn.{base_opcode} ins({', '.join(ins_vals)}) args({constant})"
        body += flag_str
        body += f" : ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{base_opcode} ins({', '.join(ins_vals)})"
        body += flag_str
    return body


def migrate_check_buffer_lds(
    result_binding,
    opcode,
    m0,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for buffer_load_lds."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(in_types_str or token_type)
    base = opcode.replace("_lds", "")
    parts = base.split("_", 2)
    new_opcode = f"{parts[0]}_{parts[1]}_lds_{parts[2]}"

    flags = []
    ins_vals = [addr]

    if dynamic:
        ins_vals.append(f"off_or_idx = {dynamic}")
        flags.append("offen")
    if uniform:
        ins_vals.append(uniform)
    ins_vals.append(m0)

    constant, const_type = ensure_const_offset_check(constant, in_types, False)

    flag_str = merge_flags_and_attr_dict(flags, attr_dict)

    if has_types:
        m0_type = in_types[0] if len(in_types) > 0 else ""
        rsrc_type = in_types[1] if len(in_types) > 1 else ""
        ins_types = [rsrc_type]
        t_idx = 2
        if dynamic:
            if uniform:
                dynamic_type = in_types[t_idx + 1] if t_idx + 1 < len(in_types) else ""
                ins_types.append(f"off_or_idx = {dynamic_type}")
                t_idx += 2
            else:
                dynamic_type = in_types[t_idx] if t_idx < len(in_types) else ""
                ins_types.append(f"off_or_idx = {dynamic_type}")
                t_idx += 1
        if uniform:
            if not dynamic:
                uniform_type = in_types[t_idx] if t_idx < len(in_types) else ""
                t_idx += 1
            else:
                uniform_type = in_types[t_idx - 2] if dynamic else ""
            ins_types.append(uniform_type)
        ins_types.append(m0_type)

        body = f"{result_binding}amdgcn.{new_opcode} ins({', '.join(ins_vals)}) args({constant})"
        body += flag_str
        body += f" : ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{new_opcode} ins({', '.join(ins_vals)})"
        body += flag_str
    return body


def migrate_check_s_load(
    result_binding,
    opcode,
    dest,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    out_type,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for s_load."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(out_type or in_types_str or token_type)

    ins_vals = [addr]
    type_idx = 1
    if uniform:
        ins_vals.append(uniform)
        type_idx += 1

    constant, const_type = ensure_const_offset_check(
        constant, in_types, type_idx < len(in_types)
    )

    if has_types:
        addr_type = in_types[0] if len(in_types) > 0 else ""
        ins_types = [addr_type]
        if uniform:
            uniform_type = in_types[1] if len(in_types) > 1 else ""
            ins_types.append(uniform_type)

        body = f"{result_binding}amdgcn.{opcode} outs({dest}) ins({', '.join(ins_vals)}) args({constant})"
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : outs({out_type}) ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = (
            f"{result_binding}amdgcn.{opcode} outs({dest}) ins({', '.join(ins_vals)})"
        )
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_s_store(
    result_binding,
    opcode,
    data,
    addr,
    uniform,
    dynamic,
    constant,
    attr_dict,
    in_types_str,
    token_type,
):
    """Migrate CHECK line for s_store."""
    in_types = split_in_types(in_types_str) if in_types_str else []
    has_types = bool(in_types_str or token_type)

    ins_vals = [data, addr]
    if uniform:
        ins_vals.append(uniform)

    constant, const_type = ensure_const_offset_check(constant, in_types, False)

    if has_types:
        data_type = in_types[0] if len(in_types) > 0 else ""
        addr_type = in_types[1] if len(in_types) > 1 else ""
        ins_types = [data_type, addr_type]
        if uniform:
            uniform_type = in_types[2] if len(in_types) > 2 else ""
            ins_types.append(uniform_type)

        body = f"{result_binding}amdgcn.{opcode} ins({', '.join(ins_vals)}) args({constant})"
        if attr_dict:
            body += f" {attr_dict}"
        body += f" : ins({', '.join(ins_types)}) args({const_type}) {token_type}"
    else:
        body = f"{result_binding}amdgcn.{opcode} ins({', '.join(ins_vals)})"
        if attr_dict:
            body += f" {attr_dict}"
    return body


def migrate_check_instruction(check_prefix, full_text, category, counter_holder):
    """Migrate a CHECK line's instruction content.

    Returns new CHECK line(s) or None if no transformation needed.
    """
    # Strip any leading text prefix like "Op: " before the MLIR instruction.
    text_prefix = ""
    op_match = re.search(
        r"(?:(?:"
        + SSA_PAT
        + r"(?:,\s*"
        + SSA_PAT
        + r")*\s*=\s*)?(?:amdgcn\.)?(?:load|store|load_lds)\s)",
        full_text,
    )
    if op_match and op_match.start() > 0:
        text_prefix = full_text[: op_match.start()]
        full_text = full_text[op_match.start() :]

    normalized = re.sub(r"\s+", " ", full_text).strip()

    # Try to match as a load instruction.
    load_re = re.compile(
        r"((?:" + SSA_PAT + r"(?:,\s*" + SSA_PAT + r")*\s*=\s*)?)"
        r"(?:amdgcn\.)?load\s+"
        r"([\w]+)\s+"
        r"dest\s+(" + SSA_PAT + r")\s+"
        r"addr\s+(" + SSA_PAT + r")\s*"
        r"(.*)"
    )
    m = load_re.match(normalized)
    if m:
        result_binding = m.group(1).strip()
        if result_binding and not result_binding.endswith(" "):
            result_binding += " "
        opcode = m.group(2)
        dest = m.group(3)
        addr = m.group(4)
        rest = m.group(5).strip()

        uniform, dynamic, constant, rest = parse_offset_clause(rest)
        attr_dict, rest = parse_attr_dict(rest)
        out_type, in_types_str, rest = parse_type_section(rest)
        token_match = re.search(r"(->\s*!amdgcn\.\w+_token<\w+>)", rest)
        token_type = token_match.group(1) if token_match else ""

        op_cat = get_opcode_category(opcode)
        if op_cat != category:
            return None

        body = None
        if category == "ds_read":
            body = migrate_check_ds_read(
                result_binding,
                opcode,
                dest,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                out_type,
                in_types_str,
                token_type,
            )
        elif category == "ds_perm":
            body = migrate_check_ds_perm(
                result_binding,
                opcode,
                dest,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                out_type,
                in_types_str,
                token_type,
            )
        elif category == "global_load":
            body = migrate_check_global_load(
                result_binding,
                opcode,
                dest,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                out_type,
                in_types_str,
                token_type,
            )
        elif category == "buffer_load":
            body = migrate_check_buffer_load(
                result_binding,
                opcode,
                dest,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                out_type,
                in_types_str,
                token_type,
            )
        elif category == "s_load":
            body = migrate_check_s_load(
                result_binding,
                opcode,
                dest,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                out_type,
                in_types_str,
                token_type,
            )

        if body is not None:
            return f"{check_prefix}{text_prefix}{body}"
        return None

    # Try to match as a store instruction.
    store_re = re.compile(
        r"((?:" + SSA_PAT + r"\s*=\s*)?)"
        r"(?:amdgcn\.)?store\s+"
        r"([\w]+)\s+"
        r"data\s+(" + SSA_PAT + r")\s+"
        r"addr\s+(" + SSA_PAT + r")\s*"
        r"(.*)"
    )
    m = store_re.match(normalized)
    if m:
        result_binding = m.group(1).strip()
        if result_binding and not result_binding.endswith(" "):
            result_binding += " "
        opcode = m.group(2)
        data = m.group(3)
        addr = m.group(4)
        rest = m.group(5).strip()

        uniform, dynamic, constant, rest = parse_offset_clause(rest)
        attr_dict, rest = parse_attr_dict(rest)
        out_type, in_types_str, rest = parse_type_section(rest)
        token_match = re.search(r"(->\s*!amdgcn\.\w+_token<\w+>)", rest)
        token_type = token_match.group(1) if token_match else ""

        op_cat = get_opcode_category(opcode)
        if op_cat != category:
            return None

        body = None
        if category == "ds_write":
            body = migrate_check_ds_write(
                result_binding,
                opcode,
                data,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                in_types_str,
                token_type,
            )
        elif category == "global_store":
            body = migrate_check_global_store(
                result_binding,
                opcode,
                data,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                in_types_str,
                token_type,
            )
        elif category == "buffer_store":
            body = migrate_check_buffer_store(
                result_binding,
                opcode,
                data,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                in_types_str,
                token_type,
            )
        elif category == "s_store":
            body = migrate_check_s_store(
                result_binding,
                opcode,
                data,
                addr,
                uniform,
                dynamic,
                constant,
                attr_dict,
                in_types_str,
                token_type,
            )

        if body is not None:
            return f"{check_prefix}{text_prefix}{body}"
        return None

    # Try to match as a load_lds instruction.
    lds_re = re.compile(
        r"((?:" + SSA_PAT + r"\s*=\s*)?)"
        r"(?:amdgcn\.)?load_lds\s+"
        r"([\w]+)\s+"
        r"m0\s+(" + SSA_PAT + r")\s+"
        r"addr\s+(" + SSA_PAT + r")\s*"
        r"(.*)"
    )
    m = lds_re.match(normalized)
    if m and category == "buffer_lds":
        result_binding = m.group(1).strip()
        if result_binding and not result_binding.endswith(" "):
            result_binding += " "
        opcode = m.group(2)
        m0 = m.group(3)
        addr = m.group(4)
        rest = m.group(5).strip()

        uniform, dynamic, constant, rest = parse_offset_clause(rest)
        attr_dict, rest = parse_attr_dict(rest)
        out_type, in_types_str, rest = parse_type_section(rest)
        token_match = re.search(r"(->\s*!amdgcn\.\w+_token<\w+>)", rest)
        token_type = token_match.group(1) if token_match else ""

        body = migrate_check_buffer_lds(
            result_binding,
            opcode,
            m0,
            addr,
            uniform,
            dynamic,
            constant,
            attr_dict,
            in_types_str,
            token_type,
        )
        return f"{check_prefix}{text_prefix}{body}"

    return None


def apply_replacements(text, replacements):
    """Apply SSA value replacements to text.

    replacements maps old SSA names (e.g. '%result' or '%[[VAL_0:.*]]')
    to new names (e.g. '%d' or '%[[ARG0]]').

    For FileCheck patterns like '%[[VAL_0:.*]]', we also replace later
    uses '%[[VAL_0]]' (without the capture pattern).
    """
    for old_name, new_name in replacements.items():
        # Direct replacement (for normal SSA names and FileCheck definitions).
        old_escaped = re.escape(old_name)
        text = re.sub(old_escaped + r"(?=[\s,):;\]}]|$)", new_name, text)

        # For FileCheck captures: if old_name is '%[[FOO:.*]]',
        # also replace later uses '%[[FOO]]'.
        fc_match = re.match(r"^%\[\[([^:]+):\.\*\]\]$", old_name)
        if fc_match:
            base_name = fc_match.group(1)
            old_use = f"%[[{base_name}]]"
            old_use_escaped = re.escape(old_use)
            # Extract the base name from the new_name for FileCheck reference.
            new_fc_match = re.match(r"^%\[\[([^:\]]+)(?::\.\*)?\]\]$", new_name)
            if new_fc_match:
                new_use = f"%[[{new_fc_match.group(1)}]]"
            else:
                new_use = new_name
            text = re.sub(old_use_escaped + r"(?=[\s,):;\]}]|$)", new_use, text)

    return text


SKIP_CHECK_TRANSFORM_PASSES = set()

# Files where CHECK lines should NOT be transformed even if they contain
# old-format instructions, because the pass still generates old format.
SKIP_CHECK_FILES = set()


def should_transform_check_lines(filepath):
    """Determine if CHECK lines in this file should be transformed.

    Files that run passes which GENERATE old-format ops from higher-
    level dialects should not have their CHECK lines transformed, since
    the C++ pass still emits old format.
    """
    filename = Path(filepath).name
    if filename in SKIP_CHECK_FILES:
        return False
    with open(filepath, "r") as f:
        first_line = f.readline()
    for pass_name in SKIP_CHECK_TRANSFORM_PASSES:
        if pass_name in first_line:
            return False
    return True


def migrate_file(filepath, category, dry_run=False):
    """Migrate a single .mlir file for the given category."""
    with open(filepath, "r") as f:
        content = f.read()

    lines = content.split("\n")
    new_lines = []
    all_replacements = {}
    counter_holder = [0]
    check_counter_holder = [0]
    i = 0
    changed = False
    transform_checks = should_transform_check_lines(filepath)

    while i < len(lines):
        line = lines[i]

        # Apply accumulated SSA replacements to this line.
        if all_replacements:
            line = apply_replacements(line, all_replacements)

        # Check for CHECK lines with old-format instructions.
        if transform_checks and CHECK_PREFIX_RE.match(line):
            # First try full instruction migration (has dest/data/m0 keywords).
            has_full_old = (
                re.search(r"(?:amdgcn\.)?load\s+\w+\s+dest\s+", line)
                or re.search(r"(?:amdgcn\.)?store\s+\w+\s+data\s+", line)
                or re.search(r"(?:amdgcn\.)?load_lds\s+\w+\s+m0\s+", line)
            )
            if has_full_old:
                check_prefix, full_text, end_idx = extract_check_instruction_block(
                    lines, i
                )
                if all_replacements:
                    full_text = apply_replacements(full_text, all_replacements)
                result = migrate_check_instruction(
                    check_prefix, full_text, category, check_counter_holder
                )
                if result is not None:
                    new_lines.append(result)
                    changed = True
                    i = end_idx + 1
                    continue

            # Then try simple op name renaming for partial CHECK lines.
            cat_patterns = CATEGORY_PATTERNS.get(category, [])
            for old_pat in cat_patterns:
                if old_pat in line:
                    for old_name, new_name in SIMPLE_CHECK_RENAMES.items():
                        if old_name in line:
                            line = line.replace(old_name, new_name)
                            changed = True
                            break
                    break

        # Check for old-format instruction start in code lines.
        is_load = re.search(r"(?:amdgcn\.)?load\s+\w+\s+dest\s+", line)
        is_store = re.search(r"(?:amdgcn\.)?store\s+\w+\s+data\s+", line)
        is_load_lds = re.search(r"(?:amdgcn\.)?load_lds\s+\w+\s+m0\s+", line)

        if is_load or is_store or is_load_lds:
            full_text, end_idx = extract_instruction_block(lines, i)
            if all_replacements:
                full_text = apply_replacements(full_text, all_replacements)

            result = None
            if is_load:
                result = process_load_instruction(full_text, category, counter_holder)
            elif is_store:
                result = process_store_instruction(full_text, category, counter_holder)
            elif is_load_lds:
                result = process_load_lds_instruction(
                    full_text, category, counter_holder
                )

            if result is not None:
                new_text, replacements = result
                new_lines.append(new_text)
                all_replacements.update(replacements)
                changed = True
                i = end_idx + 1
                continue

        new_lines.append(line)
        i += 1

    if changed:
        new_content = "\n".join(new_lines)
        if dry_run:
            print(f"Would modify: {filepath}")
        else:
            with open(filepath, "w") as f:
                f.write(new_content)
            print(f"Modified: {filepath}")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate old memory instruction format"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "ds_read",
            "ds_write",
            "ds_perm",
            "global_load",
            "global_store",
            "buffer_load",
            "buffer_store",
            "buffer_lds",
            "s_load",
            "s_store",
            "all",
        ],
        help="Category of instructions to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific files to process (default: all .mlir files)",
    )
    args = parser.parse_args()

    categories = [args.category]
    if args.category == "all":
        categories = [
            "ds_read",
            "ds_write",
            "ds_perm",
            "global_load",
            "global_store",
            "buffer_load",
            "buffer_store",
            "buffer_lds",
            "s_load",
            "s_store",
        ]

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = find_mlir_files()

    for cat in categories:
        print(f"\n=== Migrating category: {cat} ===")
        modified_count = 0
        for filepath in files:
            if migrate_file(filepath, cat, dry_run=args.dry_run):
                modified_count += 1
        print(f"Modified {modified_count} files for category {cat}")


if __name__ == "__main__":
    main()
