#!/usr/bin/env python3
"""Migrate AMDGCN memory instructions from old outs/ins/args format to new
declarative assembly format.

Usage:
    python migrate_inst_format.py <file.mlir> [--dry-run]
    python migrate_inst_format.py --batch <file_list.txt>
"""

import argparse
import re
import sys


# ---------------------------------------------------------------------------
# Helpers to parse balanced parenthesized content
# ---------------------------------------------------------------------------


def find_balanced_paren(text, start):
    """Find the matching closing paren for an open paren at `start`.

    Returns the index of the closing paren, or -1.
    """
    assert text[start] == "(", f"Expected '(' at position {start}, got '{text[start]}'"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def extract_paren_content(text, start):
    """Extract content inside balanced parens starting at `start`.

    Returns (content, end_index) where end_index is after the closing
    paren.
    """
    end = find_balanced_paren(text, start)
    if end == -1:
        return None, start
    return text[start + 1 : end], end + 1


def split_top_level_commas(text):
    """Split text by commas that are not inside nested parens/angles."""
    parts = []
    depth_paren = 0
    depth_angle = 0
    current = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren -= 1
        elif ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle -= 1
        elif ch == "," and depth_paren == 0 and depth_angle == 0:
            parts.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    remainder = "".join(current).strip()
    if remainder:
        parts.append(remainder)
    return parts


# ---------------------------------------------------------------------------
# Instruction classification
# ---------------------------------------------------------------------------

DS_READ_OPS = [
    "ds_read_b32",
    "ds_read_b64",
    "ds_read_b96",
    "ds_read_b128",
]

DS_WRITE_OPS = [
    "ds_write_b32",
    "ds_write_b64",
    "ds_write_b96",
    "ds_write_b128",
]

S_LOAD_OPS = [
    "s_load_dword",
    "s_load_dwordx2",
    "s_load_dwordx4",
    "s_load_dwordx8",
    "s_load_dwordx16",
]

S_STORE_OPS = [
    "s_store_dword",
    "s_store_dwordx2",
    "s_store_dwordx4",
]

BUFFER_LOAD_OPS = [
    "buffer_load_dword",
    "buffer_load_dwordx2",
    "buffer_load_dwordx3",
    "buffer_load_dwordx4",
]

BUFFER_LOAD_LDS_OPS = [
    "buffer_load_lds_dword",
    "buffer_load_lds_dwordx2",
    "buffer_load_lds_dwordx3",
    "buffer_load_lds_dwordx4",
]

BUFFER_STORE_OPS = [
    "buffer_store_dword",
    "buffer_store_dwordx2",
    "buffer_store_dwordx3",
    "buffer_store_dwordx4",
]

GLOBAL_LOAD_OPS = [
    "global_load_dword",
    "global_load_dwordx2",
    "global_load_dwordx3",
    "global_load_dwordx4",
]

GLOBAL_STORE_OPS = [
    "global_store_dword",
    "global_store_dwordx2",
    "global_store_dwordx3",
    "global_store_dwordx4",
]

ALL_OPS = (
    DS_READ_OPS
    + DS_WRITE_OPS
    + S_LOAD_OPS
    + S_STORE_OPS
    + BUFFER_LOAD_OPS
    + BUFFER_LOAD_LDS_OPS
    + BUFFER_STORE_OPS
    + GLOBAL_LOAD_OPS
    + GLOBAL_STORE_OPS
)


def classify_op(mnemonic):
    """Return the instruction family for a given mnemonic."""
    if mnemonic in DS_READ_OPS:
        return "ds_read"
    if mnemonic in DS_WRITE_OPS:
        return "ds_write"
    if mnemonic in S_LOAD_OPS:
        return "s_load"
    if mnemonic in S_STORE_OPS:
        return "s_store"
    if mnemonic in BUFFER_LOAD_OPS:
        return "buffer_load"
    if mnemonic in BUFFER_LOAD_LDS_OPS:
        return "buffer_load_lds"
    if mnemonic in BUFFER_STORE_OPS:
        return "buffer_store"
    if mnemonic in GLOBAL_LOAD_OPS:
        return "global_load"
    if mnemonic in GLOBAL_STORE_OPS:
        return "global_store"
    return None


# ---------------------------------------------------------------------------
# Parsing the old format
# ---------------------------------------------------------------------------


def parse_segment(text, keyword):
    """Parse a segment like `outs(...)` or `ins(...)` or `args(...)` from text.

    Returns (items, named_items, remaining_text).
    items: list of positional operands/types
    named_items: dict of named operands like {name: value}
    """
    pattern = re.compile(r"\b" + re.escape(keyword) + r"\s*\(")
    m = pattern.search(text)
    if not m:
        return None, None, text
    paren_start = m.end() - 1
    content, after = extract_paren_content(text, paren_start)
    if content is None:
        return None, None, text

    items = []
    named_items = {}
    for part in split_top_level_commas(content):
        part = part.strip()
        if not part:
            continue
        named_match = re.match(r"(\w+)\s*=\s*(.*)", part)
        if named_match:
            named_items[named_match.group(1)] = named_match.group(2).strip()
        else:
            items.append(part)

    remaining = text[: m.start()].rstrip() + text[after:]
    return items, named_items, remaining


def parse_old_instruction(full_text):
    """Parse an old-format instruction from the combined text (operands + types).

    Returns a dict with all parsed components, or None if not parseable.
    """
    result = {}

    # Split at the first `:` that separates operand section from type section.
    # But we need to be careful: the colon is ` : ` or `\n ... :` pattern.
    # The type section starts with `outs(` or `ins(` or `args(` after the colon.
    colon_re = re.compile(r"\s*:\s*(?=(?:outs|ins|args)\()")
    m = colon_re.search(full_text)
    if not m:
        return None

    operand_part = full_text[: m.start()]
    type_part = full_text[m.end() :]

    # Parse operand-section segments.
    outs_ops, outs_named, operand_part = parse_segment(operand_part, "outs")
    ins_ops, ins_named, operand_part = parse_segment(operand_part, "ins")
    args_ops, args_named, operand_part = parse_segment(operand_part, "args")

    # Remaining operand_part has attr-dict (e.g., {offen, sched.stage = 1 : i32}).
    attrs = operand_part.strip()

    # Parse type-section segments.
    outs_types, outs_types_named, type_part = parse_segment(type_part, "outs")
    ins_types, ins_types_named, type_part = parse_segment(type_part, "ins")
    args_types, args_types_named, type_part = parse_segment(type_part, "args")

    # Remaining type_part should be ` -> <trailing_types>`.
    trailing = type_part.strip()
    if trailing.startswith("->"):
        trailing = trailing[2:].strip()

    result["outs_ops"] = outs_ops or []
    result["outs_named"] = outs_named or {}
    result["ins_ops"] = ins_ops or []
    result["ins_named"] = ins_named or {}
    result["args_ops"] = args_ops or []
    result["args_named"] = args_named or {}
    result["attrs"] = attrs
    result["outs_types"] = outs_types or []
    result["outs_types_named"] = outs_types_named or {}
    result["ins_types"] = ins_types or []
    result["ins_types_named"] = ins_types_named or {}
    result["args_types"] = args_types or []
    result["args_types_named"] = args_types_named or {}
    result["trailing_types"] = trailing

    return result


# ---------------------------------------------------------------------------
# Format converters per instruction family
# ---------------------------------------------------------------------------


def convert_ds_read(parsed):
    """DS read: dest %d addr %addr offset c(%off) [attrs] :
    outs(!T1) ins(!T2) mods(i32) -> !tok"""
    dest = parsed["outs_ops"][0]
    addr = parsed["ins_ops"][0]
    const_off = parsed["args_ops"][0]
    attrs = parsed["attrs"]

    dest_ty = parsed["outs_types"][0]
    addr_ty = parsed["ins_types"][0]
    off_ty = parsed["args_types"][0]
    trailing = parsed["trailing_types"]

    operands = f"dest {dest} addr {addr} offset c({const_off})"
    if attrs:
        operands += f" {attrs}"
    types = f"outs({dest_ty}) ins({addr_ty}) mods({off_ty}) -> {trailing}"
    return operands, types


def convert_ds_write(parsed):
    """DS write: data %data addr %addr offset c(%off) [attrs] :
       ins(!Tdata, !Taddr) mods(i32) -> !tok
    Note: old format is ins(%addr, %data), new is data %data addr %addr
    and types swap accordingly."""
    addr = parsed["ins_ops"][0]
    data = parsed["ins_ops"][1]
    const_off = parsed["args_ops"][0]
    attrs = parsed["attrs"]

    addr_ty = parsed["ins_types"][0]
    data_ty = parsed["ins_types"][1]
    off_ty = parsed["args_types"][0]
    trailing = parsed["trailing_types"]

    operands = f"data {data} addr {addr} offset c({const_off})"
    if attrs:
        operands += f" {attrs}"
    types = f"ins({data_ty}, {addr_ty}) mods({off_ty}) -> {trailing}"
    return operands, types


def convert_s_load(parsed):
    """S load: dest %dst addr %addr offset [u(%soff) | c(%off)] [attrs] :
    outs(!T1) ins(!T2[, !T3]) [mods(i32)] -> !tok"""
    dest = parsed["outs_ops"][0]
    addr = parsed["ins_ops"][0]
    attrs = parsed["attrs"]

    dest_ty = parsed["outs_types"][0]
    addr_ty = parsed["ins_types"][0]
    trailing = parsed["trailing_types"]

    has_soffset = "soffset" in parsed["ins_named"]
    has_const_offset = len(parsed["args_ops"]) > 0

    if has_const_offset and not has_soffset:
        const_off = parsed["args_ops"][0]
        off_ty = parsed["args_types"][0]
        operands = f"dest {dest} addr {addr} offset c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"outs({dest_ty}) ins({addr_ty}) mods({off_ty}) -> {trailing}"
    elif has_soffset and not has_const_offset:
        soff = parsed["ins_named"]["soffset"]
        soff_ty = parsed["ins_types_named"]["soffset"]
        operands = f"dest {dest} addr {addr} offset u({soff})"
        if attrs:
            operands += f" {attrs}"
        types = f"outs({dest_ty}) ins({addr_ty}, {soff_ty}) -> {trailing}"
    else:
        return None, None

    return operands, types


def convert_s_store(parsed):
    """S store: data %data addr %addr offset [u(%soff) | c(%off)] [attrs] :
    ins(!T1, !T2[, !T3]) [mods(i32)] -> !tok"""
    data = parsed["ins_ops"][0]
    addr = parsed["ins_ops"][1]
    attrs = parsed["attrs"]

    data_ty = parsed["ins_types"][0]
    addr_ty = parsed["ins_types"][1]
    trailing = parsed["trailing_types"]

    has_soffset = "soffset" in parsed["ins_named"]
    has_const_offset = len(parsed["args_ops"]) > 0

    if has_const_offset and not has_soffset:
        const_off = parsed["args_ops"][0]
        off_ty = parsed["args_types"][0]
        operands = f"data {data} addr {addr} offset c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({data_ty}, {addr_ty}) mods({off_ty}) -> {trailing}"
    elif has_soffset and not has_const_offset:
        soff = parsed["ins_named"]["soffset"]
        soff_ty = parsed["ins_types_named"]["soffset"]
        operands = f"data {data} addr {addr} offset u({soff})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({data_ty}, {addr_ty}, {soff_ty}) -> {trailing}"
    else:
        return None, None

    return operands, types


def convert_buffer_load(parsed):
    """Buffer load: dest %dest addr %rsrc offset u(%soff) [+ off_idx(%v) +] c(%off) [attrs] :
    outs(!T1) ins(!T2, !Tsoff[, !Toff_or_idx]) mods(i32) -> !tok"""
    dest = parsed["outs_ops"][0]
    attrs = parsed["attrs"]
    const_off = parsed["args_ops"][0]
    off_ty = parsed["args_types"][0]
    dest_ty = parsed["outs_types"][0]
    trailing = parsed["trailing_types"]

    has_off_or_idx = "off_or_idx" in parsed["ins_named"]

    # ins has: rsrc[, soffset] -- positional items after rsrc.
    rsrc = parsed["ins_ops"][0]
    rsrc_ty = parsed["ins_types"][0]

    if has_off_or_idx:
        off_or_idx_val = parsed["ins_named"]["off_or_idx"]
        off_or_idx_ty = parsed["ins_types_named"]["off_or_idx"]
        soff = parsed["ins_ops"][1]
        soff_ty = parsed["ins_types"][1]
        operands = f"dest {dest} addr {rsrc} offset u({soff}) + off_idx({off_or_idx_val}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"outs({dest_ty}) ins({rsrc_ty}, {soff_ty}, {off_or_idx_ty}) mods({off_ty}) -> {trailing}"
    else:
        soff = parsed["ins_ops"][1]
        soff_ty = parsed["ins_types"][1]
        operands = f"dest {dest} addr {rsrc} offset u({soff}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = (
            f"outs({dest_ty}) ins({rsrc_ty}, {soff_ty}) mods({off_ty}) -> {trailing}"
        )

    return operands, types


def convert_buffer_load_lds(parsed):
    """Buffer load LDS: addr %rsrc m0 %m0 offset u(%soff) [+ off_idx(%v) +] c(%off) [attrs] :
    ins(!Trsrc, !Tm0, !Tsoff[, !Toff_or_idx]) mods(i32) -> !tok"""
    attrs = parsed["attrs"]
    const_off = parsed["args_ops"][0]
    off_ty = parsed["args_types"][0]
    trailing = parsed["trailing_types"]

    has_off_or_idx = "off_or_idx" in parsed["ins_named"]

    rsrc = parsed["ins_ops"][0]
    rsrc_ty = parsed["ins_types"][0]

    if has_off_or_idx:
        off_or_idx_val = parsed["ins_named"]["off_or_idx"]
        off_or_idx_ty = parsed["ins_types_named"]["off_or_idx"]
        # After rsrc, remaining positional: soff, m0.
        soff = parsed["ins_ops"][1]
        soff_ty = parsed["ins_types"][1]
        m0 = parsed["ins_ops"][2]
        m0_ty = parsed["ins_types"][2]
        operands = f"addr {rsrc} m0 {m0} offset u({soff}) + off_idx({off_or_idx_val}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({rsrc_ty}, {m0_ty}, {soff_ty}, {off_or_idx_ty}) mods({off_ty}) -> {trailing}"
    else:
        # Positional: rsrc, soff, m0.
        soff = parsed["ins_ops"][1]
        soff_ty = parsed["ins_types"][1]
        m0 = parsed["ins_ops"][2]
        m0_ty = parsed["ins_types"][2]
        operands = f"addr {rsrc} m0 {m0} offset u({soff}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({rsrc_ty}, {m0_ty}, {soff_ty}) mods({off_ty}) -> {trailing}"

    return operands, types


def convert_buffer_store(parsed):
    """Buffer store: data %data addr %rsrc offset u(%soff) [+ off_idx(%v) +] c(%off) [attrs] :
    ins(!Tdata, !Trsrc, !Tsoff[, !Toff_or_idx]) mods(i32) -> !tok"""
    attrs = parsed["attrs"]
    const_off = parsed["args_ops"][0]
    off_ty = parsed["args_types"][0]
    trailing = parsed["trailing_types"]

    has_off_or_idx = "off_or_idx" in parsed["ins_named"]

    data = parsed["ins_ops"][0]
    data_ty = parsed["ins_types"][0]
    rsrc = parsed["ins_ops"][1]
    rsrc_ty = parsed["ins_types"][1]

    if has_off_or_idx:
        off_or_idx_val = parsed["ins_named"]["off_or_idx"]
        off_or_idx_ty = parsed["ins_types_named"]["off_or_idx"]
        soff = parsed["ins_ops"][2]
        soff_ty = parsed["ins_types"][2]
        operands = f"data {data} addr {rsrc} offset u({soff}) + off_idx({off_or_idx_val}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({data_ty}, {rsrc_ty}, {soff_ty}, {off_or_idx_ty}) mods({off_ty}) -> {trailing}"
    else:
        soff = parsed["ins_ops"][2]
        soff_ty = parsed["ins_types"][2]
        operands = f"data {data} addr {rsrc} offset u({soff}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({data_ty}, {rsrc_ty}, {soff_ty}) mods({off_ty}) -> {trailing}"

    return operands, types


def convert_global_load(parsed):
    """Global load: dest %dst addr %addr offset [d(%voff) +] c(%c0) [attrs] :
    outs(!T1) ins(!T2[, !T3]) mods(i32) -> !tok"""
    dest = parsed["outs_ops"][0]
    addr = parsed["ins_ops"][0]
    const_off = parsed["args_ops"][0]
    attrs = parsed["attrs"]

    dest_ty = parsed["outs_types"][0]
    addr_ty = parsed["ins_types"][0]
    off_ty = parsed["args_types"][0]
    trailing = parsed["trailing_types"]

    has_offset = "offset" in parsed["ins_named"]

    if has_offset:
        voff = parsed["ins_named"]["offset"]
        voff_ty = parsed["ins_types_named"]["offset"]
        operands = f"dest {dest} addr {addr} offset d({voff}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = (
            f"outs({dest_ty}) ins({addr_ty}, {voff_ty}) mods({off_ty}) -> {trailing}"
        )
    else:
        operands = f"dest {dest} addr {addr} offset c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"outs({dest_ty}) ins({addr_ty}) mods({off_ty}) -> {trailing}"

    return operands, types


def convert_global_store(parsed):
    """Global store: data %data addr %addr offset [d(%voff) +] c(%c0) [attrs] :
    ins(!T1, !T2[, !T3]) mods(i32) -> !tok"""
    data = parsed["ins_ops"][0]
    addr = parsed["ins_ops"][1]
    const_off = parsed["args_ops"][0]
    attrs = parsed["attrs"]

    data_ty = parsed["ins_types"][0]
    addr_ty = parsed["ins_types"][1]
    off_ty = parsed["args_types"][0]
    trailing = parsed["trailing_types"]

    has_offset = "offset" in parsed["ins_named"]

    if has_offset:
        voff = parsed["ins_named"]["offset"]
        voff_ty = parsed["ins_types_named"]["offset"]
        operands = f"data {data} addr {addr} offset d({voff}) + c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({data_ty}, {addr_ty}, {voff_ty}) mods({off_ty}) -> {trailing}"
    else:
        operands = f"data {data} addr {addr} offset c({const_off})"
        if attrs:
            operands += f" {attrs}"
        types = f"ins({data_ty}, {addr_ty}) mods({off_ty}) -> {trailing}"

    return operands, types


CONVERTERS = {
    "ds_read": convert_ds_read,
    "ds_write": convert_ds_write,
    "s_load": convert_s_load,
    "s_store": convert_s_store,
    "buffer_load": convert_buffer_load,
    "buffer_load_lds": convert_buffer_load_lds,
    "buffer_store": convert_buffer_store,
    "global_load": convert_global_load,
    "global_store": convert_global_store,
}


# ---------------------------------------------------------------------------
# Main file-level migration
# ---------------------------------------------------------------------------


def build_op_pattern():
    """Build a regex that matches `amdgcn.<op> ` for all affected ops."""
    escaped = [re.escape(op) for op in ALL_OPS]
    return re.compile(r"amdgcn\.(" + "|".join(escaped) + r")\b")


OP_PATTERN = build_op_pattern()


def join_continuation_lines(lines):
    """Join lines where an instruction spans multiple lines.

    A continuation is detected when a line ends without `->` and the
    next line (stripped) starts with `:` followed by `outs(` or `ins(`
    or `args(`.
    """
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if this line contains an affected op and the next line is a
        # type continuation.
        if i + 1 < len(lines) and OP_PATTERN.search(line):
            next_line = lines[i + 1]
            next_stripped = next_line.lstrip()
            if next_stripped.startswith(": ") and re.match(
                r":\s*(?:outs|ins|args)\(", next_stripped
            ):
                combined = line.rstrip() + "\n" + lines[i + 1]
                result.append(combined)
                i += 2
                continue
        result.append(line)
        i += 1
    return result


def migrate_instruction(match_text, mnemonic):
    """Given the text from after `amdgcn.<mnemonic> ` to end-of-instruction,
    convert it to the new format.

    Returns (new_text, success).
    """
    family = classify_op(mnemonic)
    if family is None:
        return match_text, False

    parsed = parse_old_instruction(match_text)
    if parsed is None:
        return match_text, False

    converter = CONVERTERS.get(family)
    if converter is None:
        return match_text, False

    try:
        operands, types = converter(parsed)
    except (IndexError, KeyError, TypeError):
        return match_text, False

    if operands is None:
        return match_text, False

    new_text = f"{operands} : {types}"
    return new_text, True


def process_line(line):
    """Process a single (possibly joined) line, migrating any affected instruction."""
    m = OP_PATTERN.search(line)
    if not m:
        return line

    mnemonic = m.group(1)

    prefix = line[: m.end()]
    rest = line[m.end() :]

    # The rest starts with a space then the operands.
    if not rest or rest[0] != " ":
        return line

    rest = rest[1:]

    new_rest, success = migrate_instruction(rest, mnemonic)
    if not success:
        return line

    return prefix + " " + new_rest


def split_back_to_lines(joined_lines):
    """Split joined lines back, preserving the original line structure."""
    result = []
    for line in joined_lines:
        if "\n" in line:
            parts = line.split("\n")
            result.extend(parts)
        else:
            result.append(line)
    return result


def migrate_file(filepath, dry_run=False):
    """Migrate a single MLIR file.

    Returns True if changes were made.
    """
    with open(filepath, "r") as f:
        original = f.read()

    lines = original.split("\n")

    # Join continuation lines.
    joined = join_continuation_lines(lines)

    # Process each (possibly joined) line.
    processed = []
    changed = False
    for line in joined:
        new_line = process_line(line)
        if new_line != line:
            changed = True
        processed.append(new_line)

    if not changed:
        return False

    # Split joined lines back to their original multi-line structure.
    # For converted instructions, we need to re-split at the ` : ` boundary
    # to maintain the 2-line format.
    final_lines = []
    for line in processed:
        if "\n" in line:
            # This was a joined multi-line instruction that was converted.
            # Find the original indentation of the second line.
            parts = line.split("\n")
            if len(parts) == 2:
                # Get indentation from original second line.
                second_line_indent = len(parts[1]) - len(parts[1].lstrip())
                indent = " " * second_line_indent

                # Find the ` : ` separator in the converted instruction.
                m = OP_PATTERN.search(line)
                if m:
                    after_op = line[m.end() :]
                    colon_re = re.compile(r"\s*:\s*(?=(?:outs|ins|mods)\()")
                    cm = colon_re.search(after_op)
                    if cm:
                        split_point = m.end() + cm.start()
                        first_part = line[:split_point].rstrip()
                        second_part = line[m.end() + cm.end() :].strip()
                        # Remove any embedded newlines from first_part.
                        first_part = first_part.replace("\n", "").rstrip()
                        # Re-strip leading whitespace artifacts.
                        first_part = re.sub(r"\s+", " ", first_part)
                        # Reconstruct with proper indentation, preserving the
                        # leading whitespace of the first line.
                        leading_ws = len(parts[0]) - len(parts[0].lstrip())
                        first_part_stripped = first_part.lstrip()
                        final_lines.append(" " * leading_ws + first_part_stripped)
                        final_lines.append(indent + ": " + second_part)
                        continue

            # Fallback: just split normally.
            for p in parts:
                final_lines.append(p)
        else:
            final_lines.append(line)

    result = "\n".join(final_lines)

    if dry_run:
        if result != original:
            print(f"Would modify: {filepath}")
            return True
        return False

    if result != original:
        with open(filepath, "w") as f:
            f.write(result)
        print(f"Modified: {filepath}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Migrate AMDGCN memory instruction format in MLIR files."
    )
    parser.add_argument("files", nargs="*", help="MLIR files to migrate")
    parser.add_argument("--batch", help="File containing list of paths to migrate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report which files would be changed",
    )
    args = parser.parse_args()

    files = list(args.files)
    if args.batch:
        with open(args.batch) as f:
            for line in f:
                line = line.strip()
                if line:
                    files.append(line)

    if not files:
        print("No files specified.", file=sys.stderr)
        sys.exit(1)

    total_changed = 0
    for filepath in files:
        try:
            if migrate_file(filepath, dry_run=args.dry_run):
                total_changed += 1
        except Exception as e:
            print(f"Error processing {filepath}: {e}", file=sys.stderr)

    print(
        f"\n{'Would modify' if args.dry_run else 'Modified'}: "
        f"{total_changed}/{len(files)} files"
    )


if __name__ == "__main__":
    main()
