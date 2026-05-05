#!/usr/bin/env python3
"""Migrate VOP3P instructions to the new format.

Transforms old-format AMDGCN VOP3P instruction syntax in .mlir and .py files
to the new instruction-per-op format.

Patterns handled:
  - amdgcn.vop3p.vop3p_mai -> amdgcn.<mnemonic>
  - amdgcn.vop3p.vop3p_scaled_mai -> amdgcn.<mnemonic>
  - amdgcn.vop3p v_accvgpr_{read,write}_b32 -> amdgcn.v_accvgpr_{read,write}
  - "amdgcn.vop3p.vop3p_mai" (generic form)
  - Modifier syntax: attr = N -> attr(N : type)
  - Python API renames
"""

import argparse
import re
import sys
from pathlib import Path


def migrate_mlir_content(content: str) -> str:
    """Migrate all old-format patterns in MLIR content."""
    content = migrate_mfma_generic_form(content)
    content = migrate_mfma_generic_form_no_dot(content)
    content = migrate_mfma_pretty(content)
    content = migrate_mfma_pretty_no_dot(content)
    content = migrate_scaled_mfma_pretty(content)
    content = migrate_scaled_mfma_pretty_no_dot(content)
    content = migrate_accvgpr(content)
    content = migrate_check_lines(content)
    return content


def migrate_mfma_generic_form(content: str) -> str:
    """Migrate generic form: "amdgcn.vop3p.vop3p_mai"(operands) {attrs} : (types) -> type"""
    pattern = re.compile(
        r'"amdgcn\.vop3p\.vop3p_mai"\('
        r"([^)]+)"  # operands: %dst, %a, %b, %c
        r"\)\s*\{([^}]*)\}\s*:\s*"
        r"\(([^)]+)\)\s*->\s*"
        r"((?:!amdgcn\.\w+(?:<[^>]*>)?|<[^>]+>))",  # result type
        re.DOTALL,
    )

    def repl(m):
        operands_str = m.group(1).strip()
        attrs_str = m.group(2).strip()
        in_types_str = m.group(3).strip()
        out_type = m.group(4).strip()

        # Extract mnemonic from opcode attribute.
        opcode_match = re.search(r"opcode\s*=\s*#amdgcn\.inst<([^>]+)>", attrs_str)
        if not opcode_match:
            return m.group(0)
        mnemonic = opcode_match.group(1)

        operands = split_operands(operands_str)
        if len(operands) < 4:
            return m.group(0)
        dst, a, b, c = operands[0], operands[1], operands[2], operands[3]

        in_types = split_operands(in_types_str)
        if len(in_types) < 4:
            return m.group(0)
        _dst_type, a_type, b_type, c_type = (
            in_types[0],
            in_types[1],
            in_types[2],
            in_types[3],
        )

        # Extract modifiers from remaining attrs (skip opcode and acc_cd).
        mods = extract_modifiers_from_generic_attrs(attrs_str)
        mod_str = format_modifiers(mods)

        return (
            f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c})"
            f"{mod_str}\n"
            f"    : outs({out_type})\n"
            f"      ins({a_type}, {b_type}, {c_type})"
        )

    return pattern.sub(repl, content)


def migrate_mfma_generic_form_no_dot(content: str) -> str:
    """Migrate generic form without dot: "amdgcn.vop3p_mai"(operands) {attrs} : (types) -> type"""
    pattern = re.compile(
        r'"amdgcn\.vop3p_mai"\('
        r"([^)]+)"
        r"\)\s*\{([^}]*)\}\s*:\s*"
        r"\(([^)]+)\)\s*->\s*"
        r"((?:!amdgcn\.\w+(?:<[^>]*>)?|<[^>]+>|\(\)))",
        re.DOTALL,
    )

    def repl(m):
        operands_str = m.group(1).strip()
        attrs_str = m.group(2).strip()
        in_types_str = m.group(3).strip()
        out_type = m.group(4).strip()

        opcode_match = re.search(r"opcode\s*=\s*#amdgcn\.inst<([^>]+)>", attrs_str)
        if not opcode_match:
            return m.group(0)
        mnemonic = opcode_match.group(1)

        operands = split_operands(operands_str)
        if len(operands) < 4:
            return m.group(0)
        dst, a, b, c = operands[0], operands[1], operands[2], operands[3]

        in_types = split_operands(in_types_str)
        if len(in_types) < 4:
            return m.group(0)
        _dst_type, a_type, b_type, c_type = (
            in_types[0],
            in_types[1],
            in_types[2],
            in_types[3],
        )

        mods = extract_modifiers_from_generic_attrs(attrs_str)
        mod_str = format_modifiers(mods)

        # Handle void result type (used in invalid tests).
        if out_type == "()":
            return (
                f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c})"
                f"{mod_str}\n"
                f"    : outs()\n"
                f"      ins({a_type}, {b_type}, {c_type})"
            )

        return (
            f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c})"
            f"{mod_str}\n"
            f"    : outs({out_type})\n"
            f"      ins({a_type}, {b_type}, {c_type})"
        )

    return pattern.sub(repl, content)


def migrate_mfma_pretty(content: str) -> str:
    """Migrate pretty-printed MFMA: amdgcn.vop3p.vop3p_mai #amdgcn.inst<M> or <M>"""
    # This regex matches the multi-line MFMA pattern.
    # The instruction can span multiple lines: operands, modifiers, types.
    pattern = re.compile(
        r"(amdgcn\.vop3p\.vop3p_mai)\s+"
        r"(?:#amdgcn\.inst)?<([^>]+)>\s+"  # mnemonic
        r"(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)"  # %dst, %a, %b, %c
        r"((?:\s+(?:cbsz|abid|blgp|acc_cd)\s*(?:=\s*\d+)?)*)"  # modifiers
        r"\s*:\s*"
        r"([\s\S]*?)"  # input types (a, b, c types)
        r"\s*->\s*"
        r"((?:!amdgcn\.\w+(?:<[^>]*>)?|<[^>]+>))",  # result type
        re.MULTILINE,
    )

    def repl(m):
        mnemonic = m.group(2)
        dst, a, b, c = m.group(3), m.group(4), m.group(5), m.group(6)
        mods_str = m.group(7).strip()
        types_str = m.group(8).strip()
        out_type = m.group(9).strip()

        # Resolve shorthand result type like <[? + 4]> to full form.
        if out_type.startswith("<") and not out_type.startswith("!"):
            out_type = resolve_shorthand_type(out_type, dst)

        in_types = split_type_list(types_str)
        if len(in_types) < 3:
            return m.group(0)

        mods = parse_old_modifiers(mods_str)
        mod_str = format_modifiers(mods)

        a_type = (
            resolve_shorthand_type(in_types[0].strip(), a)
            if in_types[0].strip().startswith("<")
            else in_types[0].strip()
        )
        b_type = (
            resolve_shorthand_type(in_types[1].strip(), b)
            if in_types[1].strip().startswith("<")
            else in_types[1].strip()
        )
        c_type = in_types[2].strip()

        return (
            f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c})"
            f"{mod_str}\n"
            f"    : outs({out_type})\n"
            f"      ins({a_type}, {b_type}, {c_type})"
        )

    return pattern.sub(repl, content)


def migrate_mfma_pretty_no_dot(content: str) -> str:
    """Migrate pretty-printed MFMA without dot prefix: amdgcn.vop3p_mai <M>"""
    pattern = re.compile(
        r"(amdgcn\.vop3p_mai)\s+"
        r"(?:#amdgcn\.inst)?<([^>]+)>\s+"  # mnemonic
        r"(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)"
        r"((?:\s+(?:cbsz|abid|blgp|acc_cd)\s*(?:=\s*\d+)?)*)"
        r"\s*:\s*"
        r"([\s\S]*?)"
        r"\s*->\s*"
        r"((?:!amdgcn\.\w+(?:<[^>]*>)?|<[^>]+>))",
        re.MULTILINE,
    )

    def repl(m):
        mnemonic = m.group(2)
        dst, a, b, c = m.group(3), m.group(4), m.group(5), m.group(6)
        mods_str = m.group(7).strip()
        types_str = m.group(8).strip()
        out_type = m.group(9).strip()

        if out_type.startswith("<") and not out_type.startswith("!"):
            out_type = resolve_shorthand_type(out_type, dst)

        in_types = split_type_list(types_str)
        if len(in_types) < 3:
            return m.group(0)

        mods = parse_old_modifiers(mods_str)
        mod_str = format_modifiers(mods)

        a_type = (
            resolve_shorthand_type(in_types[0].strip(), a)
            if in_types[0].strip().startswith("<")
            else in_types[0].strip()
        )
        b_type = (
            resolve_shorthand_type(in_types[1].strip(), b)
            if in_types[1].strip().startswith("<")
            else in_types[1].strip()
        )
        c_type = in_types[2].strip()

        return (
            f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c})"
            f"{mod_str}\n"
            f"    : outs({out_type})\n"
            f"      ins({a_type}, {b_type}, {c_type})"
        )

    return pattern.sub(repl, content)


def migrate_scaled_mfma_pretty(content: str) -> str:
    """Migrate pretty-printed scaled MFMA."""
    pattern = re.compile(
        r"(amdgcn\.vop3p\.vop3p_scaled_mai)\s+"
        r"(?:#amdgcn\.inst)?<([^>]+)>\s*\n?\s*"  # mnemonic
        r"(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)"  # %dst, %a, %b, %c, %s0, %s1
        r"((?:\s+(?:cbsz|abid|blgp|op_sel_0|op_sel_1|acc_cd)\s*(?:=\s*\d+)?)*)"  # modifiers
        r"\s*:\s*"
        r"([\s\S]*?)"  # input types
        r"\s*->\s*"
        r"((?:!amdgcn\.\w+(?:<[^>]*>)?|<[^>]+>))",  # result type
        re.MULTILINE,
    )

    def repl(m):
        mnemonic = m.group(2)
        dst = m.group(3)
        a, b, c, s0, s1 = m.group(4), m.group(5), m.group(6), m.group(7), m.group(8)
        mods_str = m.group(9).strip()
        types_str = m.group(10).strip()
        out_type = m.group(11).strip()

        if out_type.startswith("<") and not out_type.startswith("!"):
            out_type = resolve_shorthand_type(out_type, dst)

        in_types = split_type_list(types_str)
        if len(in_types) < 5:
            return m.group(0)

        mods = parse_old_modifiers(mods_str)
        mod_str = format_modifiers(mods)

        a_type = (
            resolve_shorthand_type(in_types[0].strip(), a)
            if in_types[0].strip().startswith("<")
            else in_types[0].strip()
        )
        b_type = (
            resolve_shorthand_type(in_types[1].strip(), b)
            if in_types[1].strip().startswith("<")
            else in_types[1].strip()
        )
        c_type = in_types[2].strip()
        s0_type = in_types[3].strip()
        s1_type = in_types[4].strip()

        return (
            f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c}, {s0}, {s1})"
            f"{mod_str}\n"
            f"    : outs({out_type})\n"
            f"      ins({a_type}, {b_type}, {c_type}, {s0_type}, {s1_type})"
        )

    return pattern.sub(repl, content)


def migrate_scaled_mfma_pretty_no_dot(content: str) -> str:
    """Migrate pretty-printed scaled MFMA without dot: amdgcn.vop3p_scaled_mai <M>"""
    pattern = re.compile(
        r"(amdgcn\.vop3p_scaled_mai)\s+"
        r"(?:#amdgcn\.inst)?<([^>]+)>\s*\n?\s*"
        r"(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)\s*,\s*(%\w+)"
        r"((?:\s+(?:cbsz|abid|blgp|op_sel_0|op_sel_1|acc_cd)\s*(?:=\s*\d+)?)*)"
        r"\s*:\s*"
        r"([\s\S]*?)"
        r"\s*->\s*"
        r"((?:!amdgcn\.\w+(?:<[^>]*>)?|<[^>]+>))",
        re.MULTILINE,
    )

    def repl(m):
        mnemonic = m.group(2)
        dst = m.group(3)
        a, b, c, s0, s1 = m.group(4), m.group(5), m.group(6), m.group(7), m.group(8)
        mods_str = m.group(9).strip()
        types_str = m.group(10).strip()
        out_type = m.group(11).strip()

        if out_type.startswith("<") and not out_type.startswith("!"):
            out_type = resolve_shorthand_type(out_type, dst)

        in_types = split_type_list(types_str)
        if len(in_types) < 5:
            return m.group(0)

        mods = parse_old_modifiers(mods_str)
        mod_str = format_modifiers(mods)

        a_type = (
            resolve_shorthand_type(in_types[0].strip(), a)
            if in_types[0].strip().startswith("<")
            else in_types[0].strip()
        )
        b_type = (
            resolve_shorthand_type(in_types[1].strip(), b)
            if in_types[1].strip().startswith("<")
            else in_types[1].strip()
        )
        c_type = in_types[2].strip()
        s0_type = in_types[3].strip()
        s1_type = in_types[4].strip()

        return (
            f"amdgcn.{mnemonic} outs({dst}) ins({a}, {b}, {c}, {s0}, {s1})"
            f"{mod_str}\n"
            f"    : outs({out_type})\n"
            f"      ins({a_type}, {b_type}, {c_type}, {s0_type}, {s1_type})"
        )

    return pattern.sub(repl, content)


def migrate_accvgpr(content: str) -> str:
    """Migrate ACCVGPR read/write instructions."""
    # Pattern: amdgcn.vop3p v_accvgpr_{read,write}_b32 outs %dst ins %src : DST_TYPE, SRC_TYPE
    # Optional result: %r = ... or just the instruction without result.
    pattern = re.compile(
        r"amdgcn\.vop3p\s+v_accvgpr_(read|write)_b32\s+"
        r"outs\s+(%\w+)\s+"
        r"ins\s+(%\w+)\s*:\s*"
        r"([^,\n]+),\s*"  # dst type
        r"([^\n]+)",  # src type
    )

    def repl(m):
        rw = m.group(1)
        dst, src = m.group(2), m.group(3)
        dst_type = m.group(4).strip()
        src_type = m.group(5).strip()
        return (
            f"amdgcn.v_accvgpr_{rw} outs({dst}) ins({src})\n"
            f"    : outs({dst_type}) ins({src_type})"
        )

    return pattern.sub(repl, content)


def migrate_check_lines(content: str) -> str:
    """Migrate CHECK-* lines and other references to old instruction names."""
    # Replace forms with mnemonic: amdgcn.vop3p[.]vop3p_{scaled_,}mai [#amdgcn.inst]<M>
    content = re.sub(
        r"amdgcn\.(?:vop3p\.)?vop3p_scaled_mai\s+(?:#amdgcn\.inst)?<([^>]+)>",
        r"amdgcn.\1",
        content,
    )
    content = re.sub(
        r"amdgcn\.(?:vop3p\.)?vop3p_mai\s+(?:#amdgcn\.inst)?<([^>]+)>",
        r"amdgcn.\1",
        content,
    )

    # Bare references without mnemonic (e.g. in CHECK-NOT or comments).
    content = re.sub(
        r"amdgcn\.(?:vop3p\.)?vop3p_scaled_mai\b",
        "amdgcn.v_mfma_scale",
        content,
    )
    content = re.sub(
        r"amdgcn\.(?:vop3p\.)?vop3p_mai\b",
        "amdgcn.v_mfma",
        content,
    )

    # Python assert string.
    content = content.replace('"v_mfma_f32_16x16x16_f16"', '"v_mfma_f32_16x16x16_f16"')

    return content


def migrate_python_content(content: str) -> str:
    """Migrate Python file content."""
    content = content.replace("v_accvgpr_write", "v_accvgpr_write")
    content = content.replace("v_accvgpr_read", "v_accvgpr_read")
    content = content.replace('"v_mfma_f32_16x16x16_f16"', '"v_mfma_f32_16x16x16_f16"')
    return content


# --- Helper functions ---


def split_operands(s: str) -> list[str]:
    """Split comma-separated operands, respecting nested angle brackets."""
    result = []
    depth = 0
    current = []
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current).strip())
    return result


def split_type_list(s: str) -> list[str]:
    """Split a comma-separated list of types, handling angle brackets and nested parens."""
    result = []
    depth = 0
    current = []
    for ch in s:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        t = "".join(current).strip()
        if t:
            result.append(t)
    return result


MODIFIER_TYPES = {
    "cbsz": "i8",
    "abid": "i16",
    "blgp": "i8",
    "op_sel_0": "i8",
    "op_sel_1": "i8",
}


def parse_old_modifiers(mods_str: str) -> dict[str, int]:
    """Parse old-format modifiers like 'cbsz = 2 blgp = 4 acc_cd'."""
    mods = {}
    if not mods_str:
        return mods
    for m in re.finditer(r"(cbsz|abid|blgp|op_sel_0|op_sel_1)\s*=\s*(\d+)", mods_str):
        mods[m.group(1)] = int(m.group(2))
    return mods


def extract_modifiers_from_generic_attrs(attrs_str: str) -> dict[str, int]:
    """Extract modifiers from generic form attributes string."""
    mods = {}
    for m in re.finditer(r"(cbsz|abid|blgp|op_sel_0|op_sel_1)\s*=\s*(\d+)", attrs_str):
        mods[m.group(1)] = int(m.group(2))
    return mods


def format_modifiers(mods: dict[str, int]) -> str:
    """Format modifiers in the new format: cbsz(N : i8) abid(N : i16) blgp(N : i8)."""
    if not mods:
        return ""
    parts = []
    for key in ["cbsz", "abid", "blgp", "op_sel_0", "op_sel_1"]:
        if key in mods:
            ty = MODIFIER_TYPES[key]
            parts.append(f"{key}({mods[key]} : {ty})")
    if not parts:
        return ""
    return " " + " ".join(parts)


def resolve_shorthand_type(shorthand: str, operand_name: str) -> str:
    """Resolve shorthand type like <[? + 2]> using context.

    In some test files, the input types are written as shorthand like
    <[? + 2]> without the full !amdgcn.vgpr prefix. We cannot resolve
    these without more context, so we just prepend !amdgcn.vgpr as a
    best guess.
    """
    if shorthand.startswith("!"):
        return shorthand
    return f"!amdgcn.vgpr{shorthand}"


def process_file(filepath: Path, dry_run: bool = False) -> bool:
    """Process a single file.

    Returns True if changes were made.
    """
    content = filepath.read_text()
    if filepath.suffix == ".mlir":
        new_content = migrate_mlir_content(content)
    elif filepath.suffix == ".py":
        new_content = migrate_python_content(content)
    else:
        return False

    if new_content == content:
        return False

    if dry_run:
        print(f"Would modify: {filepath}")
    else:
        filepath.write_text(new_content)
        print(f"Modified: {filepath}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Migrate VOP3P instructions to new format"
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to search for .mlir and .py files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"Error: {args.root} is not a directory", file=sys.stderr)
        sys.exit(1)

    modified = 0
    for ext in ("*.mlir", "*.py"):
        for filepath in sorted(args.root.rglob(ext)):
            # Skip build directories and worktrees.
            rel = filepath.relative_to(args.root)
            parts = rel.parts
            if any(p in ("build", "sandbox", ".git", "__pycache__") for p in parts):
                continue
            if process_file(filepath, args.dry_run):
                modified += 1

    print(f"\n{'Would modify' if args.dry_run else 'Modified'} {modified} file(s)")


if __name__ == "__main__":
    main()
