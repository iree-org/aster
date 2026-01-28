#!/usr/bin/env python3
"""Script to remove unused type aliases and function declarations from .mlir files."""

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def extract_type_aliases(content: str) -> Dict[str, Tuple[int, str]]:
    """Extract type alias declarations: !name = ..."""
    aliases = {}
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        # Match type alias: !name = ...
        match = re.match(r"^(\s*)!(\w+)\s*=\s*(.+)$", line)
        if match:
            indent, name, definition = match.groups()
            aliases[name] = (i, line)
    return aliases


def extract_function_declarations(content: str) -> Dict[str, Tuple[int, str]]:
    """Extract function declarations: func.func private @name(...) without body"""
    funcs = {}
    lines = content.split("\n")
    for i, line in enumerate(lines, 1):
        # Match function declaration (not definition): func.func private @name(...) without {
        # Declarations are single-line and don't have { on the same line
        stripped = line.strip()
        if "{" in stripped:
            # This is a function definition, skip it
            continue

        # Match function declaration pattern
        match = re.match(r"func\.func\s+private\s+@(\w+)\s*\([^)]*\)", stripped)
        if match:
            name = match.group(1)
            # Check if next non-empty line has { (would indicate it's a multi-line definition)
            is_definition = False
            for j in range(i, min(i + 2, len(lines))):
                next_line = lines[j].strip()
                if next_line and not next_line.startswith("//"):
                    if "{" in next_line:
                        is_definition = True
                    break

            if not is_definition:
                funcs[name] = (i, line)
    return funcs


def find_usages(content: str, name: str, exclude_lines: Set[int] = None) -> Set[int]:
    """Find all usages of a name in the content."""
    if exclude_lines is None:
        exclude_lines = set()
    usages = set()
    lines = content.split("\n")

    # Pattern to match usage of the name (as type or function call)
    # Type usage: !name
    # Function call: @name or func.call @name
    patterns = [
        rf"!{name}\b",  # Type usage
        rf"@{name}\b",  # Function reference
    ]

    for i, line in enumerate(lines, 1):
        if i in exclude_lines:
            continue
        for pattern in patterns:
            if re.search(pattern, line):
                usages.add(i)

    return usages


def is_declaration_line(
    line_num: int, declarations: Dict[str, Tuple[int, str]]
) -> bool:
    """Check if a line number is a declaration line."""
    for decl_line, _ in declarations.values():
        if decl_line == line_num:
            return True
    return False


def remove_unused_declarations(file_path: Path) -> bool:
    """Remove unused type aliases and function declarations from a file."""
    content = file_path.read_text()
    original_content = content
    lines = content.split("\n")

    # Extract declarations
    type_aliases = extract_type_aliases(content)
    func_declarations = extract_function_declarations(content)

    # First pass: find unused function declarations
    unused_funcs = []
    for name, (line_num, line) in func_declarations.items():
        usages = find_usages(content, name, exclude_lines={line_num})
        if not usages:
            unused_funcs.append((line_num, name, line))

    unused_func_lines = {line_num for line_num, _, _ in unused_funcs}

    # Second pass: find unused type aliases
    # A type is unused if it's not used anywhere, OR only used in unused function declarations
    unused_types = []
    for name, (line_num, line) in type_aliases.items():
        usages = find_usages(content, name, exclude_lines={line_num})

        if not usages:
            # Not used anywhere
            unused_types.append((line_num, name, line))
        else:
            # Check if all usages are only in unused function declaration lines
            all_in_unused_funcs = True
            for usage_line in usages:
                # Check if this line is an unused function declaration
                if usage_line not in unused_func_lines:
                    all_in_unused_funcs = False
                    break
            if all_in_unused_funcs:
                unused_types.append((line_num, name, line))

    if not unused_types and not unused_funcs:
        return False

    # Remove unused declarations
    lines_to_remove = set()

    for line_num, name, line in unused_types:
        lines_to_remove.add(line_num - 1)  # Convert to 0-based index

    for line_num, name, line in unused_funcs:
        lines_to_remove.add(line_num - 1)  # Convert to 0-based index

    new_lines = []
    for i, line in enumerate(lines):
        if i not in lines_to_remove:
            new_lines.append(line)

    new_content = "\n".join(new_lines)

    # Write back if changed
    if new_content != original_content:
        file_path.write_text(new_content)
        if unused_types:
            print(
                f"{file_path}: Removed {len(unused_types)} unused type alias(es): {[n for _, n, _ in unused_types]}"
            )
        if unused_funcs:
            print(
                f"{file_path}: Removed {len(unused_funcs)} unused function declaration(s): {[n for _, n, _ in unused_funcs]}"
            )
        return True

    return False


def is_library_file(file_path: Path) -> bool:
    """Check if a file is a library file that should not be cleaned."""
    # Check if it's in library/common directory
    if "library/common" in str(file_path):
        return True

    # Check if it contains amdgcn.library declaration
    try:
        content = file_path.read_text()
        if "amdgcn.library" in content:
            return True
    except:
        pass

    return False


def main():
    # Get the mlir_kernels directory (parent of utils directory where this script lives)
    script_dir = Path(__file__).parent
    mlir_kernels_dir = script_dir.parent

    if not mlir_kernels_dir.exists():
        print(f"Error: {mlir_kernels_dir} does not exist")
        sys.exit(1)

    mlir_files = list(mlir_kernels_dir.rglob("*.mlir"))
    print(f"Found {len(mlir_files)} .mlir files")

    modified_count = 0
    skipped_count = 0
    for mlir_file in sorted(mlir_files):
        if is_library_file(mlir_file):
            skipped_count += 1
            continue
        if remove_unused_declarations(mlir_file):
            modified_count += 1

    print(f"\nSkipped {skipped_count} library file(s)")
    print(f"Modified {modified_count} file(s)")


if __name__ == "__main__":
    main()
