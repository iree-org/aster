# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for layout-driven MLIR code generation.

from __future__ import annotations

from aster.layout import Layout, product
from aster.testing import compile_mlir_module_to_asm

# ---------------------------------------------------------------------------
# Shared helpers (imported by integration/test_layout_copy_e2e.py)
# ---------------------------------------------------------------------------


def build_copy_kernel(
    name: str,
    thread_layout: Layout,
    target: str = "gfx942",
    isa: str = "cdna3",
):
    """Build a flat-global dwordx4 copy kernel using KernelBuilder + layout."""
    from aster.dialects.kernel_builder import KernelBuilder
    from aster.dialects.amdgcn import AccessKind

    b = KernelBuilder(f"{name}_module", name, target=target, isa=isa)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()

    tid = b.thread_id("x")
    byte_off = b.linearize_layout(tid, thread_layout)

    src_addr = b.global_addr(src_ptr, byte_off)
    data = b.global_load_dwordx4(src_addr)
    b.wait_vmcnt(0)

    dst_addr = b.global_addr(dst_ptr, byte_off)
    b.global_store_dwordx4(data, dst_addr)
    b.wait_vmcnt(0)

    return b.build()


def build_and_compile_copy_kernel(
    name: str,
    thread_layout: Layout,
    print_ir_after_all: bool = False,
) -> str:
    """Build a copy kernel and compile to assembly. Returns asm string.

    Must be called inside an active ``with ir.Context():`` block.
    """
    module = build_copy_kernel(name, thread_layout)
    return compile_mlir_module_to_asm(module, print_ir_after_all=print_ir_after_all)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_build_copy_kernel_ir():
    """Verify the generated IR has the expected structure."""
    from aster import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_copy_kernel("copy_test", Layout(sizes=(4, 16), strides=(16, 64)))
        text = str(module)
        assert "amdgcn.module" in text
        assert "global_load_dwordx4" in text
        assert "global_store_dwordx4" in text
        assert "ptr.ptr_add" in text
        assert "layout.linearize" in text


def test_build_copy_kernel_asm():
    """Compile all the way to assembly and verify it succeeds."""
    from aster import ir

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        asm = build_and_compile_copy_kernel(
            "copy_asm", Layout(sizes=(4, 16), strides=(16, 64))
        )
        assert "global_load_dwordx4" in asm
        assert "global_store_dwordx4" in asm
        assert "s_endpgm" in asm


# ---------------------------------------------------------------------------
# Standalone: python test_layout_codegen.py  -- prints IR + pipeline + ASM
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from aster import ir

    layout = Layout(sizes=(4, 16), strides=(16, 64))

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_copy_kernel("copy_demo", layout)
        print(module)
        asm = compile_mlir_module_to_asm(module, print_ir_after_all=True)
        print(asm)
