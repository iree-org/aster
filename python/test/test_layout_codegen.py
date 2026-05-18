# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Tests for layout-driven MLIR code generation.

from __future__ import annotations

from aster.layout import Layout
from aster.compiler.core import compile_mlir_module_to_asm

# ---------------------------------------------------------------------------
# Shared helpers (imported by integration/test_layout_copy_e2e.py)
# ---------------------------------------------------------------------------


def build_copy_kernel(
    name: str,
    thread_layout: Layout,
    target: str = "gfx942",
):
    """Build a flat-global dwordx4 copy kernel using KernelBuilder + layout."""
    from aster.dialects.kernel_builder import KernelBuilder
    from aster.dialects.amdgcn import AccessKind

    b = KernelBuilder(f"{name}_module", name, target=target)
    b.add_ptr_arg(AccessKind.ReadOnly)
    b.add_ptr_arg(AccessKind.WriteOnly)
    src_ptr, dst_ptr = b.load_args()

    tid = b.thread_id("x")
    byte_off = b.layout_apply(tid, thread_layout)

    src_addr = b.global_addr(src_ptr, byte_off)
    data, load_tok = b.global_load_dwordx4(src_addr)
    b.wait_deps(load_tok)

    dst_addr = b.global_addr(dst_ptr, byte_off)
    b.global_store_dwordx4(data, dst_addr)

    return b.build()


def build_and_compile_copy_kernel(
    name: str,
    thread_layout: Layout,
    print_ir_after_all: bool = False,
) -> str:
    """Build a copy kernel and compile to assembly. Returns asm string.

    Must be called inside an active ``with ir.Context():`` block.
    """
    from aster.compiler.core import PrintOptions

    module = build_copy_kernel(name, thread_layout)
    return compile_mlir_module_to_asm(
        module,
        print_opts=PrintOptions.from_flags(print_ir_after_all=print_ir_after_all),
    )


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
        assert "layout.apply" in text


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


def test_swizzled_layout_lower_two_part_address():
    """SwizzledLayout.lower emits the irreducible unswizzled + XOR-swizzled sum."""
    from aster import ir
    from aster.layout import Swizzle, SwizzledLayout
    from aster.dialects.kernel_builder import KernelBuilder
    from aster.dialects.amdgcn import AccessKind

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        b = KernelBuilder("swz_module", "swz", target="gfx942")
        b.add_ptr_arg(AccessKind.ReadOnly)
        tid = b.thread_id("x")
        sl = SwizzledLayout(
            unswizzled_base=Layout(sizes=(4,), strides=(64,)),
            swizzled_base=Layout(sizes=(4,), strides=(16,)),
            swizzle_spec=Swizzle(bits=2, base=0, shift=3),
        )
        off = b.layout_apply(tid, sl)
        text = str(b.build())
        assert text.count("layout.apply") == 2
        assert "layout.swizzle" in text
        assert "affine.apply" in text
        assert off is not None


def test_swizzled_layout_dispatch_matches_manual():
    """Polymorphic layout_apply(SwizzledLayout) == manual two-call form."""
    from aster import ir
    from aster.layout import Swizzle, SwizzledLayout
    from aster.dialects.kernel_builder import KernelBuilder
    from aster.dialects.amdgcn import AccessKind

    unswz = Layout(sizes=(4,), strides=(64,))
    swz = Layout(sizes=(4,), strides=(16,))
    spec = Swizzle(bits=2, base=0, shift=3)

    def emit(use_swizzled_layout: bool) -> str:
        ctx = ir.Context()
        ctx.allow_unregistered_dialects = True
        with ctx:
            b = KernelBuilder("m", "k", target="gfx942")
            b.add_ptr_arg(AccessKind.ReadOnly)
            tid = b.thread_id("x")
            if use_swizzled_layout:
                b.layout_apply(
                    tid,
                    SwizzledLayout(
                        unswizzled_base=unswz, swizzled_base=swz, swizzle_spec=spec
                    ),
                )
            else:
                u = b.layout_apply(tid, unswz)
                s = b.layout_apply(tid, swz, swizzle=spec)
                d0, d1 = ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)
                b.affine_apply(d0 + d1, [u, s])
            return str(b.build())

    assert emit(True) == emit(False)


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
        from aster.compiler.core import PrintOptions

        asm = compile_mlir_module_to_asm(
            module, print_opts=PrintOptions(print_ir_after_all=True)
        )
        print(asm)
