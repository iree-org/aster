#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Tests for KernelBuilder: Python API for constructing amdgcn kernel IR.

Layer 1: per-constructor unit tests (roundtrip via ir.Module.parse).
Layer 2: per-composite-helper IR validation via aster-opt.
Layer 3: integration test - construct buffer_load + MFMA + buffer_store kernel.
"""

import pytest
import aster.ir as ir
from aster import utils
from aster.dialects import amdgcn as amdgcn_dialect
from aster.dialects.kernel_builder import KernelBuilder
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

# ---------------------------------------------------------------------------
# Layer 1: Attribute constructor roundtrips
# ---------------------------------------------------------------------------


def _ctx():
    """Return a fresh Context with amdgcn dialect loaded."""
    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    return ctx


def test_target_attr_roundtrip():
    """#amdgcn.target<gfx942> parses and survives a module roundtrip."""
    ctx = _ctx()
    with ctx:
        attr = ir.Attribute.parse("#amdgcn.target<gfx942>")
        assert str(attr) == "#amdgcn.target<gfx942>"


def test_isa_attr_roundtrip():
    """#amdgcn.isa<cdna3> parses and survives a module roundtrip."""
    ctx = _ctx()
    with ctx:
        attr = ir.Attribute.parse("#amdgcn.isa<cdna3>")
        assert str(attr) == "#amdgcn.isa<cdna3>"


def test_buffer_arg_attr_roundtrip():
    """get_buffer_argument() produces a parseable attribute."""
    from aster.dialects.amdgcn import (
        get_buffer_argument,
        AccessKind,
        AddressSpaceKind,
        KernelArgumentFlags,
    )

    ctx = _ctx()
    with ctx:
        attr = get_buffer_argument(
            address_space=AddressSpaceKind.Global,
            access=AccessKind.ReadOnly,
            flags=KernelArgumentFlags.None_,
            ctx=ctx,
        )
        attr_str = str(attr)
        assert "buffer_arg" in attr_str
        # address_space = global may be elided as default; check access mode only
        assert "read_only" in attr_str
        # Roundtrip: parse back
        reparsed = ir.Attribute.parse(attr_str)
        assert str(reparsed) == attr_str


def test_kernel_arguments_attr_roundtrip():
    """get_kernel_arguments() produces a parseable attribute."""
    from aster.dialects.amdgcn import (
        get_buffer_argument,
        get_kernel_arguments,
        AccessKind,
        AddressSpaceKind,
        KernelArgumentFlags,
    )

    ctx = _ctx()
    with ctx:
        buf = get_buffer_argument(
            address_space=AddressSpaceKind.Global,
            access=AccessKind.ReadWrite,
            flags=KernelArgumentFlags.None_,
            ctx=ctx,
        )
        k_args = get_kernel_arguments([buf, buf], ctx=ctx)
        k_args_str = str(k_args)
        assert "kernel_arguments" in k_args_str
        reparsed = ir.Attribute.parse(k_args_str)
        assert str(reparsed) == k_args_str


def test_inst_attr_roundtrip():
    """#amdgcn.inst<s_waitcnt> roundtrips."""
    ctx = _ctx()
    with ctx:
        attr = ir.Attribute.parse("#amdgcn.inst<s_waitcnt>")
        assert "s_waitcnt" in str(attr)


# ---------------------------------------------------------------------------
# Layer 1: Register type constructor roundtrips
# ---------------------------------------------------------------------------


def test_vgpr_type_roundtrip():
    """VGPRType.get() produces a type that roundtrips through ir.Type.parse."""
    from aster._mlir_libs._amdgcn import VGPRType

    ctx = _ctx()
    with ctx:
        t = VGPRType.get(ctx, reg=None)
        t_str = str(t)
        assert "vgpr" in t_str
        reparsed = ir.Type.parse(t_str)
        assert str(reparsed) == t_str


def test_vgpr_range_type_roundtrip():
    """VGPRRangeType.get() produces a type that roundtrips."""
    from aster._mlir_libs._amdgcn import VGPRRangeType

    ctx = _ctx()
    with ctx:
        t = VGPRRangeType.get(ctx, size=4, reg=None)
        t_str = str(t)
        assert "vgpr" in t_str
        reparsed = ir.Type.parse(t_str)
        assert str(reparsed) == t_str


def test_sgpr_range_type_roundtrip():
    """SGPRRangeType.get() produces a type that roundtrips."""
    from aster._mlir_libs._amdgcn import SGPRRangeType

    ctx = _ctx()
    with ctx:
        t = SGPRRangeType.get(ctx, size=2, reg=None)
        t_str = str(t)
        assert "sgpr" in t_str
        reparsed = ir.Type.parse(t_str)
        assert str(reparsed) == t_str


# ---------------------------------------------------------------------------
# Layer 2: KernelBuilder construction (no aster-opt needed, just IR validity)
# ---------------------------------------------------------------------------


def test_kernel_builder_empty_kernel():
    """KernelBuilder produces a parseable amdgcn.module with an empty kernel."""
    ctx = _ctx()
    with ctx:
        b = KernelBuilder("mymod", "empty_kernel", target="gfx942", isa="cdna3")
        module = b.build()
        assert module is not None
        text = str(module)
        assert "amdgcn.module" in text
        assert "empty_kernel" in text


def test_kernel_builder_alloca_vgpr():
    """KernelBuilder.alloca_vgpr() produces an AllocaOp in the kernel body."""
    ctx = _ctx()
    with ctx:
        b = KernelBuilder("m", "k", target="gfx942", isa="cdna3")
        v = b.alloca_vgpr()
        assert v is not None
        module = b.build()
        text = str(module)
        assert "alloca" in text


def test_kernel_builder_alloca_sgpr():
    """KernelBuilder.alloca_sgpr() produces an AllocaOp in the kernel body."""
    ctx = _ctx()
    with ctx:
        b = KernelBuilder("m", "k", target="gfx942", isa="cdna3")
        s = b.alloca_sgpr()
        assert s is not None
        module = b.build()
        text = str(module)
        assert "alloca" in text


def test_kernel_builder_wait_vmcnt():
    """KernelBuilder.wait_vmcnt() inserts s_waitcnt vmcnt=0."""
    ctx = _ctx()
    with ctx:
        b = KernelBuilder("m", "k", target="gfx942", isa="cdna3")
        b.wait_vmcnt(0)
        module = b.build()
        text = str(module)
        assert "s_waitcnt" in text
        assert "vmcnt" in text


def test_kernel_builder_wait_lgkmcnt():
    """KernelBuilder.wait_lgkmcnt() inserts s_waitcnt lgkmcnt=0."""
    ctx = _ctx()
    with ctx:
        b = KernelBuilder("m", "k", target="gfx942", isa="cdna3")
        b.wait_lgkmcnt(0)
        module = b.build()
        text = str(module)
        assert "s_waitcnt" in text
        assert "lgkmcnt" in text


def test_kernel_builder_arith_constant():
    """KernelBuilder.constant_i32() produces an arith.constant op."""
    ctx = _ctx()
    with ctx:
        b = KernelBuilder("m", "k", target="gfx942", isa="cdna3")
        c = b.constant_i32(42)
        assert c is not None
        module = b.build()
        text = str(module)
        assert "arith.constant" in text


def test_kernel_builder_with_ptr_args():
    """KernelBuilder with two pointer args produces load_arg ops."""
    from aster.dialects.amdgcn import AccessKind

    ctx = _ctx()
    with ctx:
        b = KernelBuilder("m", "k", target="gfx942", isa="cdna3")
        b.add_ptr_arg(AccessKind.ReadOnly)
        b.add_ptr_arg(AccessKind.ReadWrite)
        ptrs = b.load_args()
        assert len(ptrs) == 2
        module = b.build()
        text = str(module)
        assert "amdgcn.load_arg" in text


# ---------------------------------------------------------------------------
# Layer 3: Integration - buffer_load + MFMA + buffer_store kernel structure
# ---------------------------------------------------------------------------


def compile_to_asm(module: ir.Module) -> str:
    """Run DEFAULT_SROA_PASS_PIPELINE and return the assembly string.

    Must be called inside an active ``with ctx:`` block.
    """
    from aster._mlir_libs._mlir import passmanager

    ctx = ir.Context.current
    pm = passmanager.PassManager.parse(DEFAULT_SROA_PASS_PIPELINE, ctx)
    pm.run(module.operation)

    amdgcn_mod = None
    for op in module.body:
        if isinstance(op, amdgcn_dialect.ModuleOp):
            amdgcn_mod = op
            break
    assert amdgcn_mod is not None
    return utils.translate_module(amdgcn_mod)


def build_tiledmma_module(target: str = "gfx942", isa: str = "cdna3") -> ir.Module:
    """Build a tiledmma kernel: A[64x4 f16] x B[64x4 f16] -> C[64x4 f32].

    Shared helper used by the unit structure test and the E2E execution test.
    Must be called inside an active ``with ctx:`` block.
    """
    from aster.dialects.amdgcn import AccessKind

    b = KernelBuilder("tiledmma_mod", "tiledmma", target=target, isa=isa)
    b.add_ptr_arg(AccessKind.ReadOnly)  # A matrix ptr
    b.add_ptr_arg(AccessKind.ReadOnly)  # B matrix ptr
    b.add_ptr_arg(AccessKind.WriteOnly)  # C (output) ptr

    a_ptr, b_ptr, c_ptr = b.load_args()

    tid = b.thread_id_x()  # index -- lowered by pipeline
    ab_voff = b.byte_offset(tid, 8)  # 4 f16 = 8 bytes per thread
    c_voff = b.byte_offset(tid, 16)  # 4 f32 = 16 bytes per thread
    num_records = b.s_mov_b32(65536)
    stride = b.constant_i32(0)
    soffset = b.s_mov_b32(0)

    a_rsrc = b.make_buffer_rsrc(a_ptr, num_records, stride)
    b_rsrc = b.make_buffer_rsrc(b_ptr, num_records, stride)
    c_rsrc = b.make_buffer_rsrc(c_ptr, num_records, stride)

    a_frag = b.buffer_load_dwordx2(a_rsrc, soffset, ab_voff)
    b_frag = b.buffer_load_dwordx2(b_rsrc, soffset, ab_voff)
    b.wait_vmcnt(0)

    acc = b.init_agprx4(b.constant_i32(0))
    acc = b.mfma("v_mfma_f32_16x16x16_f16", acc, a_frag, b_frag)

    b.buffer_store_dwordx4(acc, c_rsrc, soffset, c_voff)
    b.wait_vmcnt(0)

    return b.build()


def test_kernel_builder_tiledmma_structure():
    """build_tiledmma_module() produces the expected IR and compiles to asm."""
    ctx = _ctx()
    with ctx:
        module = build_tiledmma_module()
        text = str(module)

        assert "amdgcn.module" in text
        assert "tiledmma" in text
        assert "amdgcn.load_arg" in text
        assert "gpu.thread_id" in text
        assert "make_buffer_rsrc" in text
        assert "buffer_load_dwordx2" in text
        assert "s_waitcnt" in text
        assert "v_accvgpr_write_b32" in text
        assert "vop3p_mai" in text
        assert "buffer_store_dwordx4" in text

        asm = compile_to_asm(module)
        assert "v_mfma_f32_16x16x16_f16" in asm
        assert "buffer_load_dwordx2" in asm
        return asm


if __name__ == "__main__":
    print(test_kernel_builder_tiledmma_structure())
