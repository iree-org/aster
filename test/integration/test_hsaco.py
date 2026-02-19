"""Test HSACO generation for all supported targets via ModuleTranslation."""

import os

import pytest

from aster import ir, utils
from aster.dialects import amdgcn

TARGET_CONFIGS = [
    ("gfx942", "cdna3"),
    ("gfx950", "cdna4"),
    ("gfx1201", "rdna4"),
]


def _build_module_ir(target, isa):
    """Return MLIR source for a kernel that uses instructions available on all ISAs.

    Uses s_mov_b32 (scalar move, all ISAs) and alloca/end_kernel (universal) to exercise
    the regalloc -> ASM -> HSACO pipeline with a non-trivial instruction sequence.
    """
    return f"""\
amdgcn.module @test_module target = #amdgcn.target<{target}> isa = #amdgcn.isa<{isa}> {{
  amdgcn.kernel @test_kernel {{
    %s0 = amdgcn.alloca : !amdgcn.sgpr<4>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<5>
    %c42 = arith.constant 42 : i32
    %c7 = arith.constant 7 : i32
    amdgcn.sop1 s_mov_b32 outs %s0 ins %c42 : !amdgcn.sgpr<4>, i32
    amdgcn.sop1 s_mov_b32 outs %s1 ins %c7 : !amdgcn.sgpr<5>, i32
    amdgcn.end_kernel
  }}
}}
"""


def _run_pipeline(module, ctx):
    """Translate module to ASM (registers are pre-assigned, no regalloc needed)."""
    amdgcn_mod = None
    for op in module.body:
        if isinstance(op, amdgcn.ModuleOp):
            amdgcn_mod = op
            break
    assert amdgcn_mod is not None, "Failed to find AMDGCN module"

    return utils.translate_module(amdgcn_mod)


@pytest.mark.parametrize(
    "target,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS]
)
def test_translate_to_asm(target, isa):
    """Test MLIR -> regalloc -> ASM translation for each target."""
    with ir.Context(), ir.Location.unknown():
        ctx = ir.Context.current
        module = ir.Module.parse(_build_module_ir(target, isa))
        asm = _run_pipeline(module, ctx)

        assert f"amdgcn-amd-amdhsa--{target}" in asm
        assert "s_mov_b32" in asm
        assert "s_endpgm" in asm
        assert ".amdhsa_kernel test_kernel" in asm


@pytest.mark.parametrize(
    "target,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS]
)
def test_assemble_to_hsaco(target, isa):
    """Test MLIR -> regalloc -> ASM -> HSACO assembly for each target."""
    with ir.Context(), ir.Location.unknown():
        ctx = ir.Context.current
        module = ir.Module.parse(_build_module_ir(target, isa))
        asm = _run_pipeline(module, ctx)

        hsaco_path = utils.assemble_to_hsaco(asm, target=target)
        if hsaco_path is None:
            pytest.skip(
                f"LLVM assembler does not support {target} "
                f"(rebuild LLVM with a version that includes {target} support)"
            )
        try:
            assert os.path.exists(hsaco_path), f"HSACO not created for {target}"
            assert os.path.getsize(hsaco_path) > 0, f"HSACO is empty for {target}"
        finally:
            if hsaco_path and os.path.exists(hsaco_path):
                os.unlink(hsaco_path)


if __name__ == "__main__":
    test_translate_to_asm("gfx942", "cdna3")
    test_translate_to_asm("gfx950", "cdna4")
    test_translate_to_asm("gfx1201", "rdna4")
    test_assemble_to_hsaco("gfx942", "cdna3")
    test_assemble_to_hsaco("gfx950", "cdna4")
    test_assemble_to_hsaco("gfx1201", "rdna4")
