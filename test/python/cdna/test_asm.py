# RUN: %PYTHON %s | FileCheck %s

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aster import ir, utils
from aster.dialects import amdgcn, builtin
from cdna.test_cdna import build_test_module


def test_asm_translation():
    """Test translation of AMDGCN module to assembly."""
    with ir.Context() as ctx, ir.Location.unknown():
        # Build the test module
        module = build_test_module(ctx)

        # Find the AMDGCN module
        amdgcn_mod = None
        for op in module.body:
            if isinstance(op, amdgcn.ModuleOp):
                amdgcn_mod = op
                break

        assert amdgcn_mod is not None, "Failed to find AMDGCN module"

        # Run register allocation pass
        from aster._mlir_libs._mlir import passmanager

        pm = passmanager.PassManager.parse(
            "builtin.module(amdgcn.module(amdgcn-register-allocation))", ctx
        )
        pm.run(module)

        # Translate to assembly
        asm = utils.translate_module(amdgcn_mod)

        # Print the assembly
        print(asm)

        # CHECK-LABEL: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
        # CHECK: .text
        # CHECK: .globl ds_all_kernel
        # CHECK: .p2align 8
        # CHECK: .type ds_all_kernel,@function

        # CHECK: ds_all_kernel:
        # CHECK: ds_read_b32 v[[REG1:[0-9]+]], v[[REG2:[0-9]+]]
        # CHECK: ds_write_b32 v[[REG2]], v[[REG1]]
        # CHECK: ds_read_b64 v[[[REG3:[0-9]+:[0-9]+]]], v[[REG2]]
        # CHECK: ds_write_b64 v[[REG2]], v[[[REG3]]]
        # CHECK: ds_read_b96 v[[[REG4:[0-9]+:[0-9]+]]], v[[REG2]] offset:4
        # CHECK: ds_write_b96 v[[REG2]], v[[[REG4]]] offset:4
        # CHECK: ds_read_b128 v[[[REG5:[0-9]+:[0-9]+]]], v[[REG2]] offset:8
        # CHECK: ds_write_b128 v[[REG2]], v[[[REG5]]] offset:8
        # CHECK: s_endpgm

        # CHECK: .text
        # CHECK: .globl vop3p_mai_kernel
        # CHECK: vop3p_mai_kernel:

        # CHECK: v_mfma_f32_16x16x16_f16 v[[[REG6:[0-9]+:[0-9]+]]], v[[[REG7:[0-9]+:[0-9]+]]], v[[[REG8:[0-9]+:[0-9]+]]], v[[[REG9:[0-9]+:[0-9]+]]]
        # CHECK: v_mfma_f32_16x16x16_bf16 v[[[REG10:[0-9]+:[0-9]+]]], v[[[REG7]]], v[[[REG8]]], v[[[REG9]]]
        # CHECK: s_endpgm

        # CHECK: .text
        # CHECK: .globl smem_all_kernel
        # CHECK: smem_all_kernel:
        # CHECK: s_load_dword
        # CHECK: s_store_dword
        # CHECK: s_load_dwordx2
        # CHECK: s_store_dwordx2
        # CHECK: s_load_dwordx4
        # CHECK: s_store_dwordx4
        # CHECK: s_memtime
        # CHECK: s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
        # CHECK: s_waitcnt vmcnt(5) expcnt(2) lgkmcnt(1)
        # CHECK: s_trap 2
        # CHECK: s_barrier
        # CHECK: s_endpgm

        # CHECK: .text
        # CHECK: .globl vop2_kernel
        # CHECK: vop2_kernel:
        # CHECK: v_lshrrev_b32_e32 v[[REG11:[0-9]+]], v[[REG12:[0-9]+]], v[[REG13:[0-9]+]]
        # CHECK: v_lshrrev_b32_e32 v[[REG11]], s[[REG14:[0-9]+]], v[[REG13]]
        # CHECK: v_lshrrev_b32_e32 v[[REG11]], 8, v[[REG13]]
        # CHECK: v_lshlrev_b32_e32 v[[REG11]], v[[REG12]], v[[REG13]]
        # CHECK: v_lshlrev_b32_e32 v[[REG11]], s[[REG14]], v[[REG13]]
        # CHECK: v_lshlrev_b32_e32 v[[REG11]], 8, v[[REG13]]
        # CHECK: s_endpgm

        # CHECK: .text
        # CHECK: .globl vop1_kernel
        # CHECK: vop1_kernel:
        # CHECK: v_mov_b32_e32 v[[REG15:[0-9]+]], v[[REG16:[0-9]+]]
        # CHECK: v_mov_b32_e32 v[[REG15]], s[[REG17:[0-9]+]]
        # CHECK: v_mov_b32_e32 v[[REG15]], 42
        # CHECK: s_endpgm


if __name__ == "__main__":
    test_asm_translation()
