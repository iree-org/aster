# RUN: %PYTHON %s | FileCheck %s

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aster import ir, utils
from aster.dialects import amdgcn, builtin


def test_hsaco_generation():
    """Test translation of AMDGCN module to hsaco file."""
    with ir.Context() as ctx, ir.Location.unknown():
        # Build the test module from existing test
        module = ir.Module.parse(
            """
amdgcn.module @test_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @ds_all_kernel {
    %addr = amdgcn.alloca : !amdgcn.vgpr
    %vgpr0 = amdgcn.alloca : !amdgcn.vgpr

    // Read from LDS
    %data, %tok = load ds_read_b32 dest %vgpr0 addr %addr : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr) -> !amdgcn.read_token<shared>

    // Write to LDS
    store ds_write_b32 data %data addr %addr : ins(!amdgcn.vgpr, !amdgcn.vgpr) -> !amdgcn.write_token<shared>

    end_kernel
  }
}
"""
        )

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
        pm.run(module.operation)

        # Translate to assembly
        asm = utils.translate_module(amdgcn_mod)
        print(asm)

        # CHECK-LABEL: .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
        # CHECK: .text
        # CHECK: .globl ds_all_kernel
        # CHECK: ds_all_kernel:
        # CHECK: ds_read_b32
        # CHECK: ds_write_b32
        # CHECK: s_endpgm

        hsaco_path = utils.assemble_to_hsaco(asm, target="gfx942")
        assert os.path.exists(hsaco_path), f"HSACO file {hsaco_path} was not created"
        print(f"Successfully generated HSACO: {hsaco_path}")
        os.unlink(hsaco_path)


if __name__ == "__main__":
    test_hsaco_generation()
