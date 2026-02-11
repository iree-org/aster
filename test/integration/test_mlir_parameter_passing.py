"""Integration test for GPU kernel parameter passing using MLIR."""

import pytest
import numpy as np

from aster import utils
import aster.ir as ir
from aster.dialects import amdgcn
from aster.dialects._amdgcn_ops_gen import (
    ModuleOp,
    KernelOp,
    EndKernelOp,
)
from aster.dialects._amdgcn_enum_gen import Target, ISAVersion
from aster.dialects.api import (
    alloca_sgpr,
    alloca_vgpr,
    make_register_range,
    s_load_dwordx2,
    global_load_dwordx2,
    global_store_dwordx2,
    v_mov_b32_e32,
    s_waitcnt,
)
from integration.test_utils import execute_kernel_and_verify, hsaco_file


def build_copy_kernel_module(ctx: ir.Context, kernel_name: str, mcpu: str):
    # Determine target and ISA version from mcpu
    if mcpu == "gfx942":
        target = Target.GFX942
        isa_version = ISAVersion.CDNA3
    elif mcpu == "gfx1201":
        target = Target.GFX1201
        isa_version = ISAVersion.RDNA4
    else:
        raise ValueError(f"Unsupported mcpu: {mcpu}")

    with ir.Location.unknown():
        # Create the module
        module = ModuleOp(
            target=target, isa_version=isa_version, sym_name=kernel_name + "_module"
        )
        module.body_region.blocks.append()

        with ir.InsertionPoint(module.body_region.blocks[0]):
            # Create kernel
            kernel = KernelOp(kernel_name)
            kernel.body_region.blocks.append()

            with ir.InsertionPoint(kernel.body_region.blocks[0]):
                d_sgprs = [alloca_sgpr(reg=i) for i in range(2, 20)]

                # Allocate registers - SGPRs for kernel arguments
                s0 = alloca_sgpr(0)
                s1 = alloca_sgpr(1)
                kernarg_ptr = make_register_range([s0, s1]).result

                # Load source pointer from kernarg at offset 0
                src_ptr = make_register_range(d_sgprs[0:2])
                src_ptr = s_load_dwordx2(src_ptr, kernarg_ptr, offset=0).result

                # Load destination pointer from kernarg at offset 8
                dst_ptr = make_register_range(d_sgprs[2:4])
                dst_ptr = s_load_dwordx2(dst_ptr, kernarg_ptr, offset=8).result
                s_waitcnt(lgkmcnt=0)

                # Extract individual SGPRs from ranges
                s2, s3, s4, s5 = [alloca_sgpr(reg) for reg in (2, 3, 4, 5)]

                # Allocate VGPRs for moving pointers
                v10, v11, v12, v13, v14, v15 = [
                    alloca_vgpr(reg) for reg in range(10, 16)
                ]

                # Move source pointer components to VGPRs
                v10 = v_mov_b32_e32(v10, s2)
                v11 = v_mov_b32_e32(v11, s3)
                v10_11 = make_register_range([v10, v11]).result

                # Load 2 dwords from source
                v14_15 = make_register_range([v14, v15]).result
                data = global_load_dwordx2(v14_15, v10_11, offset=0).result
                s_waitcnt(vmcnt=0)

                # Store 2 dwords to destination
                v12 = v_mov_b32_e32(v12, s4)
                v13 = v_mov_b32_e32(v13, s5)
                v12_13 = make_register_range([v12, v13]).result
                global_store_dwordx2(data, v12_13, offset=0)
                s_waitcnt(vmcnt=0)

                # End kernel
                EndKernelOp()

    return module, kernel


@pytest.mark.parametrize("mcpu", ["gfx1201", "gfx942"])
def test_mlir_copy_kernel(mcpu, wavefront_size=32):
    """Test parameter passing using MLIR-generated kernels."""

    # Create MLIR context and build module
    with ir.Context() as ctx:
        # Register AMDGCN dialect
        ctx.allow_unregistered_dialects = True

        kernel_name = "copy_kernel_mlir"
        module, kernel = build_copy_kernel_module(ctx, kernel_name, mcpu)

        kernel.wavefront_size32 = wavefront_size == 32

        kernel_args = [
            amdgcn.get_buffer_argument(
                address_space=amdgcn.AddressSpaceKind.Global,
                access=amdgcn.AccessKind.ReadOnly,
                flags=amdgcn.KernelArgumentFlags.None_,
                name="src_ptr",
            ),
            amdgcn.get_buffer_argument(
                address_space=amdgcn.AddressSpaceKind.Global,
                access=amdgcn.AccessKind.WriteOnly,
                flags=amdgcn.KernelArgumentFlags.None_,
                name="dst_ptr",
            ),
        ]
        kernel.arguments = amdgcn.get_kernel_arguments(kernel_args)

        # Translate to assembly with metadata
        asm_complete = utils.translate_module(
            module,
            debug_print=False,
        )
        print(asm_complete)

        # Prepare test data
        input_data = np.array([1, 2], dtype=np.int32)
        output_data = np.zeros(2, dtype=np.int32)

        def verify_fn(input_args, output_args):
            assert np.array_equal(input_args[0], output_args[0]), "Copy kernel failed!"

        # Assemble to hsaco
        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=mcpu, wavefront_size=wavefront_size
        )
        assert hsaco_path is not None, "Failed to assemble kernel to HSACO"

        with hsaco_file(hsaco_path):
            # Skip execution if GPU doesn't match
            if not utils.system_has_mcpu(mcpu=mcpu):
                print(module)
                print(asm_complete)
                pytest.skip(
                    f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
                )

            # Execute kernel and verify results
            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=[input_data],
                output_args=[output_data],
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                verify_fn=verify_fn,
            )


if __name__ == "__main__":
    test_mlir_copy_kernel("gfx942")
    test_mlir_copy_kernel("gfx1201")
