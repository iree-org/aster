"""Integration test for MFMA end-to-end kernel execution."""

import os
from integration_test.test_utils import execute_kernel_and_verify
import pytest
import numpy as np

from aster import ir, utils
from aster.dialects import amdgcn


def load_mlir_module_from_file(file_path: str, ctx):
    from aster._mlir_libs._mlir import ir as mlir_ir

    with open(file_path, "r") as f:
        mlir_content = f.read()
    with mlir_ir.Location.unknown():
        module = mlir_ir.Module.parse(mlir_content, context=ctx)
    return module


def compile_mlir_file_to_asm(mlir_file: str, ctx) -> tuple[str, str]:
    module = load_mlir_module_from_file(mlir_file, ctx)

    # Apply passes: inline, register allocation, symbol dce
    from aster._mlir_libs._mlir import passmanager

    pm = passmanager.PassManager.parse(
        "builtin.module(inline,amdgcn.module(amdgcn-register-allocation),symbol-dce)",
        ctx,
    )
    pm.run(module.operation)

    # get the amdgcn.module from the module
    amdgcn_module = None
    kernel_name = "compute_kernel"  # hardcoded for now, could get from aster.kernel
    for op in module.body:
        if isinstance(op, amdgcn.ModuleOp):
            amdgcn_module = op
            break
    assert amdgcn_module is not None, "Failed to find AMDGCN module"

    asm_complete = utils.translate_module(
        amdgcn_module,
        debug_print=False,
    )
    return asm_complete, kernel_name


@pytest.mark.parametrize("mcpu", ["gfx942"])
def test_mfma_e2e_kernel(mcpu, wavefront_size=32):
    """Test MFMA end-to-end kernel execution from parsed MLIR file."""

    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(test_dir, "..", "test", "integration", "mfma-e2e.mlir")

    with ir.Context() as ctx:
        asm_complete, kernel_name = compile_mlir_file_to_asm(mlir_file, ctx)
        print(asm_complete)

        a_data = np.array([1.0] * 16 * 16, dtype=np.float16)
        b_data = np.array([2.0] * 16 * 16, dtype=np.float16)
        c_data = np.zeros(16 * 16, dtype=np.float32)

        def verify_fn(input_args, output_args):
            a, b = [np.array(arg).reshape(16, 16) for arg in input_args]
            c = np.array(output_args[0]).reshape(16, 16)
            ref = np.matmul(a, b).astype(np.float32)
            print(c)
            print(ref)
            assert np.array_equal(c, ref), "MFMA kernel failed!"

        # Skip execution if GPU doesn't match
        if not utils.system_has_mcpu(mcpu=mcpu):
            pytest.skip(f"GPU {mcpu} not available")

        execute_kernel_and_verify(
            asm_code=asm_complete,
            kernel_name=kernel_name,
            input_args=[a_data, b_data],
            output_args=[c_data],
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            verify_fn=verify_fn,
        )


if __name__ == "__main__":
    test_mfma_e2e_kernel("gfx942")
