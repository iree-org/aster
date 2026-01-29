"""Unit tests for loop execution."""

import os
import pytest
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    hsaco_file,
)

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


# Loop pass pipeline
LOOP_PASS_PIPELINE = (
    "builtin.module("
    "  aster-optimize-arith,"
    "  func.func(aster-amdgcn-set-abi),"
    # Convert SCF control flow to AMDGCN control flow
    "  amdgcn-convert-scf-control-flow,"
    "  canonicalize,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    "  canonicalize,cse,"
    "  aster-to-amdgcn,"
    "  amdgcn-convert-waits,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-hoist-ops"
    "    )"
    "  ),"
    "  amdgcn-register-allocation,"
    "  canonicalize,cse"
    ")"
)


def get_mlir_file():
    """Get path to the test MLIR file."""
    return os.path.join(os.path.dirname(__file__), "test_loops.mlir")


def compile_and_run(
    kernel_name: str, output_data: np.ndarray, grid_dim=(1, 1, 1), block_dim=(64, 1, 1)
):
    """Compile and run a test kernel, returning the output buffer."""
    mlir_file = get_mlir_file()

    with ir.Context() as ctx:
        asm_complete, module = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            LOOP_PASS_PIPELINE,
            ctx,
            print_ir_after_all=False,
        )

        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=MCPU, wavefront_size=WAVEFRONT_SIZE
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble kernel to HSACO")

        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=MCPU):
                print(asm_complete)
                pytest.skip(f"GPU {MCPU} not available")

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                # TODO: properly handle int arguments
                input_args=[np.array([output_data.size], dtype=np.int32)],
                output_args=[output_data],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


class TestUniformLoopLowering:
    """Test uniform loop lowering."""

    def test_uniform_loop(self):
        """The output buffer should contain [0, ..., n - 1] * 4 after execution."""
        num_threads = 64
        output = np.zeros(64, dtype=np.int32)
        compile_and_run("test_uniform_loop", output, block_dim=(num_threads, 1, 1))
        expected = np.arange(num_threads, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected * 4)
