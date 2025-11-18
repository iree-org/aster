"""Integration test for MFMA end-to-end kernel execution."""

import os
import pytest
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    DEFAULT_SROA_PASS_PIPELINE,
    hsaco_file,
)


@pytest.mark.parametrize(
    # fmt: off
    "mlir_filename,kernel_name,m,n,k,pass_pipeline",
    [
        ("mfma-e2e.mlir", "compute_kernel", 16, 16, 16, DEFAULT_SROA_PASS_PIPELINE),
        ("mfma-to-global-store-scheduled-allocated-1x1x1.mlir", "test_matmul_kernel",
         16, 16, 16, DEFAULT_SROA_PASS_PIPELINE),
    ],
    # fmt: on
)
@pytest.mark.parametrize("mcpu", ["gfx942"])
def test_mfma_e2e_kernel(
    mlir_filename, kernel_name: str, m, n, k, pass_pipeline, mcpu, wavefront_size=32
):
    """Test MFMA end-to-end kernel execution from parsed MLIR file."""

    test_dir = os.path.dirname(os.path.abspath(__file__))
    mlir_file = os.path.join(test_dir, "..", "test", "integration", mlir_filename)

    with ir.Context() as ctx:
        asm_complete, module_after_passes = compile_mlir_file_to_asm(
            mlir_file, kernel_name, pass_pipeline, ctx
        )
        print(asm_complete, flush=True)

        a_data = np.array([1.0] * m * k, dtype=np.float16)
        b_data = np.array([2.0] * k * n, dtype=np.float16)
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify_fn(input_args, output_args):
            a, b = [np.array(arg).reshape(m, k) for arg in input_args[:1]] + [
                np.array(arg).reshape(k, n) for arg in input_args[1:2]
            ]
            c = np.array(output_args[0]).reshape(m, n)
            ref = np.matmul(a, b).astype(np.float32)
            assert np.array_equal(c, ref), f"MFMA kernel failed! c: {c}, ref: {ref}"

        # Assemble to hsaco
        hsaco_path = utils.assemble_to_hsaco(
            asm_complete, target=mcpu, wavefront_size=wavefront_size
        )
        assert hsaco_path is not None, "Failed to assemble kernel to HSACO"

        with hsaco_file(hsaco_path):
            # Skip execution if GPU doesn't match
            if not utils.system_has_mcpu(mcpu=mcpu):
                print(module_after_passes)
                print(asm_complete)
                pytest.skip(
                    f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
                )

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=kernel_name,
                input_args=[a_data, b_data],
                output_args=[c_data],
                mcpu=mcpu,
                wavefront_size=wavefront_size,
                verify_fn=verify_fn,
            )


if __name__ == "__main__":
    test_mfma_e2e_kernel(
        mlir_filename="mfma-to-global-store-scheduled-allocated-1x1x1.mlir",
        kernel_name="test_matmul_kernel",
        m=16,
        n=16,
        k=16,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu="gfx942",
    )
