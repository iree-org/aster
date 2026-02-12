"""Integration test for MFMA end-to-end kernel execution."""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE


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

    compile_and_run(
        mlir_filename,
        kernel_name,
        input_data=[a_data, b_data],
        output_data=[c_data],
        pass_pipeline=pass_pipeline,
        mcpu=mcpu,
        wavefront_size=wavefront_size,
        verify_fn=verify_fn,
        library_paths=[],
        skip_on_cross_compile=True,
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
