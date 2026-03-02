"""Test: Zero-initialized C tile should contain all zeros."""

import numpy as np

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensZeroC:
    """Test @zero_C function from kittens/global_16x16_f16.mlir."""

    def test_zero_C_produces_zeros(self):
        """Zero-initialized C tile should contain all zeros."""
        output = np.zeros(16 * 16, dtype=np.int32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_000_zero_c.mlir"),
            kernel_name="test_zero_C",
            input_args=[],
            output_args=[output],
        )

        expected = np.zeros(16 * 16, dtype=np.int32)
        np.testing.assert_array_equal(output, expected)
