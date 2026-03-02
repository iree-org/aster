"""Test: Global load/store roundtrip preserves data."""

import numpy as np

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensLoadStoreA:
    """Test @load_A_f16 and @store_A_f16 functions from kittens/global_16x16_f16.mlir."""

    def test_load_store_roundtrip(self):
        """Load A tile and store it back - should preserve original data."""
        input_f16 = np.arange(16 * 16, dtype=np.float16)
        input_data = input_f16.view(np.uint16)
        output_data = np.full(16 * 16, 0xFFFF, dtype=np.uint16)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_001_load_store_a.mlir"),
            kernel_name="test_load_store_A",
            input_args=[input_data],
            output_args=[output_data],
        )

        np.testing.assert_array_equal(output_data, input_data)
