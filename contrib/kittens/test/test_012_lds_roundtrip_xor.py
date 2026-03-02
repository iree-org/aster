"""Test: LDS roundtrip with XOR swizzle."""

import numpy as np

from kittens_helpers import run_kittens_kernel, get_mlir_file


class TestKittensLDSRoundtripXorSwizzle:
    """Test LDS roundtrip with XOR swizzle: Global -> LDS -> Register -> Global."""

    def test_lds_roundtrip_xor_swizzle_f16(self):
        """Data should survive Global -> LDS (XOR swizzle) -> Register -> Global path."""
        input_f16 = np.arange(16 * 16, dtype=np.float16)
        input_data = input_f16.view(np.uint16)
        output_data = np.full(16 * 16, 0xFFFF, dtype=np.uint16)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_012_lds_roundtrip_xor.mlir"),
            kernel_name="test_lds_roundtrip_xor_swizzle",
            input_args=[input_data],
            output_args=[output_data],
        )

        np.testing.assert_array_equal(output_data, input_data)
