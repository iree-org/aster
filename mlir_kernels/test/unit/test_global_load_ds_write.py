"""Unit tests for copies.mlir library functions."""

import numpy as np

from aster.testing import compile_and_run


class TestGlobalLoadDsWrite:
    """Test @global_load_wave_256xf16_via_dwordx2_wait + @ds_write_dwordx2_wait functions."""

    def test_decoupled_load_store(self):
        """Load from global via memref, write to LDS, verify roundtrip."""
        num_threads = 64
        # Input: 16x16 matrix of f16 values
        input_data = np.arange(16 * 16, dtype=np.uint16)
        # Output: each thread reads back 8 bytes (4 f16 values)
        output = np.zeros(num_threads * 4, dtype=np.uint16)

        compile_and_run(
            "test_global_load_ds_write.mlir",
            "test_global_load_ds_write",
            [input_data],
            output,
        )

        # The data flow is identical to global_load_to_lds_wave_16x16_f16_wait
        expected = np.zeros(num_threads * 4, dtype=np.uint16)
        for tid in range(num_threads):
            lane_id = tid % 64
            iii = lane_id // 4
            jjj_base = lane_id % 4
            jjj = jjj_base * 4
            for k in range(4):
                src_idx = iii * 16 + jjj + k
                dst_idx = tid * 4 + k
                expected[dst_idx] = src_idx

        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    # Run all tests
    TestGlobalLoadDsWrite().test_decoupled_load_store()
