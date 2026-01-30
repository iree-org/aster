"""Unit tests for maybe_global_load_multi_tile_coalesced and maybe_lds_write_multi_tile_coalesced."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


class TestMaybeMultiTileCoalesced:
    """Test the maybe_*_multi_tile_coalesced library functions (bulk version).

    This tests the bulk multi-tile functions that use
    global_load_wave_multi_tile_256xf16_via_dwordx2_wait and
    lds_write_wave_multi_tile_256xf16_via_dwordx2_wait.
    """

    def test_multi_tile_coalesced_with_nt_2x4(self):
        """Test with NT_I=2, NT_J=4 on a 64x128 array (4x8 tiles)."""
        self._run_test(rows=64, cols=128, nt_i=2, nt_j=4)

    def _run_test(self, rows: int, cols: int, nt_i: int, nt_j: int):
        """Run multi-tile coalesced test with configurable parameters."""
        ii = rows // 16  # Total tiles in I dimension
        jj = cols // 16  # Total tiles in J dimension
        global_stride_bytes = cols * 2  # f16 = 2 bytes
        nt_product = nt_i * nt_j

        def preprocess(mlir: str) -> str:
            return (
                mlir.replace("{{SIZE_J}}", str(cols))
                .replace("{{II}}", str(ii))
                .replace("{{JJ}}", str(jj))
                .replace("{{NT_I}}", str(nt_i))
                .replace("{{NT_J}}", str(nt_j))
                .replace("{{GLOBAL_STRIDE_BYTES}}", str(global_stride_bytes))
                .replace("{{NT_PRODUCT}}", str(nt_product))
            )

        input_data = np.arange(rows * cols, dtype=np.uint16)
        output = np.zeros(rows * cols, dtype=np.uint16)

        compile_and_run(
            "test_maybe_multi_tile_coalesced.mlir",
            "test_maybe_multi_tile_coalesced",
            [input_data],
            output,
            preprocess=preprocess,
            print_ir_after_all=False,
        )

        # Verify output matches input (identity copy through LDS)
        input_2d = input_data.reshape(rows, cols)
        output_2d = output.reshape(rows, cols)

        for ti in range(ii):
            for tj in range(jj):
                r0, r1 = ti * 16, (ti + 1) * 16
                c0, c1 = tj * 16, (tj + 1) * 16
                tile_in = input_2d[r0:r1, c0:c1]
                tile_out = output_2d[r0:r1, c0:c1]
                match = np.array_equal(tile_in, tile_out)
                if not match:
                    print(f"Tile ({ti},{tj}) input:\n {tile_in}")
                    print(f"Tile ({ti},{tj}) output:\n {tile_out}")
                    print(f"Tile ({ti},{tj}) diff:\n {tile_in - tile_out}")

        np.testing.assert_array_equal(output, input_data)


if __name__ == "__main__":
    # Run all tests
    TestMaybeMultiTileCoalesced().test_multi_tile_coalesced_with_nt_2x4()
