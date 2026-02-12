"""Unit tests for LDS bank conflict analysis using indexing.mlir functions."""

import numpy as np

from aster.testing import compile_and_run, make_grid_block_preprocess


def analyze_bank_conflicts(banks, title=""):
    """Analyze and print bank conflict information.

    Args:
        banks: Array of shape (64, 4) where banks[tid] = [b0, b1, b2, b3]
               for the 4 banks accessed by each thread's b64 load.
        title: Description for the output.
    """
    print(f"\n{'='*70}")
    print(f"LDS Bank Analysis: {title}")
    print(f"{'='*70}")

    # Print banks accessed by each thread
    print("\nBanks accessed per thread (tid: [b0, b1, b2, b3]):")
    for tid in range(64):
        b = banks[tid]
        print(f"  lane {tid:2d}: [{b[0]:2d}, {b[1]:2d}, {b[2]:2d}, {b[3]:2d}]", end="")
        if (tid + 1) % 4 == 0:
            print()

    # Most important: check if different threads access the same bank
    # Group threads by their first bank to detect potential conflicts
    print("\n\nPotential bank conflicts (threads accessing same bank):")
    for bank in range(32):
        threads_using_bank = []
        for tid in range(64):
            if bank in banks[tid]:
                threads_using_bank.append(tid)
        if len(threads_using_bank) > 4:  # More than 4 threads = definite conflict
            print(
                f"  bank {bank:2d}: {len(threads_using_bank)} threads -> {threads_using_bank}"
            )


class TestLdsBanks:
    """Test LDS bank computation functions for debugging bank conflicts."""

    def test_lds_banks_A_16x16xf16(self):
        """Test banks for non-swizzled MFMA A matrix pattern."""
        num_threads = 64
        grid_dim, block_dim = (1, 1, 1), (64, 1, 1)
        # Output: 4 banks per thread (b64 = 8 bytes = 4 x 2-byte banks)
        output = np.zeros(num_threads * 4, dtype=np.int32)
        compile_and_run(
            "test_lds_banks.mlir",
            "test_lds_banks_A_16x16xf16",
            output_data=output,
            grid_dim=grid_dim,
            block_dim=block_dim,
            preprocess=make_grid_block_preprocess(grid_dim, block_dim),
        )

        banks = output.reshape(64, 4)
        # fmt: off
        expected = np.array([
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 0-3
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 4-7
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 8-11
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 12-15
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 16-19
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 20-23
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 24-27
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 28-31
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 32-35
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 36-39
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 40-43
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 44-47
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 48-51
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 52-55
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 56-59
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 60-63
        ], dtype=np.int32)
        # fmt: on
        np.testing.assert_array_equal(banks, expected, "Non-swizzled bank mismatch")

        analyze_bank_conflicts(banks, "Non-swizzled MFMA A 16x16xf16 (b64)")

    def test_lds_banks_swizzled_A_16x16xf16(self):
        """Test banks for swizzled MFMA A matrix pattern."""
        num_threads = 64
        grid_dim, block_dim = (1, 1, 1), (64, 1, 1)
        output = np.zeros(num_threads * 4, dtype=np.int32)
        compile_and_run(
            "test_lds_banks.mlir",
            "test_lds_banks_swizzled_A_16x16xf16",
            output_data=output,
            grid_dim=grid_dim,
            block_dim=block_dim,
            preprocess=make_grid_block_preprocess(grid_dim, block_dim),
        )

        banks = output.reshape(64, 4)

        # Verify swizzled bank computation:
        #   swizzled_col = (col_high XOR row_group) * 4 + col_low
        # where:
        #   row = 4 * (tid // 16), col = tid % 16,
        #   row_group = row // 4 = tid // 16,
        #   col_high = col // 4, col_low = col % 4
        # fmt: off
        expected = np.array([
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 0-3
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 4-7
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 8-11
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 12-15
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 16-19
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 20-23
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 24-27
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 28-31
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 32-35
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 36-39
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 40-43
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 44-47
            [12, 13, 14, 15], [28, 29, 30, 31], [12, 13, 14, 15], [28, 29, 30, 31],  # tid 48-51
            [ 8,  9, 10, 11], [24, 25, 26, 27], [ 8,  9, 10, 11], [24, 25, 26, 27],  # tid 52-55
            [ 4,  5,  6,  7], [20, 21, 22, 23], [ 4,  5,  6,  7], [20, 21, 22, 23],  # tid 56-59
            [ 0,  1,  2,  3], [16, 17, 18, 19], [ 0,  1,  2,  3], [16, 17, 18, 19],  # tid 60-63
        ], dtype=np.int32)
        # fmt: on
        np.testing.assert_array_equal(banks, expected, "Swizzled bank mismatch")

        analyze_bank_conflicts(banks, "Swizzled MFMA A 16x16xf16 (b64)")


if __name__ == "__main__":
    # TestLdsBanks().test_lds_banks_A_16x16xf16()
    TestLdsBanks().test_lds_banks_swizzled_A_16x16xf16()
