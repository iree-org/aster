"""Unit tests for LDS bank conflict analysis using indexing.mlir functions."""

import os
import pytest
import numpy as np

from aster import ir, utils
from integration_test.test_utils import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    SYNCHRONOUS_SROA_PASS_PIPELINE,
    hsaco_file,
)
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def get_mlir_file():
    """Get path to the test MLIR file."""
    return os.path.join(os.path.dirname(__file__), "test_lds_banks.mlir")


def compile_and_run(
    kernel_name: str, output_data: np.ndarray, grid_dim=(1, 1, 1), block_dim=(64, 1, 1)
):
    """Compile and run a test kernel, returning the output buffer."""
    mlir_file = get_mlir_file()

    def preprocess(x):
        x = x.replace(
            "{{NUM_THREADS}}", str(block_dim[0] * block_dim[1] * block_dim[2])
        )
        x = x.replace("{{NUM_BLOCKS}}", str(grid_dim[0] * grid_dim[1] * grid_dim[2]))
        return x

    with ir.Context() as ctx:
        asm_complete, module = compile_mlir_file_to_asm(
            mlir_file,
            kernel_name,
            SYNCHRONOUS_SROA_PASS_PIPELINE,
            ctx,
            library_paths=get_library_paths(),
            print_ir_after_all=False,
            preprocess=preprocess,
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
                input_args=[],
                output_args=[output_data],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=grid_dim,
                block_dim=block_dim,
            )


def analyze_bank_conflicts(banks, title=""):
    """Analyze and print bank conflict information.

    Args:
        banks: Array of shape (64, 2) where banks[tid] = [b0, b1]
               for the banks accessed by each thread's b64 load.
               With 4 bytes per bank, b64 (8 bytes) touches 2 banks.
        title: Description for the output.
    """
    print(f"\n{'='*70}")
    print(f"LDS Bank Analysis: {title}")
    print(f"{'='*70}")

    # Print banks accessed by each thread
    print("\nBanks accessed per thread (tid: [b0, b1]):")
    for tid in range(64):
        b = banks[tid]
        print(f"  lane {tid:2d}: [{b[0]:2d}, {b[1]:2d}]", end="")
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
        # Output: 2 banks per thread (b64 = 8 bytes = 2 x 4-byte banks)
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_lds_banks_A_16x16xf16", output)

        banks = output.reshape(64, 2)
        # fmt: off
        # mfma_index_A_16x16xf16: row = tid % 16, col = 4 * (tid / 16)
        # byte_addr = row * 32 + col * 2
        # bank = (byte_addr / 4) % 32
        expected = np.array([
            [ 0,  1], [ 8,  9], [16, 17], [24, 25],  # tid 0-3
            [ 0,  1], [ 8,  9], [16, 17], [24, 25],  # tid 4-7
            [ 0,  1], [ 8,  9], [16, 17], [24, 25],  # tid 8-11
            [ 0,  1], [ 8,  9], [16, 17], [24, 25],  # tid 12-15
            [ 2,  3], [10, 11], [18, 19], [26, 27],  # tid 16-19
            [ 2,  3], [10, 11], [18, 19], [26, 27],  # tid 20-23
            [ 2,  3], [10, 11], [18, 19], [26, 27],  # tid 24-27
            [ 2,  3], [10, 11], [18, 19], [26, 27],  # tid 28-31
            [ 4,  5], [12, 13], [20, 21], [28, 29],  # tid 32-35
            [ 4,  5], [12, 13], [20, 21], [28, 29],  # tid 36-39
            [ 4,  5], [12, 13], [20, 21], [28, 29],  # tid 40-43
            [ 4,  5], [12, 13], [20, 21], [28, 29],  # tid 44-47
            [ 6,  7], [14, 15], [22, 23], [30, 31],  # tid 48-51
            [ 6,  7], [14, 15], [22, 23], [30, 31],  # tid 52-55
            [ 6,  7], [14, 15], [22, 23], [30, 31],  # tid 56-59
            [ 6,  7], [14, 15], [22, 23], [30, 31],  # tid 60-63
        ], dtype=np.int32)
        # fmt: on
        np.testing.assert_array_equal(banks, expected, "Non-swizzled bank mismatch")

        analyze_bank_conflicts(banks, "Non-swizzled MFMA A 16x16xf16 (b64)")

    def test_lds_banks_swizzled_A_16x16xf16(self):
        """Test banks for swizzled MFMA A matrix pattern."""
        num_threads = 64
        output = np.zeros(num_threads * 2, dtype=np.int32)
        compile_and_run("test_lds_banks_swizzled_A_16x16xf16", output)

        banks = output.reshape(16, 4, 2)

        # Verify swizzled bank computation (32 banks, 4 bytes per bank):
        #   swizzled_col = (col_high XOR row_group) * 4 + col_low
        # where:
        #   row = tid % 16, col = 4 * (tid // 16),
        #   row_group = row // 4,
        #   col_high = col // 4, col_low = col % 4 (always 0 for this pattern)
        # byte_addr = row * 32 + swizzled_col * 2
        # bank = (byte_addr / 4) % 32
        # fmt: off
        expected = np.array([
            [[ 0,  1], [10, 11], [20, 21], [30, 31]],  # tid 0-3
            [[ 8,  9], [18, 19], [28, 29], [ 6,  7]],  # tid 4-7
            [[16, 17], [26, 27], [ 4,  5], [14, 15]],  # tid 8-11
            [[24, 25], [ 2,  3], [12, 13], [22, 23]],  # tid 12-15
            [[ 2,  3], [ 8,  9], [22, 23], [28, 29]],  # tid 16-19
            [[10, 11], [16, 17], [30, 31], [ 4,  5]],  # tid 20-23
            [[18, 19], [24, 25], [ 6,  7], [12, 13]],  # tid 24-27
            [[26, 27], [ 0,  1], [14, 15], [20, 21]],  # tid 28-31
            [[ 4,  5], [14, 15], [16, 17], [26, 27]],  # tid 32-35
            [[12, 13], [22, 23], [24, 25], [ 2,  3]],  # tid 36-39
            [[20, 21], [30, 31], [ 0,  1], [10, 11]],  # tid 40-43
            [[28, 29], [ 6,  7], [ 8,  9], [18, 19]],  # tid 44-47
            [[ 6,  7], [12, 13], [18, 19], [24, 25]],  # tid 48-51
            [[14, 15], [20, 21], [26, 27], [ 0,  1]],  # tid 52-55
            [[22, 23], [28, 29], [ 2,  3], [ 8,  9]],  # tid 56-59
            [[30, 31], [ 4,  5], [10, 11], [16, 17]],  # tid 60-63
        ], dtype=np.int32)
        # fmt: on
        np.testing.assert_array_equal(banks, expected, "Swizzled bank mismatch")

        analyze_bank_conflicts(banks.reshape(64, 2), "Swizzled MFMA A 16x16xf16 (b64)")


if __name__ == "__main__":
    # TestLdsBanks().test_lds_banks_A_16x16xf16()
    TestLdsBanks().test_lds_banks_swizzled_A_16x16xf16()
