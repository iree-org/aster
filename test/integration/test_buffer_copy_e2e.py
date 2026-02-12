"""Integration test for buffer copy kernel with OOB bounds checking.

The kernel uses buffer descriptors (make_buffer_rsrc) with stride=0 (raw mode).
num_records is a byte count; the hardware checks voffset + soffset < num_records.
"""

import os
import pytest
import numpy as np

from aster import ir, utils
from typing import List, Optional
from aster.testing import (
    execute_kernel_and_verify,
    compile_mlir_file_to_asm,
    hsaco_file,
)
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
KERNEL_NAME = "buffer_copy_kernel"


def _get_mlir_file():
    return os.path.join(os.path.dirname(__file__), "buffer-copy-e2e.mlir")


def _compile_kernel():
    """Compile the buffer copy kernel, return (hsaco_path, asm)."""
    with ir.Context() as ctx:
        asm, _ = compile_mlir_file_to_asm(
            _get_mlir_file(), KERNEL_NAME, DEFAULT_SROA_PASS_PIPELINE, ctx
        )
    hsaco_path = utils.assemble_to_hsaco(
        asm, target=MCPU, wavefront_size=WAVEFRONT_SIZE
    )
    assert hsaco_path is not None, "Failed to assemble kernel to HSACO"
    return hsaco_path, asm


def _make_params(src_num_bytes, dst_num_bytes, soffset=0):
    """Pack buffer descriptor parameters into a numpy array.

    The kernel's params buffer layout is:     [0]: src_num_records (i32) -- byte count
    for raw mode (stride=0)     [1]: dst_num_records (i32) -- byte count for raw mode
    (stride=0)     [2]: soffset         (i32)
    """
    return np.array([src_num_bytes, dst_num_bytes, soffset], dtype=np.int32)


class TestBufferCopy:
    """Test buffer_load/store_dword via buffer descriptors."""

    def test_copy_all_in_bounds(self):
        """All 64 lanes copy one dword each -- full in-bounds copy."""
        num_elements = 64
        src = np.arange(num_elements, dtype=np.int32)
        dst = np.zeros(num_elements, dtype=np.int32)
        num_bytes = num_elements * 4
        params = _make_params(num_bytes, num_bytes)

        def verify(inputs, outputs):
            np.testing.assert_array_equal(outputs[0], inputs[0])

        hsaco_path, _ = _compile_kernel()
        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=MCPU):
                pytest.skip(f"GPU {MCPU} not available")

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=KERNEL_NAME,
                input_args=[src, params],
                output_args=[dst],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=(1, 1, 1),
                block_dim=(num_elements, 1, 1),
                verify_fn=verify,
            )

    def test_oob_bounds_checking(self):
        """64 lanes launched, only 32 elements accessible.

        Lanes 0-31 copy normally. Lanes 32-63 read 0 (OOB read) and their writes are
        silently dropped, leaving dst sentinel intact.
        """
        total_lanes = 64
        valid = 32
        sentinel = np.int32(0x7EADBEEF)

        src = np.arange(total_lanes, dtype=np.int32)
        dst = np.full(total_lanes, sentinel, dtype=np.int32)
        valid_bytes = valid * 4
        params = _make_params(valid_bytes, valid_bytes)

        def verify(inputs, outputs):
            result = outputs[0]
            np.testing.assert_array_equal(
                result[:valid],
                inputs[0][:valid],
                err_msg="In-bounds elements should match",
            )
            np.testing.assert_array_equal(
                result[valid:],
                np.full(total_lanes - valid, sentinel, dtype=np.int32),
                err_msg="OOB elements should keep sentinel",
            )

        hsaco_path, _ = _compile_kernel()
        with hsaco_file(hsaco_path):
            if not utils.system_has_mcpu(mcpu=MCPU):
                pytest.skip(f"GPU {MCPU} not available")

            execute_kernel_and_verify(
                hsaco_path=hsaco_path,
                kernel_name=KERNEL_NAME,
                input_args=[src, params],
                output_args=[dst],
                mcpu=MCPU,
                wavefront_size=WAVEFRONT_SIZE,
                grid_dim=(1, 1, 1),
                block_dim=(total_lanes, 1, 1),
                verify_fn=verify,
            )


if __name__ == "__main__":
    TestBufferCopy().test_copy_all_in_bounds()
