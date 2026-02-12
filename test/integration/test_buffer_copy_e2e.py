"""Integration test for buffer copy kernel with OOB bounds checking.

The kernel uses buffer descriptors (make_buffer_rsrc) with stride=0 (raw mode).
num_records is a byte count; the hardware checks voffset + soffset < num_records.
"""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
KERNEL_NAME = "buffer_copy_kernel"


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

        compile_and_run(
            "buffer-copy-e2e.mlir",
            KERNEL_NAME,
            input_data=[src, params],
            output_data=[dst],
            pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            block_dim=(num_elements, 1, 1),
            verify_fn=verify,
            library_paths=[],
            skip_on_cross_compile=True,
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

        compile_and_run(
            "buffer-copy-e2e.mlir",
            KERNEL_NAME,
            input_data=[src, params],
            output_data=[dst],
            pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            block_dim=(total_lanes, 1, 1),
            verify_fn=verify,
            library_paths=[],
            skip_on_cross_compile=True,
        )


if __name__ == "__main__":
    TestBufferCopy().test_copy_all_in_bounds()
