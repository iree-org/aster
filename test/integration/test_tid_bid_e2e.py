"""E2E test: verify packed workitem ID extraction and block ID handling.

On CDNA3/CDNA4 (gfx942/gfx950), workitem IDs are packed in VGPR0 as
{Z[9:0], Y[9:0], X[9:0]}. This test verifies that thread_id x/y/z and
block_id x/y/z are correctly extracted from the packed register and
stored to global memory.

Test values:
  thread_id: x=42 (1D/2D tests), x=10,y=5,z=7 (3D, max WG size=1024)
  block_id:  x=42,y=5,z=7
"""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def test_tid_x_only():
    """thread_id x only -- no masking needed (X is lowest 10 bits)."""
    block = (64, 1, 1)
    n_threads = 64
    output = np.zeros(n_threads, dtype=np.int32)

    def verify(inputs, outputs):
        np.testing.assert_array_equal(
            outputs[0][42],
            42,
            err_msg="thread_id x should be 42 for lane 42",
        )
        # Also check a few other lanes for sanity
        for i in [0, 1, 31, 63]:
            np.testing.assert_array_equal(
                outputs[0][i],
                i,
                err_msg=f"thread_id x should be {i} for lane {i}",
            )

    compile_and_run(
        "tid-bid-e2e.mlir",
        "tid_x_only",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


def test_tid_xy():
    """thread_id x + y -- verifies packed extraction with masking."""
    block = (64, 8, 1)
    n_threads = 64 * 8  # 512
    # 2 dwords per thread: [tid_x, tid_y]
    output = np.zeros(n_threads * 2, dtype=np.int32)

    TARGET_X, TARGET_Y = 42, 5

    def verify(inputs, outputs):
        buf = outputs[0]
        linear = TARGET_Y * 64 + TARGET_X  # = 362
        np.testing.assert_array_equal(
            buf[linear * 2],
            TARGET_X,
            err_msg=f"thread_id x should be {TARGET_X}",
        )
        np.testing.assert_array_equal(
            buf[linear * 2 + 1],
            TARGET_Y,
            err_msg=f"thread_id y should be {TARGET_Y}",
        )
        # Cross-check another thread
        lx, ly = 0, 0
        lin0 = ly * 64 + lx
        np.testing.assert_array_equal(buf[lin0 * 2], 0)
        np.testing.assert_array_equal(buf[lin0 * 2 + 1], 0)

    compile_and_run(
        "tid-bid-e2e.mlir",
        "tid_xy",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


def test_tid_xyz():
    """thread_id x + y + z -- full packed extraction from VGPR0.

    Uses block=(16,8,8)=1024 threads (max WG size on gfx942). Tests x=10, y=5, z=7 (42
    doesn't fit with y=8,z=8 in 1024 threads).
    """
    block = (16, 8, 8)
    n_threads = 16 * 8 * 8  # 1024
    # 3 dwords per thread: [tid_x, tid_y, tid_z]
    output = np.zeros(n_threads * 3, dtype=np.int32)

    TARGET_X, TARGET_Y, TARGET_Z = 10, 5, 7

    def verify(inputs, outputs):
        buf = outputs[0]
        # linear = (z << 7) | (y << 4) | x = z*128 + y*16 + x
        linear = TARGET_Z * 128 + TARGET_Y * 16 + TARGET_X  # = 986
        np.testing.assert_array_equal(
            buf[linear * 3],
            TARGET_X,
            err_msg=f"thread_id x should be {TARGET_X}",
        )
        np.testing.assert_array_equal(
            buf[linear * 3 + 1],
            TARGET_Y,
            err_msg=f"thread_id y should be {TARGET_Y}",
        )
        np.testing.assert_array_equal(
            buf[linear * 3 + 2],
            TARGET_Z,
            err_msg=f"thread_id z should be {TARGET_Z}",
        )
        # Cross-check thread (0,0,0)
        np.testing.assert_array_equal(buf[0], 0)
        np.testing.assert_array_equal(buf[1], 0)
        np.testing.assert_array_equal(buf[2], 0)

    compile_and_run(
        "tid-bid-e2e.mlir",
        "tid_xyz",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


def test_bid_xyz():
    """block_id x + y + z -- SGPR system registers.

    Launches grid=(43,6,8)=2064 workgroups, block=(64,1,1). Verifies block_id=(42,5,7)
    is correctly reported.
    """
    grid = (43, 6, 8)
    block = (64, 1, 1)
    total_wgs = 43 * 6 * 8  # 2064
    # 3 dwords per WG: [bid_x, bid_y, bid_z]
    output = np.zeros(total_wgs * 3, dtype=np.int32)

    TARGET_BX, TARGET_BY, TARGET_BZ = 42, 5, 7

    def verify(inputs, outputs):
        buf = outputs[0]
        # linear_bid = bid_x + 43 * (bid_y + 6 * bid_z)
        linear_bid = TARGET_BX + 43 * (TARGET_BY + 6 * TARGET_BZ)  # = 2063
        np.testing.assert_array_equal(
            buf[linear_bid * 3],
            TARGET_BX,
            err_msg=f"block_id x should be {TARGET_BX}",
        )
        np.testing.assert_array_equal(
            buf[linear_bid * 3 + 1],
            TARGET_BY,
            err_msg=f"block_id y should be {TARGET_BY}",
        )
        np.testing.assert_array_equal(
            buf[linear_bid * 3 + 2],
            TARGET_BZ,
            err_msg=f"block_id z should be {TARGET_BZ}",
        )
        # Cross-check block (0,0,0)
        np.testing.assert_array_equal(buf[0], 0)
        np.testing.assert_array_equal(buf[1], 0)
        np.testing.assert_array_equal(buf[2], 0)

    compile_and_run(
        "tid-bid-e2e.mlir",
        "bid_xyz",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=grid,
        verify_fn=verify,
        library_paths=[],
    )


def test_tid_x_bid_x():
    """Combined thread_id x + block_id x.

    Launches grid=(43,1,1), block=(64,1,1). Verifies tid_x=42 and bid_x=42 in the same
    kernel.
    """
    grid = (43, 1, 1)
    block = (64, 1, 1)
    total_threads = 43 * 64  # 2752
    # 2 dwords per thread: [tid_x, bid_x]
    output = np.zeros(total_threads * 2, dtype=np.int32)

    TARGET_TX, TARGET_BX = 42, 42

    def verify(inputs, outputs):
        buf = outputs[0]
        # linear = (bid_x << 6) | tid_x = bid_x * 64 + tid_x
        linear = TARGET_BX * 64 + TARGET_TX  # = 2730
        np.testing.assert_array_equal(
            buf[linear * 2],
            TARGET_TX,
            err_msg=f"thread_id x should be {TARGET_TX}",
        )
        np.testing.assert_array_equal(
            buf[linear * 2 + 1],
            TARGET_BX,
            err_msg=f"block_id x should be {TARGET_BX}",
        )
        # Cross-check thread 0 in block 0
        np.testing.assert_array_equal(buf[0], 0)
        np.testing.assert_array_equal(buf[1], 0)

    compile_and_run(
        "tid-bid-e2e.mlir",
        "tid_x_bid_x",
        input_data=[],
        output_data=[output],
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=block,
        grid_dim=grid,
        verify_fn=verify,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
