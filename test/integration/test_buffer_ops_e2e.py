"""Integration tests for MUBUF buffer operations.

Tests two addressing modes:
- OFFEN (raw mode, stride=0): VGPR provides byte offset, num_records is byte count
- IDXEN (structured mode, stride>0): VGPR provides element index, num_records is
  element count, hardware computes address = base + index * stride + soffset
"""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
SENTINEL = np.int32(0x7EADBEEF)


def _run(mlir_file, kernel_name, input_data, output_data, verify_fn):
    """Wrapper with common test parameters baked in."""
    compile_and_run(
        mlir_file,
        kernel_name,
        input_data=input_data,
        output_data=output_data,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(TOTAL_LANES, 1, 1),
        verify_fn=verify_fn,
        library_paths=[],
        skip_on_cross_compile=True,
    )


def _verify_copy(inputs, outputs):
    """Verify output[0] == input[0] (full copy)."""
    np.testing.assert_array_equal(outputs[0], inputs[0])


def _make_oob_verify(valid, sentinel=SENTINEL):
    """Create verify function for OOB tests: in-bounds match, OOB keeps sentinel."""

    def verify(inputs, outputs):
        result = outputs[0]
        np.testing.assert_array_equal(
            result[:valid], inputs[0][:valid], err_msg="In-bounds elements should match"
        )
        np.testing.assert_array_equal(
            result[valid:],
            np.full(len(result) - valid, sentinel, dtype=np.int32),
            err_msg="OOB elements should keep sentinel",
        )

    return verify


# ---------------------------------------------------------------------------
# OFFEN mode (raw, stride=0)
# ---------------------------------------------------------------------------

OFFEN_MLIR = "buffer-copy-e2e.mlir"
OFFEN_KERNEL = "buffer_copy_kernel"


def _make_offen_params(src_num_bytes, dst_num_bytes, soffset=0):
    """Pack OFFEN params: [src_num_records (bytes), dst_num_records (bytes), soffset]."""
    return np.array([src_num_bytes, dst_num_bytes, soffset], dtype=np.int32)


class TestBufferOffen:
    """Test buffer_load/store_dword with OFFEN addressing (raw mode, stride=0)."""

    def test_offen_copy_all_in_bounds(self):
        """All 64 lanes copy one dword each -- full in-bounds copy."""
        src = np.arange(TOTAL_LANES, dtype=np.int32)
        dst = np.zeros(TOTAL_LANES, dtype=np.int32)
        num_bytes = TOTAL_LANES * 4
        params = _make_offen_params(num_bytes, num_bytes)
        _run(OFFEN_MLIR, OFFEN_KERNEL, [src, params], [dst], _verify_copy)

    def test_offen_oob_non_power_of_2(self):
        """64 lanes, only 37 accessible (non-power-of-2).

        OOB lanes read 0, writes dropped.
        """
        valid = 37
        src = np.arange(TOTAL_LANES, dtype=np.int32)
        dst = np.full(TOTAL_LANES, SENTINEL, dtype=np.int32)
        valid_bytes = valid * 4
        params = _make_offen_params(valid_bytes, valid_bytes)
        _run(OFFEN_MLIR, OFFEN_KERNEL, [src, params], [dst], _make_oob_verify(valid))


# ---------------------------------------------------------------------------
# IDXEN mode (structured, stride>0)
# ---------------------------------------------------------------------------

IDXEN_MLIR = "buffer-idxen-e2e.mlir"


def _make_idxen_params(num_records, soffset=0):
    """Pack IDXEN params: [num_records (element count), soffset]."""
    return np.array([num_records, soffset], dtype=np.int32)


class TestBufferIdxenStride4:
    """Test IDXEN with stride=4 (dword elements, minimal structured mode)."""

    KERNEL = "buffer_idxen_copy_kernel"

    def test_idxen_stride4_copy_all_in_bounds(self):
        """All 64 lanes copy one dword each via index-based addressing."""
        src = np.arange(TOTAL_LANES, dtype=np.int32)
        dst = np.zeros(TOTAL_LANES, dtype=np.int32)
        params = _make_idxen_params(TOTAL_LANES)
        _run(IDXEN_MLIR, self.KERNEL, [src, params], [dst], _verify_copy)

    def test_idxen_stride4_oob_non_power_of_2(self):
        """64 lanes, only 37 accessible (non-power-of-2).

        OOB lanes read 0, writes dropped.
        """
        valid = 37
        src = np.arange(TOTAL_LANES, dtype=np.int32)
        dst = np.full(TOTAL_LANES, SENTINEL, dtype=np.int32)
        params = _make_idxen_params(valid)
        _run(IDXEN_MLIR, self.KERNEL, [src, params], [dst], _make_oob_verify(valid))


class TestBufferIdxenStride1024:
    """Test IDXEN with stride=1024 -- proves hardware stride multiplication works.

    With stride=1024 and buffer_load_dword, each lane reads 4 bytes starting at byte
    offset index*1024. The source buffer has known values scattered at 1024-byte
    intervals; if stride math is wrong the test fails.
    """

    KERNEL = "buffer_idxen_stride1024_kernel"
    STRIDE = 1024
    STRIDE_ELEMENTS = STRIDE // 4  # 256 int32s per stride

    def _make_strided_src(self, num_elements):
        """Create source with known values at stride=1024 intervals.

        Returns (flat_buffer, expected_per_lane_values). Lane i reads the dword at byte
        offset i*1024, i.e. int32 index i*256.
        """
        total_int32s = num_elements * self.STRIDE_ELEMENTS
        buf = np.zeros(total_int32s, dtype=np.int32)
        expected = np.zeros(num_elements, dtype=np.int32)
        for i in range(num_elements):
            val = np.int32(0xA000 + i)
            buf[i * self.STRIDE_ELEMENTS] = val
            expected[i] = val
        return buf, expected

    def _make_strided_dst(self, num_elements, fill=0):
        """Create destination buffer sized for stride=1024."""
        return np.full(num_elements * self.STRIDE_ELEMENTS, fill, dtype=np.int32)

    def _extract_strided(self, flat_buf, num_elements):
        """Extract the dword at each stride=1024 position."""
        return flat_buf[:: self.STRIDE_ELEMENTS][:num_elements]

    def test_idxen_stride1024_all_in_bounds(self):
        """All 64 lanes copy via stride=1024 scattered addressing."""
        src, expected = self._make_strided_src(TOTAL_LANES)
        dst = self._make_strided_dst(TOTAL_LANES)
        params = _make_idxen_params(TOTAL_LANES)

        def verify(inputs, outputs):
            actual = self._extract_strided(outputs[0], TOTAL_LANES)
            np.testing.assert_array_equal(
                actual, expected, err_msg="Strided values must match"
            )

        _run(IDXEN_MLIR, self.KERNEL, [src, params], [dst], verify)

    def test_idxen_stride1024_oob_non_power_of_2(self):
        """64 lanes, only 37 accessible at stride=1024 (non-power-of-2).

        Lanes 0-36 read their strided value. Lanes 37-63 are OOB (index >= 37): reads
        return 0, writes are dropped.
        """
        valid = 37
        src, expected = self._make_strided_src(TOTAL_LANES)
        dst = self._make_strided_dst(TOTAL_LANES, fill=SENTINEL)
        params = _make_idxen_params(valid)

        def verify(inputs, outputs):
            strided = self._extract_strided(outputs[0], TOTAL_LANES)
            np.testing.assert_array_equal(
                strided[:valid], expected[:valid], err_msg="In-bounds strided mismatch"
            )
            np.testing.assert_array_equal(
                strided[valid:],
                np.full(TOTAL_LANES - valid, SENTINEL, dtype=np.int32),
                err_msg="OOB strided positions should keep sentinel",
            )

        _run(IDXEN_MLIR, self.KERNEL, [src, params], [dst], verify)


if __name__ == "__main__":
    TestBufferOffen().test_offen_copy_all_in_bounds()
