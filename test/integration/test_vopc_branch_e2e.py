"""E2E tests for the VOPC comparison + per-lane select path.

Tests lsir.cmpi + lsir.select with VGPR operands, which LegalizeCF lowers
to v_cmp_* (setting VCC) + v_cndmask_b32 (per-lane select based on VCC bits).

Three scenarios:
- always_true:  tid < 100, all 64 lanes true  -> all output 42.
- always_false: tid < 0, no lane true          -> all output 99.
- per_lane:     tid < 32, lanes 0-31 true      -> lanes 0-31 output 42,
                                                  lanes 32-63 output 99.
  The per_lane test verifies true per-lane VCC semantics: each lane
  independently selects based on its own comparison result.
"""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "vopc-branch-e2e.mlir"

TRUE_VAL = np.int32(42)
FALSE_VAL = np.int32(99)


def _run(kernel_name, output_data, verify_fn):
    compile_and_run(
        MLIR_FILE,
        kernel_name,
        input_data=[],
        output_data=output_data,
        pass_pipeline=TEST_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(TOTAL_LANES, 1, 1),
        verify_fn=verify_fn,
        library_paths=[],
    )


class TestVopcSelect:
    """VOPC compare + per-lane select (lsir.cmpi VGPR -> v_cmp_* + v_cndmask_b32)."""

    def test_select_always_true(self):
        """Tid < 100: all 64 lanes true -> all select 42."""
        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            np.testing.assert_array_equal(
                outputs[0],
                np.full(TOTAL_LANES, TRUE_VAL, dtype=np.int32),
                err_msg="All lanes should select true value (42)",
            )

        _run("vopc_select_always_true_kernel", [dst], verify)

    def test_select_always_false(self):
        """Tid < 0: no lane true -> all select 99."""
        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            np.testing.assert_array_equal(
                outputs[0],
                np.full(TOTAL_LANES, FALSE_VAL, dtype=np.int32),
                err_msg="All lanes should select false value (99)",
            )

        _run("vopc_select_always_false_kernel", [dst], verify)

    def test_select_per_lane(self):
        """Tid < 32: lanes 0-31 select 42, lanes 32-63 select 99.

        Verifies per-lane VCC semantics: each lane independently picks
        true_val or false_val based on its own v_cmp result bit in VCC.
        """
        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            expected = np.full(TOTAL_LANES, FALSE_VAL, dtype=np.int32)
            expected[:32] = TRUE_VAL
            np.testing.assert_array_equal(
                outputs[0],
                expected,
                err_msg="Lanes 0-31 should be 42, lanes 32-63 should be 99",
            )

        _run("vopc_select_per_lane_kernel", [dst], verify)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
