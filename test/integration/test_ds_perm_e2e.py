"""E2E test: ds_permute_b32 and ds_bpermute_b32 cross-lane permutation.

ds_permute_b32:  VDST[(ADDR[i] + offset)/4 % 64] = DATA0[i]  (forward/scatter)
ds_bpermute_b32: VDST[i] = DATA0[(ADDR[i] + offset)/4 % 64]  (backward/gather)

Tests:
  1. permute_rotate: forward rotate shows permute puts data where addr says.
     output[j] = (j+63)%64.
  2. permute_bpermute_idempotent: bpermute(permute(data, addr), addr) = data.
     output[i] = i.
"""

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def test_permute_rotate():
    """Forward rotate by 1: lane i sends data to lane (i+1)%64."""
    output = np.zeros(WAVEFRONT_SIZE, dtype=np.int32)

    def verify(inputs, outputs):
        expected = np.array(
            [(i + 63) % WAVEFRONT_SIZE for i in range(WAVEFRONT_SIZE)],
            dtype=np.int32,
        )
        np.testing.assert_array_equal(
            outputs[0],
            expected,
            err_msg="Forward rotate: output[j] should equal (j+63)%64",
        )

    compile_and_run(
        "ds-perm-e2e.mlir",
        "permute_rotate",
        input_data=[],
        output_data=[output],
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(WAVEFRONT_SIZE, 1, 1),
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


def test_permute_bpermute_idempotent():
    """Idempotency: bpermute(permute(data, addr), addr) = data."""
    output = np.zeros(WAVEFRONT_SIZE, dtype=np.int32)

    def verify(inputs, outputs):
        expected = np.arange(WAVEFRONT_SIZE, dtype=np.int32)
        np.testing.assert_array_equal(
            outputs[0],
            expected,
            err_msg="Idempotency: bpermute(permute(data)) should equal data",
        )

    compile_and_run(
        "ds-perm-e2e.mlir",
        "permute_bpermute_idempotent",
        input_data=[],
        output_data=[output],
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(WAVEFRONT_SIZE, 1, 1),
        grid_dim=(1, 1, 1),
        verify_fn=verify,
        library_paths=[],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
