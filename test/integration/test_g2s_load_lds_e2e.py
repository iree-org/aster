"""Integration test for G2S (Global-to-LDS) buffer_load_dword with LDS flag.

Each lane loads one dword from global memory directly into LDS via buffer_load_dword_lds
(G2S DMA path). The data is then read back from LDS via ds_read_b32 and stored to the
output buffer.

With M0=44 (dword-aligned) and no inst_offset, the hardware computes:   LDS_ADDR = 44 +
ThreadID * 4 so lane i's dword lands at LDS byte offset 44 + i*4. M0 must be dword-
aligned (hardware masks low 2 bits: M0[17:2]*4).
"""

import os

import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_DIR = os.path.join(_THIS_DIR, "..", "..", "mlir_kernels", "library", "common")
_REGISTER_INIT = os.path.join(_LIBRARY_DIR, "register-init.mlir")

MCPU = "gfx950"
WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "g2s-load-lds-e2e.mlir"
KERNEL_NAME = "g2s_roundtrip_kernel"


def _make_params(num_bytes, soffset=0):
    """Pack params: [num_bytes, soffset]."""
    return np.array([num_bytes, soffset], dtype=np.int32)


def test_g2s_roundtrip():
    """Load 64 dwords via G2S, read back from LDS, verify output matches input."""
    src = np.arange(TOTAL_LANES, dtype=np.int32)
    dst = np.zeros(TOTAL_LANES, dtype=np.int32)
    num_bytes = TOTAL_LANES * 4
    params = _make_params(num_bytes, soffset=0)

    def verify(inputs, outputs):
        np.testing.assert_array_equal(
            outputs[0],
            inputs[0],
            err_msg="G2S roundtrip failed: output does not match input",
        )

    compile_and_run(
        MLIR_FILE,
        KERNEL_NAME,
        input_data=[src, params],
        output_data=[dst],
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(TOTAL_LANES, 1, 1),
        verify_fn=verify,
        library_paths=[_REGISTER_INIT],
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
