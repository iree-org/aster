#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""E2E test: KernelBuilder tiledmma kernel vs numpy reference.

Constructs a buffer_load + v_mfma_f32_16x16x16_f16 + buffer_store kernel
using KernelBuilder (pure Python, no MLIR text), compiles through the ASTER
pipeline, assembles to HSACO, and executes on GPU. Result is compared against
numpy matmul.

Skipped if gfx942 assembler is unavailable or no gfx942 GPU is present.
"""

import pathlib
import sys

import numpy as np
import pytest

# Allow importing the shared build helper from python/test/test_kernel_builder.py.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from test_kernel_builder import build_tiledmma_module, compile_to_asm

from aster import ir, utils
from aster.testing import execute_kernel_and_verify, hsaco_file


@pytest.mark.parametrize(
    "mcpu",
    [
        "gfx942",
    ],
)
@pytest.mark.parametrize(
    "wavefront_size",
    [
        64,
    ],
)
def test_tiledmma_executes_correctly(mcpu, wavefront_size):
    """KernelBuilder tiledmma kernel result matches np.matmul."""
    n_threads = wavefront_size  # one wavefront

    # n_threads x 4 f16 for A and B (viewed as 16x16 f16 matrix)
    A = (np.random.randn(n_threads * 4) * 0.1).astype(np.float16)
    B = (np.random.randn(n_threads * 4) * 0.1).astype(np.float16)
    C = np.zeros(n_threads * 4, dtype=np.float32)

    ctx = ir.Context()
    ctx.allow_unregistered_dialects = True
    with ctx:
        module = build_tiledmma_module(target=mcpu)
        asm = compile_to_asm(module)

    path = utils.assemble_to_hsaco(asm, target=mcpu, wavefront_size=wavefront_size)
    if path is None:
        pytest.skip(f"LLVM assembler does not support {mcpu}")

    with hsaco_file(path):
        if not utils.system_has_mcpu(mcpu=mcpu):
            pytest.skip(f"{mcpu} GPU not available")

        execute_kernel_and_verify(
            hsaco_path=path,
            kernel_name="tiledmma",
            input_args=[A, B],
            output_args=[C],
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            grid_dim=(1, 1, 1),
            block_dim=(n_threads, 1, 1),
        )

    # C.reshape(n_threads, 4): each thread's 4 accumulators
    # Reference: A.reshape(16, 16) @ B.reshape(16, 16) -> 16x16 f32 matrix
    A_mat = A.reshape(16, 16).astype(np.float32)
    B_mat = B.reshape(16, 16).astype(np.float32)
    expected = (A_mat @ B_mat).flatten()
    np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
