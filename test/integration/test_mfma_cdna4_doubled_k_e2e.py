# Copyright 2026 The ASTER Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration tests for CDNA4 doubled-K MFMA instructions on gfx950.

Three kernels test distinct instructions:

Kernel 1 (mfma_16x16x32_f16_ones):
  A = f16 1.0 (packed 0x3C003C00), B = f16 1.0, C = 0
  Expected: sum_{k=0}^{31} 1.0 * 1.0 = 32.0

Kernel 2 (mfma_16x16x32_bf16_ones):
  A = bf16 1.0 (packed 0x3F803F80), B = bf16 1.0, C = 0
  Expected: sum_{k=0}^{31} 1.0 * 1.0 = 32.0

Kernel 3 (mfma_32x32x16_bf16_ones):
  A = bf16 1.0 (packed 0x3F803F80), B = bf16 1.0, C = 0
  Expected: sum_{k=0}^{15} 1.0 * 1.0 = 16.0

Kernel 4 (mfma_32x32x16_f16_ones):
  A = f16 1.0 (packed 0x3C003C00), B = f16 1.0, C = 0
  Expected: sum_{k=0}^{15} 1.0 * 1.0 = 16.0
"""

import os

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run
from aster.test_pass_pipelines import TEST_SROA_PASS_PIPELINE

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_LIBRARY_DIR = os.path.join(_THIS_DIR, "..", "..", "mlir_kernels", "library", "common")
_REGISTER_INIT = os.path.join(_LIBRARY_DIR, "register-init.mlir")

MCPU = "gfx950"
WAVEFRONT_SIZE = 64
MLIR_FILE = "mfma-cdna4-doubled-k-e2e.mlir"


class TestMfmaCdna4DoubledK:
    """Test CDNA4 doubled-K MFMA instructions end-to-end on gfx950."""

    def test_16x16x32_f16_ones(self):
        """v_mfma_f32_16x16x32_f16: A=B=f16 1.0, C=0 -> D=32.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 32.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"16x16x32 f16 MFMA failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_16x16x32_f16_ones",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=TEST_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
        )

    def test_16x16x32_bf16_ones(self):
        """v_mfma_f32_16x16x32_bf16: A=B=bf16 1.0, C=0 -> D=32.0."""
        m, n = 16, 16
        c_data = np.zeros(m * n, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 32.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"16x16x32 bf16 MFMA failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_16x16x32_bf16_ones",
            output_data=[c_data],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=TEST_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
        )

    def test_32x32x16_bf16_ones(self):
        """v_mfma_f32_32x32x16_bf16: A=B=bf16 1.0, C=0 -> D=16.0."""
        # 64 lanes * 16 dwords each = 1024 f32 values
        total_elements = WAVEFRONT_SIZE * 16
        output = np.zeros(total_elements, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 16.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"32x32x16 bf16 MFMA failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_32x32x16_bf16_ones",
            output_data=[output],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=TEST_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
        )

    def test_32x32x16_f16_ones(self):
        """v_mfma_f32_32x32x16_f16: A=B=f16 1.0, C=0 -> D=16.0."""
        # 64 lanes * 16 dwords each = 1024 f32 values
        total_elements = WAVEFRONT_SIZE * 16
        output = np.zeros(total_elements, dtype=np.float32)

        def verify(inputs, outputs):
            result = np.array(outputs[0])
            expected = 16.0
            np.testing.assert_allclose(
                result,
                expected,
                err_msg=(
                    f"32x32x16 f16 MFMA failed: "
                    f"expected {expected}, "
                    f"got min={result.min()}, max={result.max()}, mean={result.mean()}"
                ),
            )

        compile_and_run(
            MLIR_FILE,
            "mfma_32x32x16_f16_ones",
            output_data=[output],
            mcpu=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            pass_pipeline=TEST_SROA_PASS_PIPELINE,
            verify_fn=verify,
            library_paths=[_REGISTER_INIT],
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
