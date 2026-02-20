"""E2E tests for SCF pipeline with stage gaps (non-consecutive stage numbers).

Tests that pipelining with gaps produces correct results by compiling kernels with non-
consecutive sched.stage attributes, running on GPU, and verifying output buffers match
expected values.
"""

import numpy as np
import pytest
from aster.pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    test_scf_pipelining_pass_pipeline as _scf_pipeline,
)

from aster.testing import compile_and_run

MLIR_FILE = "test_scf_pipeline_gaps.mlir"
BLOCK = (64, 1, 1)
NO_LIBS = []

PIPELINES = [
    pytest.param(TEST_SCF_PIPELINING_PASS_PIPELINE, id="default"),
    pytest.param(_scf_pipeline(gcd_unroll=True), id="gcd-unroll"),
]


class TestGap02:
    """Stages {0, 2} with IV in both stages.

    Stage 0: val = i * 3, Stage 2: store val to output[i].
    Expected: output[i] = i * 3.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_gap_0_2(self, pipeline):
        num_iters = 8
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_gap_0_2",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        expected = np.arange(num_iters, dtype=np.int32) * 3
        np.testing.assert_array_equal(output, expected)


class TestGap03:
    """Stages {0, 3} with wide gap.

    Stage 0: val = i * 5, Stage 3: store val to output[i].
    Shift register depth 3.
    Expected: output[i] = i * 5.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_gap_0_3(self, pipeline):
        num_iters = 8
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_gap_0_3",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        expected = np.arange(num_iters, dtype=np.int32) * 5
        np.testing.assert_array_equal(output, expected)


class TestGap025:
    """Stages {0, 2, 5} with two cascaded gaps.

    Stage 0: val = i, Stage 2: val += 10, Stage 5: store to output[i].
    Expected: output[i] = i + 10.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_gap_0_2_5(self, pipeline):
        num_iters = 10
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_gap_0_2_5",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        expected = np.arange(num_iters, dtype=np.int32) + 10
        np.testing.assert_array_equal(output, expected)


class TestGap02IterArgs:
    """Stages {0, 2} with scalar iter_arg accumulator.

    Stage 0: independent work, Stage 2: acc += 7.
    6 iterations, init 0 -> final 42.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_gap_0_2_iter_args(self, pipeline):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_gap_0_2_iter_args",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        assert output[0] == 42, f"expected 42, got {output[0]}"


if __name__ == "__main__":
    TestGap02().test_gap_0_2(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestGap03().test_gap_0_3(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestGap025().test_gap_0_2_5(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestGap02IterArgs().test_gap_0_2_iter_args(TEST_SCF_PIPELINING_PASS_PIPELINE)
