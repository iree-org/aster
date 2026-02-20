"""Unit tests for SCF pipeline transformation correctness.

Each test compiles a kernel that contains an scf.for loop with sched.stage
annotations, runs the aster-scf-pipeline pass to pipeline it, then executes
on GPU and verifies the output buffer matches the expected values.

This proves that pipelining preserves the semantic equivalence of the loop:
each original iteration's computation still produces the correct result
despite being spread across prologue, kernel, and epilogue sections.
"""

import numpy as np
import pytest
from aster.pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    test_scf_pipelining_pass_pipeline as _scf_pipeline,
)

from aster.testing import compile_and_run

PIPELINES = [
    pytest.param(TEST_SCF_PIPELINING_PASS_PIPELINE, id="default"),
    pytest.param(_scf_pipeline(gcd_unroll=True), id="gcd-unroll"),
]


class TestScfPipelineTwoStageNoIV:
    """Two-stage pipeline without IV dependence.

    Stage 0: move constant 42 to register (no IV use)
    Stage 1: store to output[0] (no IV use)

    No stage references the induction variable at all. Tests that pipelining
    works correctly when IV adjustment is completely unnecessary.
    Expected output[0] = 42.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_two_stage_no_iv(self, pipeline):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline.mlir",
            "test_two_stage_no_iv",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=pipeline,
            library_paths=[],
        )
        expected = np.array([42], dtype=np.int32)
        np.testing.assert_array_equal(output, expected)


class TestScfPipelineTwoStageIVS0Only:
    """Two-stage pipeline with IV used only in stage 0.

    Stage 0: compute val = i * 4 (uses IV)
    Stage 1: store val to output[val]  (no IV use - val serves as both data and offset)

    Only stage 0 touches the IV, so no IV adjustment is needed for stage 1.
    Expected output[i] = i * 4.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_two_stage_iv_s0_only(self, pipeline):
        num_iters = 8
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline.mlir",
            "test_two_stage_iv_s0_only",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=pipeline,
            library_paths=[],
        )
        expected = np.arange(num_iters, dtype=np.int32) * 4
        np.testing.assert_array_equal(output, expected)


class TestScfPipelineTwoStageIVDep:
    """Two-stage pipeline with IV used in both stages.

    Stage 0: compute val = i * 3 (uses IV)
    Stage 1: compute offset = i * 4, store val (uses IV - tests IV adjustment)

    Expected output[i] = i * 3.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_two_stage_iv_dep(self, pipeline):
        num_iters = 8
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline.mlir",
            "test_two_stage_iv_dep",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=pipeline,
            library_paths=[],
        )
        expected = np.arange(num_iters, dtype=np.int32) * 3
        np.testing.assert_array_equal(output, expected)


class TestScfPipelineFiveStage:
    """Five-stage pipeline (stages 0-4) with linear value chain.

    Stage 0: val = i
    Stage 1: val = val + 1
    Stage 2: val = val + 10
    Stage 3: val = val + 100
    Stage 4: store val to output[i]

    Expected output[i] = i + 111.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_five_stage(self, pipeline):
        num_iters = 10
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline.mlir",
            "test_five_stage",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=pipeline,
            library_paths=[],
        )
        expected = np.arange(num_iters, dtype=np.int32) + 111
        np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    TestScfPipelineTwoStageNoIV().test_two_stage_no_iv(
        TEST_SCF_PIPELINING_PASS_PIPELINE
    )
    TestScfPipelineTwoStageIVS0Only().test_two_stage_iv_s0_only(
        TEST_SCF_PIPELINING_PASS_PIPELINE
    )
    TestScfPipelineTwoStageIVDep().test_two_stage_iv_dep(
        TEST_SCF_PIPELINING_PASS_PIPELINE
    )
    TestScfPipelineFiveStage().test_five_stage(TEST_SCF_PIPELINING_PASS_PIPELINE)
