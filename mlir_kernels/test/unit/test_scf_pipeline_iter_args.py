"""E2E tests for SCF pipeline with iter_args.

Four focused tests covering the cross-product:
  {scalar, vgpr} iter_arg  x  {no IV, with IV}

VGPR iter_args exercise the bufferization pass (aster-amdgcn-bufferization) since
the loop-carried register value must be properly handled across
prologue/kernel/epilogue boundaries.
"""

import numpy as np
import pytest
from aster.pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    test_scf_pipelining_pass_pipeline as _scf_pipeline,
)

from aster.testing import compile_and_run

MLIR_FILE = "test_scf_pipeline_iter_args.mlir"
BLOCK = (64, 1, 1)
NO_LIBS = []

PIPELINES = [
    pytest.param(TEST_SCF_PIPELINING_PASS_PIPELINE, id="default"),
    pytest.param(_scf_pipeline(gcd_unroll=True), id="gcd-unroll"),
]


class TestIterArgsScalarNoIV:
    """2-stage, scalar iter_arg, no induction variable.

    acc += 7 each iteration, 6 iters -> 42.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_scalar_no_iv(self, pipeline):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_iter_args_scalar_no_iv",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        assert output[0] == 42, f"expected 42, got {output[0]}"


class TestIterArgsScalarWithIV:
    """2-stage, scalar iter_arg, with induction variable.

    acc += i each iteration, 8 iters -> sum(0..7) = 28.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_scalar_with_iv(self, pipeline):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_iter_args_scalar_with_iv",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        assert output[0] == 28, f"expected 28, got {output[0]}"


class TestIterArgsVgprNoIV:
    """2-stage, VGPR iter_arg (bufferization), no induction variable.

    vgpr acc += 5 each iteration, 4 iters -> 20.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_vgpr_no_iv(self, pipeline):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_iter_args_vgpr_no_iv",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        assert output[0] == 20, f"expected 20, got {output[0]}"


class TestIterArgsVgprWithIV:
    """2-stage, VGPR iter_arg (bufferization), with induction variable.

    vgpr acc += i each iteration, 8 iters -> sum(0..7) = 28.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_vgpr_with_iv(self, pipeline):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            MLIR_FILE,
            "test_iter_args_vgpr_with_iv",
            output_data=output,
            block_dim=BLOCK,
            pass_pipeline=pipeline,
            library_paths=NO_LIBS,
        )
        assert output[0] == 28, f"expected 28, got {output[0]}"


class TestScfPipelineIterArgs:
    """Six-stage pipeline (stages 0-5) with iter_arg accumulator.

    Stages 0-4: independent work (exercises deep prologue/epilogue).
    Stage 5: accumulate constant 5 into scalar iter_arg.
    12 iterations, so final sum = 5 * 12 = 60.
    Stored at byte offset 60 = int32 index 15.
    """

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_iter_args(self, pipeline):
        output = np.zeros(16, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline_iter_args.mlir",
            "test_iter_args",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=pipeline,
            library_paths=[],
        )
        assert output[15] == 60, f"expected 60, got {output[15]}"


if __name__ == "__main__":
    TestIterArgsScalarNoIV().test_scalar_no_iv(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestIterArgsScalarWithIV().test_scalar_with_iv(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestIterArgsVgprNoIV().test_vgpr_no_iv(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestIterArgsVgprWithIV().test_vgpr_with_iv(TEST_SCF_PIPELINING_PASS_PIPELINE)
    TestScfPipelineIterArgs().test_iter_args(TEST_SCF_PIPELINING_PASS_PIPELINE)
