"""Unit tests for SCF pipeline with LDS double-buffering."""

import numpy as np
from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from aster.testing import compile_and_run


class TestLdsPipelinePassthrough:
    """Two-stage LDS pass-through (no IV dependence in LDS path).

    Stage 0: alloc LDS, write constant 42 at offset 0
    Stage 1: wait, read back from LDS, store to output[0]

    4 iterations, each writing 42. After double-buffered pipelining,
    rotating LDS offsets ensure no corruption between overlapping iterations.
    Expected: output[0] = 42
    """

    def test_lds_passthrough(self):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline_lds.mlir",
            "test_lds_passthrough",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            library_paths=[],
        )
        expected = np.array([42], dtype=np.int32)
        np.testing.assert_array_equal(output, expected)


class TestLdsPipelineIVDep:
    """Two-stage LDS with IV-dependent data.

    Stage 0: compute val = i * 7, write to LDS
    Stage 1: wait, read from LDS, store to output[i*4]

    8 iterations. After double-buffered pipelining, each output slot
    must contain its iteration's value despite overlapping stages.
    Expected: output[i] = i * 7 for i in [0..8)
    """

    def test_lds_iv_dep(self):
        num_iters = 8
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline_lds.mlir",
            "test_lds_iv_dep",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            library_paths=[],
        )
        expected = np.arange(num_iters, dtype=np.int32) * 7
        np.testing.assert_array_equal(output, expected)


class TestLdsPipelineAccum:
    """Two-stage LDS with accumulator iter_arg.

    Stage 0: write constant 3 to LDS
    Stage 1: wait, read from LDS, add to vgpr accumulator (iter_arg)

    6 iterations, accumulating 3 each time. Final = 3 * 6 = 18.
    Tests interaction of LDS double-buffering with iter_args.
    Expected: output[0] = 18
    """

    def test_lds_accum(self):
        output = np.zeros(1, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline_lds.mlir",
            "test_lds_accum",
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            library_paths=[],
        )
        assert output[0] == 18, f"expected 18, got {output[0]}"


class TestLdsPipelineSixStage:
    """Six-stage double-LDS hop: global_load -> ds_write_A -> ds_read_A -> ds_write_B -> ds_read_B -> global_store.

    Two alloc_lds each spanning all 6 stages produce 6 buffers each (12 total)
    after multibuffer prep. 10 iterations, input[i] passed through two LDS hops.
    Expected: output[i] = input[i] for i in [0..10)

    KNOWN BUG: pipeliner crashes on cross-stage value gaps > 1 (Bug 8).
    Mark xfail until the pipeliner is fixed.
    """

    import pytest

    def test_lds_six_stage(self):
        num_iters = 10
        input_data = np.arange(num_iters, dtype=np.int32) * 11
        output = np.zeros(num_iters, dtype=np.int32)
        compile_and_run(
            "test_scf_pipeline_lds.mlir",
            "test_lds_six_stage",
            input_data=input_data,
            output_data=output,
            block_dim=(64, 1, 1),
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            library_paths=[],
        )
        np.testing.assert_array_equal(output, input_data)


if __name__ == "__main__":
    TestLdsPipelinePassthrough().test_lds_passthrough()
    TestLdsPipelineIVDep().test_lds_iv_dep()
    TestLdsPipelineAccum().test_lds_accum()
    TestLdsPipelineSixStage().test_lds_six_stage()
