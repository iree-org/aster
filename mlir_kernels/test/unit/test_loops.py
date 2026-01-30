"""Unit tests for loop execution."""

import numpy as np

try:
    from .test_utils import compile_and_run
except ImportError:
    from test_utils import compile_and_run


# Loop pass pipeline - different from the standard SYNCHRONOUS_SROA_PASS_PIPELINE
LOOP_PASS_PIPELINE = (
    "builtin.module("
    "  aster-optimize-arith,"
    "  func.func(aster-amdgcn-set-abi),"
    # Convert SCF control flow to AMDGCN control flow
    "  amdgcn-convert-scf-control-flow,"
    "  canonicalize,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    "  canonicalize,cse,"
    "  aster-to-amdgcn,"
    "  amdgcn-convert-waits,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-hoist-ops"
    "    )"
    "  ),"
    "  amdgcn-register-allocation,"
    "  canonicalize,cse"
    ")"
)


class TestUniformLoopLowering:
    """Test uniform loop lowering."""

    def test_uniform_loop(self):
        """The output buffer should contain [0, ..., n - 1] * 4 after execution."""
        num_threads = 64
        output = np.zeros(64, dtype=np.int32)
        compile_and_run(
            "test_loops.mlir",
            "test_uniform_loop",
            # TODO: properly handle int arguments
            input_data=[np.array([output.size], dtype=np.int32)],
            output_data=output,
            block_dim=(num_threads, 1, 1),
            pass_pipeline=LOOP_PASS_PIPELINE,
            library_paths=[],  # No library needed for this test
        )
        expected = np.arange(num_threads, dtype=np.int32)
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.testing.assert_array_equal(output, expected * 4)
