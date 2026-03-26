"""Test and benchmarking pass pipelines.

These pipelines are used for specific bringup scenarios, debugging, and
benchmarking. The production pipeline lives in pass_pipelines.py as
DEFAULT_PASS_PIPELINE.

Naming convention:
  Python constants: TEST_<NAME>_PASS_PIPELINE
  Registry keys:    "test-<name>"
  Factory functions: make_test_<name>_pass_pipeline()
"""

from aster.pass_pipelines import (
    builtin_module,
    amdgcn_module,
    amdgcn_kernel,
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
    phase_amdgcn_backend,
    phase_scf_pipelining,
    phase_nop_insertion,
)

# --------------------------------------------------------------------------- #
# Empty and minimal pipelines
# --------------------------------------------------------------------------- #

# Empty pass pipeline from low-level scheduled assembly, translate to asm only.
TEST_EMPTY_PASS_PIPELINE = builtin_module()

# Minimal pass pipeline from low-level scheduled assembly, assuming we want the
# user not to worry about NOP insertion and automate that process for them.
TEST_MINIMAL_PASS_PIPELINE = builtin_module(
    phase_nop_insertion(delays=0),
)

# --------------------------------------------------------------------------- #
# SROA test pipeline (non-pipelined, scheduling-based)
# --------------------------------------------------------------------------- #

# Used by integration tests for non-pipelined kernels (MFMA, buffer ops, etc.).
TEST_SROA_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    # Note: this is run twice with affine expansion in between, revisit need.
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
    phase_nop_insertion(delays=0),
)

# Backwards compatibility alias
DEFAULT_SROA_PASS_PIPELINE = TEST_SROA_PASS_PIPELINE

# --------------------------------------------------------------------------- #
# Nanobenchmark pipeline
# --------------------------------------------------------------------------- #

# Full pipeline for nanobenchmarks with 0-delay NOPs.
TEST_NANOBENCH_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
    phase_nop_insertion(delays=0),
)

# --------------------------------------------------------------------------- #
# Synchronous debug pipeline
# --------------------------------------------------------------------------- #

# SROA pass pipeline that runs synchronously with 32-delay NOPs.
# Used for debugging races (no wait optimization).
TEST_SYNCHRONOUS_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    # Note: this is run twice with affine expansion in between, revisit need.
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
    phase_nop_insertion(delays=32),
)

# --------------------------------------------------------------------------- #
# Loop test pipeline
# --------------------------------------------------------------------------- #

# Pipeline for loop-specific tests (no scheduling, double lowering).
TEST_LOOP_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    # TODO: Explain what and why and integrate in the relevant phases.
    amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
    PHASE_AMDGCN_BACKEND,
    phase_nop_insertion(delays=0),
)

# --------------------------------------------------------------------------- #
# SCF pipelining test pipeline
# --------------------------------------------------------------------------- #


def make_test_scf_pipelining_pass_pipeline(
    lcm_unroll=False,
    unroll_factor_multiplier=1,
    epilogue_peeling=True,
    ll_sched=False,
):
    return builtin_module(
        PHASE_PRE_SCHEDULING_CLEANUP,
        phase_scf_pipelining(
            lcm_unroll=lcm_unroll,
            unroll_factor_multiplier=unroll_factor_multiplier,
            epilogue_peeling=epilogue_peeling,
        ),
        "aster-destructure-struct-iter-args",
        "canonicalize",
        "cse",
        PHASE_SROA,
        POST_SROA_CLEANUPS,
        PHASE_CONVERT_LDS_BUFFERS,
        PHASE_LOWER_TO_AMDGCN,
        PHASE_EXPAND_MD_OPS,
        PHASE_LOWER_TO_AMDGCN,
        # TODO: Explain what and why and integrate in the relevant phases.
        amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
        phase_amdgcn_backend(ll_sched=ll_sched),
        phase_nop_insertion(delays=0),
    )


TEST_SCF_PIPELINING_PASS_PIPELINE = make_test_scf_pipelining_pass_pipeline()
TEST_SCF_PIPELINING_LL_SCHED_PASS_PIPELINE = make_test_scf_pipelining_pass_pipeline(
    ll_sched=True,
)
