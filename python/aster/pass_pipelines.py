"""Common pass pipelines used across the codebase."""

# --------------------------------------------------------------------------- #
# Helpers for compositional pass pipelines.
# --------------------------------------------------------------------------- #


def _flatten_and_clean(args):
    """Flattens nested lists/tuples and cleans up pass strings."""
    passes = []
    for arg in args:
        if isinstance(arg, (list, tuple)):
            passes.extend(_flatten_and_clean(arg))
        elif isinstance(arg, str):
            # Remove whitespace and trailing commas to ensure clean joining
            cleaned = arg.strip()
            if cleaned.endswith(","):
                cleaned = cleaned[:-1]
            if cleaned:
                passes.append(cleaned)
    return passes


def _pipeline_str(*args):
    """Joins passes with commas."""
    return ",".join(_flatten_and_clean(args))


def builtin_module(*args):
    return f"builtin.module({_pipeline_str(*args)})"


def amdgcn_module(*args):
    return f"amdgcn.module({_pipeline_str(*args)})"


def amdgcn_kernel(*args):
    return f"amdgcn.kernel({_pipeline_str(*args)})"


# --------------------------------------------------------------------------- #
# Reusable Logical Phases
# In the future, phase transitions should be checked by normal forms.
# --------------------------------------------------------------------------- #

# fmt: off

# Pre-scheduling cleanup, main purpose is to remove all included libraries that
# are not needed for a particular kernel.
PHASE_PRE_SCHEDULING_CLEANUP = (
    "aster-selective-inlining",
    "cse", "canonicalize", "symbol-dce",
)

# Scheduling passes relieves the burden of synchronization interleaving from
# the API while still maintaining good control over the schedule.
# This is one possible design point in the control / automation tradeoff space.
PHASE_SCHEDULING = (
    "amdgcn-instruction-scheduling-autoschedule",
    "aster-op-scheduling",
)

# Cleanup after scheduling or initially if scheduling is skipped
PHASE_POST_SCHEDULING_CLEANUP = (
    "aster-selective-inlining{allow-scheduled-calls=true}",
    "aster-replace-constant-gpu-dims", "cse", "canonicalize",
)

# Common SROA and memory optimization sequence.
# This is used to enable composable functions and reusable APIs.
# Values are returned through memref that act as a type eraser and must sroa +
# mem2reg away.
# This is a natural fit to implement a usable form of templating in MLIR and
# relying on canonicalization, folding, sroa, memreg to clean up.
# In practice this is quite powerful and avoids having to upfront the invention
# of yet another thing (**cough cough language or DSL**) to make MLIR usable for
# our specific ASM goals.
# Note: SROA requires inlining of everything to properly kick in.
# TODO: NORMAL FORMS or include in pass.
PHASE_SROA = (
    "cse", "canonicalize", "sroa",
    "cse", "canonicalize", "amdgcn-mem2reg",
    "aster-selective-inlining{allow-scheduled-calls=true}",
)

# Intermediate cleanup and expansion (Default/Sync version)
POST_SROA_CLEANUPS = (
    "cse", "canonicalize", "symbol-dce",
    "aster-constexpr-expansion", "cse", "canonicalize",
)

# Affine expansion
PHASE_AFFINE_EXPANSION = (
    "affine-expand-index-ops-as-affine",
)

# Backend preparation (MD ops expansion)
# TODO: this really must happen on amdgcn.kernel within a module to ensure
# that the pass happens correctly.. this should not be the case, reevaluate.
# Note: aster-amdgcn-expand-md-ops allocates special registers and does not yet
# work correctly across function calls.
# This is really needed before optimizing straight-line waits otherwise we
# may miss some dependencies (e.g. s_load_dwordx2 does not yet exist).
# TODO: NORMAL FORMS for `aster-amdgcn-expand-md-ops`.
PHASE_EXPAND_MD_OPS = amdgcn_module(
    amdgcn_kernel(
        "aster-amdgcn-expand-md-ops",
        "canonicalize", "cse",
    )
)

# Lowering to LSIR and then AMDGCN
# Note: convert to lsir and AMDGCN after straight-line wait optimization.
# Note: aster-to-int-arith contains lower-affine without linking in and
# cargo-culting the whole conversion library.
PHASE_LOWER_TO_AMDGCN = (
    "aster-to-int-arith",
    "aster-optimize-arith",
    "aster-amdgcn-set-abi", # "func.func(aster-amdgcn-set-abi)",
    # Convert SCF control flow to AMDGCN control flow
    # Note: control flow support is very limited atm, add NORMAL FORMS
    # to harden invariants.
    "amdgcn-convert-scf-control-flow",
    "canonicalize", "cse",
    "aster-to-lsir",
    "canonicalize", "cse",
    "aster-amdgcn-select-reg-classes",
    "canonicalize", "cse", "canonicalize",
    "aster-to-amdgcn",
)

# Optimize straight-line waits
# Note: this must have a aster-amdgcn-expand-md-ops run before to expose
# s_load_dwordx2.
# TODO: NORMAL FORMS or include in pass.
# Note: going to lsir early will make memory dependency more conservative,
# resulting in more waits during amdgcn-optimize-straight-line-waits.
# TODO: NORMAL FORMS or include in pass.
# IMPORTANT: this is on a path to deprecation in favor of `amdgcn.wait` usage
# and `PHASE_CONVERT_WAITS` to support a future-based ASM programming model.
PHASE_OPTIMIZE_STRAIGHT_LINE_WAITS = (
    "amdgcn-optimize-straight-line-waits",
)

# Convert amdgcn.wait ops to s_waitcnt instructions
PHASE_CONVERT_WAITS = (
    "amdgcn-convert-waits",
)

# Register allocation
# Note: this really must happen on amdgcn.kernel within a module to ensure that
# the interference graph is built correctly..  this should not be the case, reevaluate.
# Note: `aster-amdgcn-expand-md-ops` again as it is really needed to lower away
# threadidx etc ops into alloc that can be relocated.
# TODO: NORMAL FORMS for amdgcn-register-allocation.
PHASE_REGISTER_ALLOCATION = amdgcn_module(
    amdgcn_kernel(
        "aster-amdgcn-expand-md-ops",
        "amdgcn-register-allocation",
        "canonicalize", "cse",
    )
)

# Note: needs to know about instructions and actual register number for WAW
# dependencies.
# TODO: NORMAL FORMS for amdgcn-nop-insertion.
def phase_nop_insertion(delays=0):
    return (
        # Note: test_inst is added here but it is only relevant for nanobenchmarks.
        # But it is tightly coupled to `amdgcn-nop-insertion`.
        # When we have normal forms we can separate.
        # Note: removal of test_inst must happen before nop insertion. If it were to
        # happen after, nop insertion could potentially be tripped by test_inst
        # operations that do not materialize in the final asm.
        # TODO: use proper interfaces to avoid this concern.
        "amdgcn-remove-test-inst",
        f"amdgcn-nop-insertion{{conservative-extra-delays={delays}}}",
    )

# --------------------------------------------------------------------------- #
# Test and benchmarking pipelines
# --------------------------------------------------------------------------- #

# Pass pipeline for nanobenchmarks.
NANOBENCH_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_REGISTER_ALLOCATION,
    PHASE_CONVERT_WAITS,
    phase_nop_insertion(delays=0)
)

# SROA pass pipeline that runs synchronously, i.e. no wait optimization and extra
# NOP insertion. This is used for debugging races.
TEST_SYNCHRONOUS_SROA_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    # Note: this is run twice with affine expansion in between, revisit need.
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_EXPAND_MD_OPS,
    # In synchronous mode we do not optimize straight-line waits, we want them
    # exactly as specified by the programmer.
    # PHASE_OPTIMIZE_STRAIGHT_LINE_WAITS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_CONVERT_WAITS,
    PHASE_REGISTER_ALLOCATION,
    phase_nop_insertion(delays=32)
)

# Loop pass pipeline
TEST_LOOP_PASS_PIPELINE = builtin_module(
    PHASE_LOWER_TO_AMDGCN,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_CONVERT_WAITS,
    # TODO: Explain what and why and integrate in the relevant phases.
    amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
    PHASE_REGISTER_ALLOCATION
)

# --------------------------------------------------------------------------- #
# General pipelines for specific use cases
# --------------------------------------------------------------------------- #

# Empty pass pipeline from low-level scheduled assembly, translate to asm only.
EMPTY_PASS_PIPELINE = builtin_module()

# Minimal pass pipeline from low-level scheduled assembly, assuming we want the
# user not to worry about NOP insertion and automate that process for them.
MINIMAL_PASS_PIPELINE = builtin_module(
    phase_nop_insertion(delays=0)
)

# Default pass pipeline for integration tests
DEFAULT_SROA_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    # Note: this is run twice with affine expansion in between, revisit need.
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_EXPAND_MD_OPS,
    PHASE_OPTIMIZE_STRAIGHT_LINE_WAITS,
    PHASE_LOWER_TO_AMDGCN,
    # Convert amdgcn.wait ops to s_waitcnt instructions
    PHASE_CONVERT_WAITS,
    PHASE_REGISTER_ALLOCATION,
    phase_nop_insertion(delays=0)
)

# Default pass pipeline for integration tests
FUTURE_SROA_PASS_PIPELINE = builtin_module(
    PHASE_PRE_SCHEDULING_CLEANUP,
    PHASE_SCHEDULING,
    PHASE_POST_SCHEDULING_CLEANUP,
    # Note: this is run twice with affine expansion in between, revisit need.
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_AFFINE_EXPANSION,
    PHASE_SROA,
    POST_SROA_CLEANUPS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    # Convert amdgcn.wait ops to s_waitcnt instructions
    PHASE_CONVERT_WAITS,
    PHASE_REGISTER_ALLOCATION,
    phase_nop_insertion(delays=0)
)
