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

def phase_scf_pipelining(lcm_unroll=True, unroll_factor_multiplier=1,
                         epilogue_peeling=True):
    opts = []
    if lcm_unroll:
        opts.append("lcm-unroll=true")
    if unroll_factor_multiplier > 1:
        opts.append(f"unroll-factor-multiplier={unroll_factor_multiplier}")
    if epilogue_peeling:
        opts.append("epilogue-peeling=true")
    if opts:
        return (f"aster-scf-pipeline{{{' '.join(opts)}}}",)
    return ("aster-scf-pipeline",)

PHASE_SCF_PIPELINING = phase_scf_pipelining()

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
# After constexpr expansion, loops are unrolled and canonicalize folds
# memref.alloca(%cN) to memref<Nx...> (static). The same promotion sequence
# as PHASE_CONSTEXPR_EXPANSION must run to promote these back to SSA before
# they reach DPS analysis: sroa splits multi-element memrefs into scalar
# slots, mem2reg promotes them, then to_any/from_any chains fold away.
POST_SROA_CLEANUPS = (
    "cse", "canonicalize", "symbol-dce",
    "aster-constexpr-expansion", "canonicalize",
    # Fold memref.cast on iter_args and forward indexed stores to loads.
    # Must run after constexpr expansion (which unrolls library loops, making
    # constant-index stores visible) and before sroa/mem2reg.
    "aster-simplify-alloca-iter-args",
    "aster-decompose-memref-iter-args",
    # The decomposed scalar iter_args may be struct types (futures) that need
    # further destructuring into individual fields.
    "aster-destructure-struct-iter-args", "canonicalize", "cse",
    "sroa", "mem2reg", "amdgcn-mem2reg",
    "aster-forward-store-to-load",
    "aster-promote-loop-carried-memrefs",
    "cse", "canonicalize",
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
# TODO: NORMAL FORMS for `aster-amdgcn-expand-md-ops`.
PHASE_EXPAND_MD_OPS = amdgcn_module(
    amdgcn_kernel(
        "aster-amdgcn-expand-md-ops",
        "canonicalize", "cse",
    )
)

# Convert LDS buffer operations (alloc_lds, get_lds_offset) to constants.
# Must run after SROA has inlined everything and before lowering to AMDGCN.
# amdgcn-lds-alloc assigns byte offsets to alloc_lds ops; amdgcn-convert-lds-buffers
# then replaces get_lds_offset with the assigned constants. Both must run together.
# amdgcn-lds-alloc is idempotent (skips already-allocated nodes), so it is safe
# to include here even in pipelines that also add it explicitly (e.g. SCF pipelining).
PHASE_CONVERT_LDS_BUFFERS = (
    "amdgcn-lds-alloc",
    "amdgcn-convert-lds-buffers",
    "canonicalize", "cse",
)

# Lowering to LSIR and then AMDGCN
# Note: aster-to-int-arith contains lower-affine without linking in and
# cargo-culting the whole conversion library.
PHASE_LOWER_TO_AMDGCN = (
    # Decompose large affine.apply ops into smaller reusable pieces for
    # better LICM, CSE, and int-range analysis. Must run after canonicalize
    # (which composes affine chains) and before aster-to-int-arith (which
    # lowers affine to arith). Upstream: affine::decompose().
    "affine-expand-index-ops-as-affine",
    "canonicalize", "cse",
    "aster-decompose-affine-apply",
    "loop-invariant-code-motion", "cse",
    "aster-decompose-by-loop-invariant",
    "canonicalize", "cse", "loop-invariant-code-motion",
    "aster-decompose-by-cse", "cse",
    "aster-raise-to-affine",
    "canonicalize", "cse",
    # Hoist loop-invariant decomposed affine sub-expressions and deduplicate.
    # No canonicalize here: it would re-compose the affine chains we just split.
    "loop-invariant-code-motion", "cse",
    # Decompose ptr.ptr_add(affine.apply) into const/uniform/dynamic
    # components using ValueBounds + ThreadUniformAnalysis.
    "aster-affine-optimize-ptr-add{assume-positive=true}",
    "canonicalize",
    # Hoist loop-invariant decomposed affine sub-expressions and deduplicate.
    # No canonicalize here: it would re-compose the affine chains we just split.
    "loop-invariant-code-motion", "cse",
    "canonicalize",
    "aster-factorize-affine-expr",
    "aster-to-int-arith",
    "aster-remove-assume-ops{remove-passthrough=true}",
    "aster-optimize-arith",
    "aster-optimize-ptr-add",
    "canonicalize", "cse",
    "aster-resolve-any-iter-args",
    "aster-amdgcn-set-abi", # "func.func(aster-amdgcn-set-abi)",
    # Convert SCF control flow to AMDGCN control flow
    # Note: control flow support is very limited atm, add NORMAL FORMS
    # to harden invariants.
    "amdgcn-convert-scf-control-flow",
    "canonicalize", "cse",
    "aster-codegen",
    "canonicalize", "cse", "canonicalize",
    "amdgcn-optimize",
    "aster-to-amdgcn",
    amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
    "canonicalize", "cse",
    # TODO: currently busted
    # "aster-apply-sched{scheds=sched}",
    "canonicalize",
)

# Register allocation, and wait lowering.
# TODO: Move NOP insertion to backend.
# TODO: NORMAL FORMS for amdgcn-backend.
def phase_amdgcn_backend(num_vgprs=256, num_agprs=256):
    """Build the amdgcn-backend pipeline string with optional register limits."""
    opts = []
    if num_vgprs != 256:
        opts.append(f"num-vgprs={num_vgprs}")
    if num_agprs != 256:
        opts.append(f"num-agprs={num_agprs}")
    if opts:
        return f"amdgcn-backend{{{' '.join(opts)}}}"
    return "amdgcn-backend"

PHASE_AMDGCN_BACKEND = "amdgcn-backend"

# Note: needs to know about instructions and actual register number for WAW
# dependencies.
# TODO: NORMAL FORMS for amdgcn-hazards.
def phase_nop_insertion(delays=0):
    return (
        # Note: test_inst is added here but it is only relevant for nanobenchmarks.
        # Note: removal of test_inst must happen before nop insertion.
        # TODO: use proper interfaces to avoid this concern.
        "amdgcn-remove-test-inst",
        f"amdgcn-hazards{{v_nops={delays} s_nops={delays}}}",
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
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
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
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
    phase_nop_insertion(delays=32)
)

# Loop pass pipeline
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
    phase_nop_insertion(delays=0)
)

# Loop pipelining pass pipeline
def make_scf_pipelining_pass_pipeline(lcm_unroll=False, unroll_factor_multiplier=1,
                                      epilogue_peeling=True):
    return builtin_module(
        PHASE_PRE_SCHEDULING_CLEANUP,
        phase_scf_pipelining(lcm_unroll=lcm_unroll,
                             unroll_factor_multiplier=unroll_factor_multiplier,
                             epilogue_peeling=epilogue_peeling),
        "aster-destructure-struct-iter-args", "canonicalize", "cse",
        PHASE_SROA,
        POST_SROA_CLEANUPS,
        PHASE_CONVERT_LDS_BUFFERS,
        PHASE_LOWER_TO_AMDGCN,
        PHASE_EXPAND_MD_OPS,
        PHASE_LOWER_TO_AMDGCN,
        # TODO: Explain what and why and integrate in the relevant phases.
        amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
        PHASE_AMDGCN_BACKEND,
        phase_nop_insertion(delays=0)
    )

TEST_SCF_PIPELINING_PASS_PIPELINE = make_scf_pipelining_pass_pipeline()

# Constexpr expansion phase: unroll constexpr tile loops + promote to SSA.
# Must run BEFORE pipelining so the output looks like hand-written kernels.
# Includes upstream mem2reg for index-type memrefs (amdgcn-mem2reg only
# handles register types, token types, and struct wrappers).
PHASE_CONSTEXPR_EXPANSION = (
    "aster-constexpr-expansion", "canonicalize",
    "sroa", "mem2reg", "amdgcn-mem2reg",
    "aster-forward-store-to-load",
    "aster-promote-loop-carried-memrefs",
    "canonicalize",
)

# Constexpr + pipelining pass pipeline: expand constexpr tile loops first,
# then proceed with normal pipelining.
def make_constexpr_pipelining_pass_pipeline(
    lcm_unroll=False, num_vgprs=256, num_agprs=256, unroll_factor_multiplier=1,
    epilogue_peeling=True,
) -> str:
    return builtin_module(
        PHASE_PRE_SCHEDULING_CLEANUP,
        PHASE_CONSTEXPR_EXPANSION,
        phase_scf_pipelining(lcm_unroll=lcm_unroll,
                             unroll_factor_multiplier=unroll_factor_multiplier,
                             epilogue_peeling=epilogue_peeling),
        "aster-destructure-struct-iter-args", "canonicalize", "cse",
        PHASE_SROA,
        POST_SROA_CLEANUPS,
        PHASE_CONVERT_LDS_BUFFERS,
        PHASE_LOWER_TO_AMDGCN,
        # WARNING: PHASE_EXPAND_MD_OPS is NOT idempotent -- running it twice
        # clobbers enable_workgroup_id_x to false (see expand-md-ops-idempotent.mlir).
        # amdgcn-backend already runs expand-md-ops internally, so skip it here.
        # PHASE_EXPAND_MD_OPS,
        # PHASE_LOWER_TO_AMDGCN,
        amdgcn_module(amdgcn_kernel("aster-hoist-ops")),
        phase_amdgcn_backend(num_vgprs=num_vgprs, num_agprs=num_agprs),
        phase_nop_insertion(delays=0)
    )

TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE = make_constexpr_pipelining_pass_pipeline()

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
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
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
    PHASE_CONVERT_LDS_BUFFERS,
    PHASE_EXPAND_MD_OPS,
    PHASE_LOWER_TO_AMDGCN,
    PHASE_AMDGCN_BACKEND,
    phase_nop_insertion(delays=0)
)

# --------------------------------------------------------------------------- #
# Pass pipeline registry for pytest parametrization
# --------------------------------------------------------------------------- #

# Maps short readable names to actual pipeline strings
PASS_PIPELINES = {
    "default": DEFAULT_SROA_PASS_PIPELINE,
    "empty": EMPTY_PASS_PIPELINE,
    "future": FUTURE_SROA_PASS_PIPELINE,
    "loop": TEST_LOOP_PASS_PIPELINE,
    "minimal": MINIMAL_PASS_PIPELINE,
    "nanobench": NANOBENCH_PASS_PIPELINE,
    "synchronous": TEST_SYNCHRONOUS_SROA_PASS_PIPELINE,
    "scf-pipelining": TEST_SCF_PIPELINING_PASS_PIPELINE,
    "constexpr-pipelining": TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
}


def get_pass_pipeline(name: str) -> str:
    """Get a pass pipeline by its short name.

    Args:
        name: Short name of the pipeline (e.g., 'default', 'synchronous', 'future')

    Returns:
        The actual pass pipeline string

    Raises:
        KeyError: If the pipeline name is not found
    """
    if name not in PASS_PIPELINES:
        available = ", ".join(sorted(PASS_PIPELINES.keys()))
        raise KeyError(f"Unknown pass pipeline '{name}'. Available: {available}")
    return PASS_PIPELINES[name]
