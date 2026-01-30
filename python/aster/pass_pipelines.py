"""Common pass pipelines used across the codebase."""

# Minimal pass pipeline without DCE or optimizations that might remove operations
MINIMAL_PASS_PIPELINE = "builtin.module()"

# Default pass pipeline for integration tests
DEFAULT_SROA_PASS_PIPELINE = (
    "builtin.module("
    "  aster-selective-inlining,"
    "  cse,canonicalize,symbol-dce,"
    # Scheduling passes relieves the burden of synchronization interleaving from
    # the API while still maintaining good control over the schedule.
    # This is one possible design point in the control / automation tradeoff space.
    "  amdgcn-instruction-scheduling-autoschedule,"
    "  aster-op-scheduling,"
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    # Note: SROA requires inlining of everything and canonicalization of GPU
    # quantities to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    # Try wait optimization early
    # Note: analysis does not support branches so full inlining is required.
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: SROA requires inlining of everything and canonicalization of GPU
    # quantities to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  affine-expand-index-ops-as-affine,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the pass happens correctly..
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: allocates special registers and does not work with function calls.
    # This is really needed before optimizing straight-line waits otherwise we
    # may miss some dependencies (e.g. s_load_dwordx2 does not yet exist).
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    # Note: this must have a aster-amdgcn-expand-md-ops run before to expose
    # s_load_dwordx2.
    # TODO: NORMAL FORMS or include in pass.
    # Note: going to lsir early will make memory dependency more conservative,
    # resulting in more waits during amdgcn-optimize-straight-line-waits.
    # TODO: NORMAL FORMS or include in pass.
    "  amdgcn-optimize-straight-line-waits,"
    #
    # Note: convert to lsir and AMDGCN after straight-line wait optimization.
    # Note: aster-to-int-arith contains lower-affine without linking in and
    # cargo-culting the whole conversion library.
    "  aster-to-int-arith,"
    "  aster-optimize-arith,"
    "  aster-amdgcn-set-abi,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  canonicalize,cse,"
    "  canonicalize,"
    "  aster-to-amdgcn,"
    # Convert amdgcn.wait ops to s_waitcnt instructions
    "  amdgcn-convert-waits,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the interference graph is built correctly...
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: this is really needed to lower away threadidx etc ops into alloc that
    # can be relocated
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops,"
    "      amdgcn-register-allocation"
    "    )"
    "  ),"
    #
    # Note: needs to know about instructions and actual register number for
    # WAW dependencies.
    "  amdgcn-nop-insertion{conservative-extra-delays=0}"
    ")"
)

# SROA pass pipeline that runs synchronously, i.e. no wait optimization and extra
# NOP insertion. This is used for debugging races.
SYNCHRONOUS_SROA_PASS_PIPELINE = (
    "builtin.module("
    "  aster-selective-inlining,"
    "  cse,canonicalize,symbol-dce,"
    # Scheduling passes relieves the burden of synchronization interleaving from
    # the API while still maintaining good control over the schedule.
    # This is one possible design point in the control / automation tradeoff space.
    "  amdgcn-instruction-scheduling-autoschedule,"
    "  aster-op-scheduling,"
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    # Note: SROA requires inlining of everything to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    # Try wait optimization early
    # Note: analysis does not support branches so full inlining is required.
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  cse,canonicalize,symbol-dce,"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: SROA requires inlining of everything to properly kick in.
    # TODO: NORMAL FORMS or include in pass.
    "  affine-expand-index-ops-as-affine,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the pass happens correctly..
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: allocates special registers and does not work with function calls.
    # This is really needed before optimizing straight-line waits otherwise we
    # may miss some dependencies (e.g. s_load_dwordx2 does not yet exist).
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    # Note: this must have a aster-amdgcn-expand-md-ops run before to expose
    # s_load_dwordx2.
    # TODO: NORMAL FORMS or include in pass.
    # Note: going to LSIR early will make memory dependency more conservative,
    # resulting in more waits during amdgcn-optimize-straight-line-waits.
    # TODO: NORMAL FORMS or include in pass.
    # "  amdgcn-optimize-straight-line-waits,"
    #
    # Note: convert to LSIR and AMDGCN after straight-line wait optimization.
    # Note: aster-to-int-arith contains lower-affine without linking in and
    # cargo-culting the whole conversion library.
    "  aster-to-int-arith,"
    "  aster-optimize-arith,"
    "  aster-amdgcn-set-abi,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  canonicalize,cse,"
    "  canonicalize,"
    "  aster-to-amdgcn,"
    # Convert amdgcn.wait ops to s_waitcnt instructions
    "  amdgcn-convert-waits,"
    # Note: this really must happen on amdgcn.kernel within a module to ensure
    # that the interference graph is built correctly...
    "  amdgcn.module("
    "    amdgcn.kernel("
    # Note: this is really needed to lower away threadidx etc ops into alloc that
    # can be relocated
    # TODO: NORMAL FORMS or include in pass.
    "      aster-amdgcn-expand-md-ops,"
    "      amdgcn-register-allocation"
    "    )"
    "  ),"
    #
    # Note: needs to know about instructions and actual register number for
    # WAW dependencies.
    "  amdgcn-nop-insertion{conservative-extra-delays=32}"
    ")"
)

# Minimal pass pipeline for nanobenchmarks - no scheduling, synchronous waits.
# This produces unoptimized code that reflects the raw instruction sequence.
NANOBENCH_PASS_PIPELINE = (
    "builtin.module("
    # Skip scheduling
    "  aster-selective-inlining{allow-scheduled-calls=true},"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  aster-replace-constant-gpu-dims,cse,canonicalize,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    "  affine-expand-index-ops-as-affine,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  aster-constexpr-expansion,cse,canonicalize,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-amdgcn-expand-md-ops"
    "    )"
    "  ),"
    "  aster-to-int-arith,"
    "  aster-optimize-arith,"
    "  aster-amdgcn-set-abi,"
    "  aster-to-lsir,"
    "  canonicalize,cse,"
    "  aster-amdgcn-select-reg-classes,"
    "  canonicalize,cse,"
    "  canonicalize,"
    "  aster-to-amdgcn,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-amdgcn-expand-md-ops,"
    "      amdgcn-register-allocation"
    "    )"
    "  ),"
    # Convert amdgcn.wait ops to s_waitcnt instructions
    "  amdgcn-convert-waits,"
    # Note: removal of test_inst must happen before nop insertion. If it were to
    # happen after, nop insertion could potentially be tripped by test_inst
    # operations that do not materialize in the final asm.
    # TODO: use proper interfaces to avoid this concern.
    "  amdgcn-remove-test-inst,"
    "  amdgcn-nop-insertion{conservative-extra-delays=0}"
    ")"
)

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
