"""Common utilities for mlir_kernels tests, benchmarks, and nanobenchmarks."""

import os

_MLIR_KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))

# Pass pipeline for nanobenchmarks with scheduling enabled.
NANOBENCH_PASS_PIPELINE = (
    "builtin.module("
    # Scheduling passes
    "  aster-selective-inlining,"
    "  cse,canonicalize,symbol-dce,"
    "  amdgcn-instruction-scheduling-autoschedule,"
    "  aster-op-scheduling,"
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
    # Convert amdgcn.wait ops to s_waitcnt instructions
    "  amdgcn-convert-waits,"
    "  amdgcn.module("
    "    amdgcn.kernel("
    "      aster-amdgcn-expand-md-ops,"
    "      amdgcn-register-allocation"
    "    )"
    "  ),"
    # Note: removal of test_inst must happen before nop insertion. If it were to
    # happen after, nop insertion could potentially be tripped by test_inst
    # operations that do not materialize in the final asm.
    # TODO: use proper interfaces to avoid this concern.
    "  amdgcn-remove-test-inst,"
    "  amdgcn-nop-insertion{conservative-extra-delays=0}"
    ")"
)


def get_library_paths():
    """Get paths to all required library files."""
    library_dir = os.path.join(_MLIR_KERNELS_DIR, "library", "common")
    return [
        os.path.join(library_dir, "register-init.mlir"),
        os.path.join(library_dir, "indexing.mlir"),
        os.path.join(library_dir, "simple-copies.mlir"),
        os.path.join(library_dir, "copies.mlir"),
        os.path.join(library_dir, "multi-tile-copies.mlir"),
        os.path.join(library_dir, "simple-multi-tile-copies.mlir"),
    ]
