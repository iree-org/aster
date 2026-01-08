"""Common utilities for mlir_kernels tests, benchmarks, and nanobenchmarks."""

import os

_MLIR_KERNELS_DIR = os.path.dirname(os.path.abspath(__file__))

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
    "  amdgcn-constexpr-expansion,cse,canonicalize,"
    "  affine-expand-index-ops-as-affine,"
    "  cse,canonicalize,sroa,"
    "  cse,canonicalize,amdgcn-mem2reg,"
    "  cse,canonicalize,symbol-dce,"
    "  amdgcn-constexpr-expansion,cse,canonicalize,"
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
    "  amdgcn-nop-insertion{conservative-extra-delays=0},"
    "  amdgcn-remove-test-inst"
    ")"
)


def get_library_paths():
    """Get paths to all required library files."""
    library_dir = os.path.join(_MLIR_KERNELS_DIR, "library", "common")
    return [
        os.path.join(library_dir, "register_init.mlir"),
        os.path.join(library_dir, "indexing.mlir"),
        os.path.join(library_dir, "copies.mlir"),
        os.path.join(library_dir, "multi-tile-copies.mlir"),
    ]

