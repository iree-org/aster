import os

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run

MCPU = "gfx942"

# TODO: clean this up
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR_FILE = os.path.join(_THIS_DIR, "..", "linalg-to-amdgcn.mlir")
_LIBRARY_DIR = os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "mlir_kernels", "library"
)
_KITTENS_DIR = os.path.join(
    _THIS_DIR, "..", "..", "..", "..", "contrib", "kittens", "library"
)

_LIBRARY_PATHS = [
    os.path.join(_LIBRARY_DIR, "common", f)
    for f in [
        "register-init.mlir",
        "indexing.mlir",
        "indexing_ptr.mlir",
        "futures.mlir",
    ]
] + [
    os.path.join(_KITTENS_DIR, f)
    for f in [
        "compute_16x16_f16.mlir",
        "global_16x64_b.mlir",
        "lds_16x64_b.mlir",
        "lds_mfma_16x64_b.mlir",
    ]
]


def _mlir_air_pipeline(library_paths):
    libs = ",".join(library_paths)
    return (
        "builtin.module("
        "transform-interpreter, canonicalize,"
        "convert-linalg-to-amdgcn,"
        f"amdgcn-preload-library{{library-paths={libs}}},"
        "inline, symbol-dce, canonicalize,"
        "mlir-air-to-asm)"
    )


class TestLinalgMatmulE2E:

    def test_matmul_32x32(self):
        M, N, K = 32, 32, 32
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B_KxN = (np.random.randn(K, N) * 0.1).astype(np.float16)
        B_T = np.ascontiguousarray(B_KxN.T)
        C = np.zeros(M * N, dtype=np.float32)

        compile_and_run(
            file_name=_MLIR_FILE,
            kernel_name="matmul_f16_32x32",
            input_data=[A.flatten(), B_T.flatten()],
            output_data=[C],
            pass_pipeline=_mlir_air_pipeline(_LIBRARY_PATHS),
            library_paths=[],
            grid_dim=(1, 1, 1),
            block_dim=(64, 1, 1),
        )

        expected = (A.astype(np.float32) @ B_KxN.astype(np.float32)).flatten()
        np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)
