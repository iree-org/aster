"""E2E matmul test exercising the real AIR lowering path.

Pipeline:
  mlir-air-opt (preprocess):
    --transform-interpreter
    --air-par-to-herd (forall → herd, each tile = 1 wavefront)
    --one-shot-bufferize
    --air-par-to-launch (outer parallel → launch)
    --air-copy-to-dma (memref.copy → air.dma_memcpy_nd)
    --air-to-amdgcn (flatten hierarchy, herd → wavefront index)
    --convert-memspace-to-amdgcn (integer memspace → #amdgcn.addr_space)
    --convert-linalg-to-amdgcn (air.dma_memcpy_nd + linalg ops → library calls)
  then aster pipeline:
    --preload → inline → mlir-air-to-asm
"""

import os
import shutil
import subprocess

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run

MCPU = "gfx942"

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MLIR_FILE = os.path.join(_THIS_DIR, "..", "air-to-amdgcn-matmul.mlir")
_TRANSFORM_FILE = os.path.join(_THIS_DIR, "..", "air-to-amdgcn-matmul-transform.mlir")
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


def _find_mlir_air_opt():
    """Find the mlir-air-opt binary."""
    build_path = os.path.join(
        _THIS_DIR, "..", "..", "..", "..", "build", "bin", "mlir-air-opt"
    )
    if os.path.isfile(build_path):
        return os.path.abspath(build_path)
    path = shutil.which("mlir-air-opt")
    if path:
        return path
    pytest.skip("mlir-air-opt not found")


def _air_preprocess(mlir_text):
    """Run the full AIR lowering pipeline before handing to aster."""
    opt = _find_mlir_air_opt()
    result = subprocess.run(
        [
            opt,
            f"--transform-preload-library=transform-library-paths={_TRANSFORM_FILE}",
            "--transform-interpreter",
            "--air-par-to-herd",
            "--canonicalize", "--cse",
            "--one-shot-bufferize",
            "--canonicalize", "--cse",
            "--air-par-to-launch=has-air-segment=true",
            "--canonicalize", "--cse",
            "--air-copy-to-dma",
            "--air-to-amdgcn",
            "--canonicalize",
            "--convert-memspace-to-amdgcn",
            "--convert-linalg-to-amdgcn",
        ],
        input=mlir_text,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"mlir-air-opt AIR preprocessing failed:\n{result.stderr}")
    return result.stdout


def _post_air_pipeline(library_paths):
    libs = ",".join(library_paths)
    return (
        "builtin.module("
        "canonicalize,"
        f"amdgcn-preload-library{{library-paths={libs}}},"
        "inline, symbol-dce, canonicalize,"
        "mlir-air-to-asm)"
    )


class TestAirMatmulE2E:

    def test_matmul_64x64(self):
        M, N, K = 64, 64, 64
        np.random.seed(42)
        A = (np.random.randn(M, K) * 0.1).astype(np.float16)
        B_KxN = (np.random.randn(K, N) * 0.1).astype(np.float16)
        B_T = np.ascontiguousarray(B_KxN.T)
        # C must be zero-initialized (fill is erased by convert-linalg-to-amdgcn;
        # the library's zero_C handles accumulator init per tile).
        C = np.zeros(M * N, dtype=np.float32)

        compile_and_run(
            file_name=_MLIR_FILE,
            kernel_name="matmul_f16_64x64",
            input_data=[A.flatten(), B_T.flatten()],
            output_data=[C],
            pass_pipeline=_post_air_pipeline(_LIBRARY_PATHS),
            library_paths=[],
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),  # 2 wavefronts (2x1 herd)
            preprocess=_air_preprocess,
        )

        expected = (A.astype(np.float32) @ B_KxN.astype(np.float32)).flatten()
        np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)
