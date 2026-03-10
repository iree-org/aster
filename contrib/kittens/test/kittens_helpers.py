"""Shared infrastructure for kittens test suite."""

import os
from typing import List

import numpy as np
import pytest

from aster.testing import compile_and_run as _compile_and_run
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
LDS_SIZE = 2**16
WAVEFRONT_SIZE = 64


def get_mlir_kernels_library_path(relative: str) -> str:
    """Get path to a file in mlir_kernels/library/ by relative path (e.g. 'common/indexing_ptr.mlir')."""
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "mlir_kernels", "library", relative
    )


def get_kittens_16x16_lds_library_paths() -> List[str]:
    """Get paths for 16x16 MFMA with AGPR accumulators + 16x64b LDS (dwordx4, XOR swizzle).

    Uses global_16x64_b for dwordx4 global loads, lds_16x64_b for XOR-swizzled LDS
    transfers, and compute_16x16_f16 for AGPR MFMA and fire-and-forget C stores.
    """
    base_paths = get_library_paths()
    kittens_dir = os.path.join(os.path.dirname(__file__), "..", "library")
    kittens_paths = [
        get_mlir_kernels_library_path("common/indexing_ptr.mlir"),
        os.path.join(kittens_dir, "global_16x64_b.mlir"),
        os.path.join(kittens_dir, "lds_16x64_b.mlir"),
        os.path.join(kittens_dir, "lds_mfma_16x64_b.mlir"),
        os.path.join(kittens_dir, "compute_16x16_f16.mlir"),
    ]
    return base_paths + kittens_paths


def get_mlir_file(file_name: str) -> str:
    """Get path to a test MLIR file in the kittens test directory."""
    return os.path.join(os.path.dirname(__file__), file_name)


def run_kittens_kernel(
    mlir_file,
    kernel_name,
    input_args=None,
    output_args=None,
    pass_pipeline=None,
    template_substitutions=None,
    library_paths=None,
    grid_dim=(1, 1, 1),
    block_dim=(64, 1, 1),
    num_iterations=1,
    print_ir_after_all=False,
):
    """Compile an MLIR file to HSACO and execute the kernel on GPU."""
    preprocess = None
    if template_substitutions:
        subs = template_substitutions

        def preprocess(content):
            for pattern, replacement in subs.items():
                content = content.replace(pattern, replacement)
            return content

    return _compile_and_run(
        file_name=mlir_file,
        kernel_name=kernel_name,
        input_data=input_args,
        output_data=output_args,
        pass_pipeline=pass_pipeline,
        preprocess=preprocess,
        library_paths=library_paths or get_kittens_16x16_lds_library_paths(),
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        grid_dim=grid_dim,
        block_dim=block_dim,
        num_iterations=num_iterations,
        print_ir_after_all=print_ir_after_all,
    )


def _make_gemm_inputs(K):
    """Create random f16 test matrices for GEMM: A[16xK], B[16xK]."""
    np.random.seed(42)
    A = (np.random.randn(16, K) * 0.05).astype(np.float16)
    B = (np.random.randn(16, K) * 0.05).astype(np.float16)
    return A, B


PIPELINE_STAGE_CONFIGS_4 = {
    # num_stages: (STAGE_GLOBAL_LOAD, STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE)
    # 4-stage split: separates global load from DS write for better pipelining.
    1: (0, 0, 0, 0),
    2: (0, 1, 1, 1),
    3: (0, 1, 2, 2),
    4: (0, 1, 2, 3),
    5: (0, 2, 3, 4),
    6: (0, 3, 4, 5),
}


def pipelined_substitutions_16x32(k, num_stages):
    """Build template substitutions for 16x32 pipelined GEMM tests (4-stage).

    Uses lds_16x64_b.mlir: K_TILES = K//32, 4-stage pipeline (STAGE_GLOBAL_LOAD,
    STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE).
    """
    k_tiles = k // 32
    stride_ab = k * 2
    stage_gl, stage_dw, stage_dr, stage_c = PIPELINE_STAGE_CONFIGS_4[num_stages]
    return {
        "{{K}}": str(k),
        "{{K_TILES}}": str(k_tiles),
        "{{STRIDE_AB}}": str(stride_ab),
        "{{STAGE_GLOBAL_LOAD}}": str(stage_gl),
        "{{STAGE_DS_WRITE}}": str(stage_dw),
        "{{STAGE_DS_READ}}": str(stage_dr),
        "{{STAGE_COMPUTE}}": str(stage_c),
    }


def constexpr_substitutions_16x32(m_tiles, n_tiles, k, num_stages):
    """Build scalar-only template substitutions for constexpr 16x16 MFMA + 16x32 tiles.

    Uses v_mfma_f32_16x16x16_f16 with dwordx4 global loads (16x32 transfer tiles).
    Each 16x32 tile covers K=32 (2 MFMA K-steps of 16 each).
      - M/N per output tile = 16
      - K per transfer tile = 32
      - LDS tile size = 1024 bytes (2 x 512-byte XOR-swizzled 16x16 sub-tiles)
    """
    mn = m_tiles * n_tiles
    k_tiles = k // 32
    stride_ab = k * 2
    stride_c = n_tiles * 16 * 4
    shared_mem = 0
    stage_gl, stage_dw, stage_dr, stage_c = PIPELINE_STAGE_CONFIGS_4[num_stages]

    return {
        "{{M_T}}": str(m_tiles),
        "{{N_T}}": str(n_tiles),
        "{{M_T2}}": str(2 * m_tiles),
        "{{N_T2}}": str(2 * n_tiles),
        "{{MN}}": str(mn),
        "{{M_DIM}}": str(m_tiles * 16),
        "{{N_DIM}}": str(n_tiles * 16),
        "{{K}}": str(k),
        "{{K_TILES}}": str(k_tiles),
        "{{STRIDE_AB}}": str(stride_ab),
        "{{STRIDE_C}}": str(stride_c),
        "{{SHARED_MEM}}": str(shared_mem),
        "{{STAGE_GLOBAL_LOAD}}": str(stage_gl),
        "{{STAGE_DS_WRITE}}": str(stage_dw),
        "{{STAGE_DS_READ}}": str(stage_dr),
        "{{STAGE_COMPUTE}}": str(stage_c),
    }
