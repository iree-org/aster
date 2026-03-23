"""Shared infrastructure for kittens test suite."""

import os
from typing import List

import numpy as np
import pytest

from aster.execution.helpers import compile_and_run as _compile_and_run
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


def get_kittens_16x16_lds_library_paths(use_buffer: bool = False) -> List[str]:
    """Get paths for 16x16 MFMA with AGPR accumulators + 16x64_b LDS (dwordx4, XOR swizzle).

    use_buffer: True for buffer_load/buffer_store (MUBUF), False for flat.
    """
    suffix = "_buf" if use_buffer else ""
    base_paths = get_library_paths()
    kittens_dir = os.path.join(os.path.dirname(__file__), "..", "library")
    kittens_paths = [
        get_mlir_kernels_library_path("common/indexing_ptr.mlir"),
        os.path.join(kittens_dir, f"global_16x64_b{suffix}.mlir"),
        os.path.join(kittens_dir, "lds_16x64_b.mlir"),
        os.path.join(kittens_dir, "lds_mfma_16x64_b.mlir"),
        os.path.join(kittens_dir, f"compute_16x16_f16{suffix}.mlir"),
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


def shuffle_weight(B):
    """Preshuffle B[N, K] for MFMA 16x16x16 direct global-to-register loads.

    Packs each lane's 16-byte load with the exact data the MFMA instruction
    expects: [K-step-0 (4 f16), K-step-1 (4 f16)], accounting for the
    fragment layout where lane groups access non-contiguous K positions.

    MFMA 16x16x16 f16: lane d needs B[d%16, (d/16)*4 + 0..3] per K-step.
    A 32-element K-tile gives 2 K-steps (K=0..15 and K=16..31).
    The K decomposition within a 32-element block:
      k_local = kh * 16 + lane_group * 4 + k_within
      where kh in {0,1}, lane_group in {0..3}, k_within in {0..3}

    Original B[N, K] is reshaped to (n_blocks, 16, k_blocks, 2, 4, 4)
    dims: (nb, n_lane, kb, kh, lane_group, k_within)
    then transposed to: (nb, kb, lane_group, n_lane, kh, k_within)
    so that each lane's 8 contiguous elements are [kh=0 kw=0..3, kh=1 kw=0..3].
    """
    N, K = B.shape
    assert N % 16 == 0, f"N={N} must be multiple of 16"
    assert K % 32 == 0, f"K={K} must be multiple of 32"

    n_blocks = N // 16
    k_blocks = K // 32

    # Reshape: (nb, n_lane=16, kb, kh=2, lane_group=4, k_within=4)
    B6 = B.reshape(n_blocks, 16, k_blocks, 2, 4, 4)
    # Transpose to: (nb, kb, lane_group, n_lane, kh, k_within)
    B6 = B6.transpose(0, 2, 4, 1, 3, 5)
    return np.ascontiguousarray(B6).reshape(N, K)


def _make_gemm_inputs(K):
    """Create random f16 test matrices for GEMM: A[16xK], B[16xK]."""
    np.random.seed(42)
    A = (np.random.randn(16, K) * 0.05).astype(np.float16)
    B = (np.random.randn(16, K) * 0.05).astype(np.float16)
    return A, B


# A pipeline stage configs (LDS path: 4 phases).
# a_stages -> (a_load, a_lds_write, a_lds_read, a_compute)
A_STAGE_CONFIGS = {
    1: (0, 0, 0, 0),
    2: (0, 1, 1, 1),
    3: (0, 1, 2, 2),
    4: (0, 1, 2, 3),
    5: (0, 2, 3, 4),
    6: (0, 3, 4, 5),
}


def ab_stage_config(a_stages, b_stages):
    """Compute absolute stage numbers for A (LDS) and B (direct) pipeline phases.

    A phases: a_load, a_lds_write, a_lds_read, a_compute.
    B phases: b_load, b_wait, b_compute.
    Both share the last stage for compute.

    b_stages controls the gap between b_load and b_compute:
      b_stages=1: all B at compute (no pipelining).
      b_stages=2: b_load 1 stage before compute, b_wait at compute.
      b_stages=3: b_load 2 stages before compute, b_wait 1 stage before.
      b_stages=N: b_load N-1 stages before compute, b_wait N-2 stages before.

    The combined pipeline depth = max(a_stages, b_stages).
    Higher b_stages means more B loads in flight (more VGPR pressure,
    more latency hiding).
    """
    a_cfg = A_STAGE_CONFIGS[a_stages]
    a_load, a_lds_write, a_lds_read, a_compute = a_cfg

    # Combined compute stage.
    compute = max(a_compute, b_stages - 1)

    # Shift A up if B needs a deeper pipeline.
    if compute > a_compute:
        offset = compute - a_compute
        a_load += offset
        a_lds_write += offset
        a_lds_read += offset
        a_compute = compute

    # B: spread b_load and b_wait backwards from compute.
    # b_load is (b_stages - 1) stages before compute.
    # b_wait is 1 stage before compute (or at compute if b_stages <= 2).
    if b_stages <= 1:
        b_load = compute
        b_wait = compute
    elif b_stages == 2:
        b_load = compute - 1
        b_wait = compute
    else:
        b_load = compute - (b_stages - 1)
        b_wait = compute - 1
    b_compute = compute

    return (
        (a_load, a_lds_write, a_lds_read, a_compute),
        (b_load, b_wait, b_compute),
    )


def pipelined_substitutions_16x32(k, a_stages):
    """Build template substitutions for 16x32 pipelined GEMM tests (A-only)."""
    k_tiles = k // 32
    stride_ab = k * 2
    a_load, a_lds_write, a_lds_read, a_compute = A_STAGE_CONFIGS[a_stages]
    return {
        "{{K}}": str(k),
        "{{K_TILES}}": str(k_tiles),
        "{{STRIDE_AB}}": str(stride_ab),
        "{{A_STAGE_LOAD}}": str(a_load),
        "{{A_STAGE_WRITE}}": str(a_lds_write),
        "{{A_STAGE_READ}}": str(a_lds_read),
        "{{A_STAGE_COMPUTE}}": str(a_compute),
    }


def constexpr_substitutions_16x32(m_tiles, n_tiles, k, a_stages, b_stages=None):
    """Build template substitutions for constexpr 16x16 MFMA + 16x32 tiles.

    a_stages: A pipeline depth (LDS path, 1-6).
    b_stages: B pipeline depth (direct_b, 1-3). None = same as a_stages.
    """
    mn = m_tiles * n_tiles
    k_tiles = k // 32
    stride_ab = k * 2
    stride_c = n_tiles * 16 * 4
    shared_mem = 0

    effective_b = b_stages if b_stages is not None else a_stages
    (a_load, a_lds_write, a_lds_read, a_compute), (b_load, b_wait, b_compute) = (
        ab_stage_config(a_stages, effective_b)
    )

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
        "{{A_STAGE_LOAD}}": str(a_load),
        "{{A_STAGE_WRITE}}": str(a_lds_write),
        "{{A_STAGE_READ}}": str(a_lds_read),
        "{{A_STAGE_COMPUTE}}": str(a_compute),
        "{{B_STAGE_LOAD}}": str(b_load),
        "{{B_STAGE_WAIT}}": str(b_wait),
        "{{B_A_STAGE_COMPUTE}}": str(b_compute),
    }
