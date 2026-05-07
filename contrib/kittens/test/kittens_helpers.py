"""Shared infrastructure for kittens test suite."""

import os
from typing import List

import numpy as np

from aster.execution.helpers import compile_and_run as _compile_and_run
from mlir_kernels.common import get_library_paths


def get_mlir_kernels_library_path(relative: str) -> str:
    """Get path to a file in mlir_kernels/library/ by relative path (e.g. 'common/indexing_ptr.mlir')."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "mlir_kernels", "library", relative)


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
    *,
    mcpu: str,
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
    """Compile an MLIR file to HSACO and execute the kernel on GPU.

    ``mcpu`` is required (no silent default). The wavefront size is
    derived from the target arch.
    """
    from aster.core.target import Target

    target = Target.from_mcpu(mcpu)

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
        mcpu=mcpu,
        wavefront_size=target.wavefront_size,
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


# Pipeline strategies: strategy -> dict with named stage assignments.
# 7 keys: A_LOAD, A_LDS_WRITE, A_LDS_READ, B_LOAD, B_LDS_WRITE, B_LDS_READ, COMPUTE.
# Some implementations may not use all keys (e.g., direct_b skips B_LDS_WRITE/READ).
# Strategy >= 5: all A phases in separate stages (hoist-wait hoists all A waits).
# Strategy >= 6: all A+B phases in separate stages (hoist-wait hoists everything).
#
# fmt: off
PIPELINE_STRATEGIES = {
    #                  A_LOAD  A_LDS_W A_LDS_R   B_LOAD  B_LDS_W B_LDS_R   COMPUTE
    # Odd = symmetric baselines, Even = asymmetric A/B write stages.
    # Invariant: A_LDS_READ == B_LDS_READ (shared s_barrier).
    0:  dict(A_LOAD=0, A_LDS_WRITE=0, A_LDS_READ=0, B_LOAD=0, B_LDS_WRITE=0, B_LDS_READ=0, COMPUTE=0),
    1:  dict(A_LOAD=0, A_LDS_WRITE=1, A_LDS_READ=1, B_LOAD=1, B_LDS_WRITE=1, B_LDS_READ=1, COMPUTE=1),
    2:  dict(A_LOAD=0, A_LDS_WRITE=0, A_LDS_READ=1, B_LOAD=0, B_LDS_WRITE=1, B_LDS_READ=1, COMPUTE=1),
    3:  dict(A_LOAD=0, A_LDS_WRITE=1, A_LDS_READ=2, B_LOAD=0, B_LDS_WRITE=1, B_LDS_READ=2, COMPUTE=2),
    4:  dict(A_LOAD=0, A_LDS_WRITE=2, A_LDS_READ=2, B_LOAD=0, B_LDS_WRITE=1, B_LDS_READ=2, COMPUTE=2),
    5:  dict(A_LOAD=0, A_LDS_WRITE=1, A_LDS_READ=2, B_LOAD=2, B_LDS_WRITE=2, B_LDS_READ=2, COMPUTE=3),
    6:  dict(A_LOAD=0, A_LDS_WRITE=2, A_LDS_READ=2, B_LOAD=1, B_LDS_WRITE=1, B_LDS_READ=2, COMPUTE=3),
    7:  dict(A_LOAD=0, A_LDS_WRITE=2, A_LDS_READ=3, B_LOAD=0, B_LDS_WRITE=2, B_LDS_READ=3, COMPUTE=4),
    8:  dict(A_LOAD=0, A_LDS_WRITE=1, A_LDS_READ=3, B_LOAD=0, B_LDS_WRITE=2, B_LDS_READ=3, COMPUTE=4),
    9:  dict(A_LOAD=0, A_LDS_WRITE=3, A_LDS_READ=4, B_LOAD=0, B_LDS_WRITE=3, B_LDS_READ=4, COMPUTE=5),
    10: dict(A_LOAD=0, A_LDS_WRITE=2, A_LDS_READ=4, B_LOAD=0, B_LDS_WRITE=3, B_LDS_READ=4, COMPUTE=5),
}
# fmt: on


def pipeline_strategy_stages(strategy):
    """Return (a_stages, b_stages) for the given strategy number.

    Stages = max stage number + 1.
    """
    s = PIPELINE_STRATEGIES[strategy]
    a_max = max(s["A_LOAD"], s["A_LDS_WRITE"], s["A_LDS_READ"], s["COMPUTE"])
    b_max = max(s["B_LOAD"], s["B_LDS_WRITE"], s["B_LDS_READ"], s["COMPUTE"])
    return a_max + 1, b_max + 1


def pipeline_strategy_substitutions(strategy):
    """Return template substitution dict for the given strategy number."""
    s = PIPELINE_STRATEGIES[strategy]
    return {
        "{{A_STAGE_LOAD}}": str(s["A_LOAD"]),
        "{{A_STAGE_WRITE}}": str(s["A_LDS_WRITE"]),
        "{{A_STAGE_READ}}": str(s["A_LDS_READ"]),
        "{{A_STAGE_COMPUTE}}": str(s["COMPUTE"]),
        "{{B_STAGE_LOAD}}": str(s["B_LOAD"]),
        "{{B_STAGE_WAIT}}": str(s["B_LDS_READ"]),
        "{{B_A_STAGE_COMPUTE}}": str(s["COMPUTE"]),
    }


def pipelined_substitutions_16x32(k, pipeline_strategy):
    """Build template substitutions for 16x32 pipelined GEMM tests (A-only)."""
    k_tiles = k // 32
    stride_ab = k * 2
    subs = pipeline_strategy_substitutions(pipeline_strategy)
    subs.update(
        {
            "{{K}}": str(k),
            "{{K_TILES}}": str(k_tiles),
            "{{STRIDE_AB}}": str(stride_ab),
        }
    )
    return subs


def constexpr_substitutions_16x32(m_tiles, n_tiles, k, pipeline_strategy):
    """Build template substitutions for constexpr 16x16 MFMA + 16x32 tiles."""
    mn = m_tiles * n_tiles
    k_tiles = k // 32
    stride_ab = k * 2
    stride_c = n_tiles * 16 * 4
    shared_mem = 0

    subs = pipeline_strategy_substitutions(pipeline_strategy)
    subs.update(
        {
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
        }
    )
    return subs
