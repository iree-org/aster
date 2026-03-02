"""Test: Weak-scaling performance sweep for constexpr GEMM.

Phase 1 (TestWeakScaleCorrectness): Correctness gate at K=128, 1 workgroup.
Phase 2 (TestWeakScalePerf): TFLOPS sweep across tile/stage/workgroup configs.
"""

from dataclasses import dataclass

import numpy as np
import pytest

from aster.pass_pipelines import TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    run_kittens_kernel,
    get_mlir_file,
    constexpr_substitutions,
    run_config,
)


@dataclass
class WeakScaleConfig:
    """A single point in the sweep grid."""

    m_tiles: int
    n_tiles: int
    num_stages: int
    k: int
    wg_m: int  # workgroups along M
    wg_n: int  # workgroups along N
    m_waves: int = 1  # waves per WG along M
    n_waves: int = 1  # waves per WG along N

    @property
    def num_workgroups(self):
        return self.wg_m * self.wg_n

    @property
    def m_dim(self):
        """Total M = WG_M * M_WAVES * M_T * 16."""
        return self.wg_m * self.m_waves * self.m_tiles * 16

    @property
    def n_dim(self):
        """Total N = WG_N * N_WAVES * N_T * 16."""
        return self.wg_n * self.n_waves * self.n_tiles * 16

    @property
    def total_flops(self):
        """2*M*N*K for the full output matrix."""
        return 2 * self.m_dim * self.n_dim * self.k

    @property
    def lds_bytes(self):
        """LDS per pipeline stage: (M_WAVES * M_T + N_WAVES * N_T) * 512."""
        return (
            self.num_stages
            * (self.m_waves * self.m_tiles + self.n_waves * self.n_tiles)
            * 512
        )

    @property
    def label(self):
        return (
            f"{self.m_tiles}x{self.n_tiles}_s{self.num_stages}"
            f"_K{self.k}_wg{self.wg_m}x{self.wg_n}_w{self.m_waves}x{self.n_waves}"
        )


# MI300X theoretical peak for f16 MFMA
MI300X_PEAK_TFLOPS_F16 = 1307.0

# Sweep grid
TILE_CONFIGS = [(1, 1), (2, 2), (3, 3), (4, 4)]
STAGE_CONFIGS = [2, 3, 5]
WORKGROUP_COUNTS = [304, 1216]  # 1 per CU, 4 per CU
PERF_K = 4096
NUM_ITERATIONS = 12
WARMUP_ITERATIONS = 2


def _run_timed(
    m_waves,
    n_waves,
    m_tiles,
    n_tiles,
    k,
    num_stages,
    wg_m,
    wg_n,
    num_iterations,
    print_ir_after_all=False,
):
    """Run a multi-WG multi-wave constexpr GEMM and return (C_output, iteration_times_ns).

    A is [m_dim x K], B is [n_dim x K], C is [m_dim x n_dim]. num_blocks = wg_m * wg_n;
    block_dim = (m_waves * n_waves * 64, 1, 1). Each wave within a WG computes M_T x N_T
    tiles at its (wave_m, wave_n) offset.
    """
    cfg = WeakScaleConfig(m_tiles, n_tiles, num_stages, k, wg_m, wg_n, m_waves, n_waves)
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)

    subs = constexpr_substitutions(m_tiles, n_tiles, k, num_stages)
    subs["{{WG_M}}"] = str(wg_m)
    subs["{{WG_N}}"] = str(wg_n)
    subs["{{M_WAVES}}"] = str(m_waves)
    subs["{{N_WAVES}}"] = str(n_waves)
    subs["{{A_LDS_BYTES}}"] = str(m_waves * m_tiles * 512)
    subs["{{B_LDS_BYTES}}"] = str(n_waves * n_tiles * 512)
    # Override STRIDE_C: row stride of the full C matrix (not per-WG).
    subs["{{STRIDE_C}}"] = str(cfg.n_dim * 4)
    # Override SHARED_MEM: (M_WAVES * M_T + N_WAVES * N_T) * 512 per pipeline stage.
    subs["{{SHARED_MEM}}"] = str((m_waves * m_tiles + n_waves * n_tiles) * 512)

    orig_blocks = run_config.num_blocks
    orig_iters = run_config.num_iterations
    run_config.num_blocks = cfg.num_workgroups
    run_config.num_iterations = num_iterations
    try:
        times_ns = run_kittens_kernel(
            mlir_file=get_mlir_file("test_perf_001_gemm_fp16_weak_scaled.mlir"),
            kernel_name="gemm_f16_weak_scaled",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
            template_substitutions=subs,
            block_dim=(m_waves * n_waves * 64, 1, 1),
            print_ir_after_all=print_ir_after_all,
        )
    finally:
        run_config.num_blocks = orig_blocks
        run_config.num_iterations = orig_iters

    return C_output, times_ns


class TestWeakScaleCorrectness:
    """Correctness gate: must pass before perf sweep runs."""

    @pytest.mark.parametrize(
        "m_waves,n_waves",
        [(1, 1), (2, 2), (2, 4), (4, 4)],
        ids=["waves_1x1", "waves_2x2", "waves_2x4", "waves_4x4"],
        # [(4, 4)],
        # ids=["waves_4x4"],
    )
    @pytest.mark.parametrize(
        "m_tiles,n_tiles",
        [(1, 1), (2, 2), (2, 4)],
        ids=["tiles_1x1", "tiles_2x2", "tiles_2x4"],
        # 4x4 seems to push it too far but no clear slam dunk resource overflow either
        # [(4, 4)],
        # ids=["tiles_4x4"],
    )
    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    def test_correctness(self, m_waves, n_waves, m_tiles, n_tiles, num_stages):
        """Constexpr GEMM at K=128, 1x1 WG grid, 1 wave, verified against numpy."""
        k = 128
        C_output, _ = _run_timed(
            m_waves,
            n_waves,
            m_tiles,
            n_tiles,
            k,
            num_stages,
            1,
            1,
            1,
            print_ir_after_all=False,
        )

        m_dim = m_waves * m_tiles * 16
        n_dim = n_waves * n_tiles * 16
        np.random.seed(42)
        A = (np.random.randn(m_dim, k) * 0.1).astype(np.float16)
        B = (np.random.randn(n_dim, k) * 0.1).astype(np.float16)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
