"""Test: Correctness gate for weak-scaled constexpr GEMM.

Verifies multi-WG, multi-wave, multi-tile GEMM at K=128 against numpy reference.
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

    m_wg: int  # workgroups along M
    n_wg: int  # workgroups along N
    m_waves: int  # waves per WG along M
    n_waves: int  # waves per WG along N
    m_tiles: int
    n_tiles: int
    num_stages: int
    k: int

    @property
    def num_workgroups(self):
        return self.m_wg * self.n_wg

    @property
    def m_dim(self):
        """Total M = M_WG * M_WAVES * M_T * 16."""
        return self.m_wg * self.m_waves * self.m_tiles * 16

    @property
    def n_dim(self):
        """Total N = N_WG * N_WAVES * N_T * 16."""
        return self.n_wg * self.n_waves * self.n_tiles * 16

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
            f"m{self.m_dim}xn{self.n_dim}xk{self.k}"
            f"_wg{self.m_wg}x{self.n_wg}_w{self.m_waves}x{self.n_waves}"
            f"_t{self.m_tiles}x{self.n_tiles}_s{self.num_stages}"
        )


def run_weak_scaled_gemm(
    m_wg,
    n_wg,
    m_waves,
    n_waves,
    m_tiles,
    n_tiles,
    num_stages,
    k,
    num_iterations,
    print_ir_after_all=False,
):
    """Run a multi-WG multi-wave constexpr GEMM and return (C_output, iteration_times_ns).

    A is [m_dim x K], B is [n_dim x K], C is [m_dim x n_dim]. num_blocks = m_wg * n_wg;
    block_dim = (m_waves * n_waves * 64, 1, 1). Each wave within a WG computes M_T x N_T
    tiles at its (wave_m, wave_n) offset.
    """
    cfg = WeakScaleConfig(m_wg, n_wg, m_waves, n_waves, m_tiles, n_tiles, num_stages, k)
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)

    subs = constexpr_substitutions(m_tiles, n_tiles, k, num_stages)
    subs["{{M_WG}}"] = str(m_wg)
    subs["{{N_WG}}"] = str(n_wg)
    subs["{{M_WAVES}}"] = str(m_waves)
    subs["{{N_WAVES}}"] = str(n_waves)
    subs["{{A_LDS_BYTES}}"] = str(m_waves * m_tiles * 512)
    subs["{{B_LDS_BYTES}}"] = str(n_waves * n_tiles * 512)
    # Override STRIDE_C: row stride of the full C matrix (not per-WG).
    subs["{{STRIDE_C}}"] = str(cfg.n_dim * 4)
    # Override SHARED_MEM: (M_WAVES * M_T + N_WAVES * N_T) * 512 per pipeline stage.
    subs["{{SHARED_MEM}}"] = str((m_waves * m_tiles + n_waves * n_tiles) * 512)
    # Override NUM_THREADS/NUM_BLOCKS for multi-wave multi-WG launch config.
    subs["{{NUM_THREADS}}"] = str(m_waves * n_waves * 64)
    subs["{{NUM_BLOCKS}}"] = str(cfg.num_workgroups)

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
        "m_wg,n_wg",
        # note: most minor dimension must be power of 2 to delinearize from 1-D
        # as aster does not yet support general divisions.
        # alternatively the kernel could use block_id x and block_id y
        [(19, 4)],
        ids=["wg4x19"],
    )
    @pytest.mark.parametrize(
        "m_waves,n_waves",
        [(1, 1), (2, 2), (2, 4), (4, 4)],
        ids=["waves_1x1", "waves_2x2", "waves_2x4", "waves_4x4"],
    )
    @pytest.mark.parametrize(
        "m_tiles,n_tiles",
        [(1, 1), (2, 2), (2, 4)],
        ids=["tiles_1x1", "tiles_2x2", "tiles_2x4"],
    )
    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    def test_correctness(
        self, m_wg, n_wg, m_waves, n_waves, m_tiles, n_tiles, num_stages
    ):
        """Constexpr GEMM at K=128, verified against numpy."""
        k = 128
        C_output, _ = run_weak_scaled_gemm(
            m_wg,
            n_wg,
            m_waves,
            n_waves,
            m_tiles,
            n_tiles,
            num_stages,
            k,
            1,
            print_ir_after_all=False,
        )

        cfg = WeakScaleConfig(
            m_wg, n_wg, m_waves, n_waves, m_tiles, n_tiles, num_stages, k
        )
        np.random.seed(42)
        A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
