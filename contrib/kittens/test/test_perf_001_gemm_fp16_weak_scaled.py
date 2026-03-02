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
    num_workgroups: int

    @property
    def m_dim(self):
        return self.m_tiles * 16

    @property
    def n_dim(self):
        return self.n_tiles * 16

    @property
    def total_flops(self):
        """2*M*N*K per workgroup, times num_workgroups."""
        return 2 * self.m_dim * self.n_dim * self.k * self.num_workgroups

    @property
    def lds_bytes(self):
        return self.num_stages * (self.m_tiles + self.n_tiles) * 512

    @property
    def label(self):
        return (
            f"{self.m_tiles}x{self.n_tiles}_s{self.num_stages}"
            f"_K{self.k}_wg{self.num_workgroups}"
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


def _run_timed(m_tiles, n_tiles, k, num_stages, num_workgroups, num_iterations):
    """Run a constexpr GEMM and return (C_output, iteration_times_ns)."""
    cfg = WeakScaleConfig(m_tiles, n_tiles, num_stages, k, num_workgroups)
    np.random.seed(42)
    A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
    B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)

    orig_blocks = run_config.num_blocks
    orig_iters = run_config.num_iterations
    run_config.num_blocks = num_workgroups
    run_config.num_iterations = num_iterations
    try:
        times_ns = run_kittens_kernel(
            mlir_file=get_mlir_file("test_perf_001_gemm_fp16_weak_scaled.mlir"),
            kernel_name="gemm_f16_weak_scaled",
            input_args=[A.flatten(), B.flatten()],
            output_args=[C_output],
            pass_pipeline=TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
            template_substitutions=constexpr_substitutions(
                m_tiles, n_tiles, k, num_stages
            ),
        )
    finally:
        run_config.num_blocks = orig_blocks
        run_config.num_iterations = orig_iters

    return C_output, times_ns


class TestWeakScaleCorrectness:
    """Correctness gate: must pass before perf sweep runs."""

    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize(
        "m_tiles,n_tiles",
        [(1, 1), (2, 2), (4, 4)],
        ids=["1x1", "2x2", "4x4"],
    )
    def test_correctness(self, m_tiles, n_tiles, num_stages):
        """Constexpr GEMM at K=128, 1 WG, verified against numpy."""
        k = 128
        C_output, _ = _run_timed(m_tiles, n_tiles, k, num_stages, 1, 1)

        m_dim = m_tiles * 16
        n_dim = n_tiles * 16
        np.random.seed(42)
        A = (np.random.randn(m_dim, k) * 0.1).astype(np.float16)
        B = (np.random.randn(n_dim, k) * 0.1).astype(np.float16)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestWeakScalePerf:
    """Weak-scaling TFLOPS sweep across tile/stage/workgroup configs.

    Runs each config with NUM_ITERATIONS kernel launches, discards WARMUP_ITERATIONS for
    GPU frequency ramp-up, takes min of the rest. Prints a sorted table at the end.
    """

    def test_perf_sweep(self):
        results = []
        failed = []

        configs = [
            WeakScaleConfig(m_t, n_t, stages, PERF_K, wg)
            for m_t, n_t in TILE_CONFIGS
            for stages in STAGE_CONFIGS
            for wg in WORKGROUP_COUNTS
        ]

        for i, cfg in enumerate(configs):
            tag = f"[{i + 1}/{len(configs)}] {cfg.label}"
            try:
                _, times_ns = _run_timed(
                    cfg.m_tiles,
                    cfg.n_tiles,
                    cfg.k,
                    cfg.num_stages,
                    cfg.num_workgroups,
                    NUM_ITERATIONS,
                )
            except Exception as e:
                failed.append((cfg.label, str(e)))
                print(f"  FAIL  {tag}: {e}")
                continue

            measured = times_ns[WARMUP_ITERATIONS:]
            min_ns = min(measured)
            min_ms = min_ns / 1e6
            tflops = (
                cfg.total_flops / min_ns * 1e-3
            )  # flops / ns = GFLOPS, / 1e3 = TFLOPS
            pct_peak = tflops / MI300X_PEAK_TFLOPS_F16 * 100

            results.append((cfg, min_ms, tflops, pct_peak))
            print(
                f"  OK    {tag}: {min_ms:.2f} ms  {tflops:.1f} TFLOPS  ({pct_peak:.1f}%)"
            )

        # Sorted table
        results.sort(key=lambda r: r[2], reverse=True)
        hdr = f"{'Config':<35} | {'Time ms':>8} | {'TFLOPS':>8} | {'% Peak':>7} | {'LDS':>6}"
        sep = "-" * len(hdr)
        print(f"\n{hdr}\n{sep}")
        for cfg, ms, tflops, pct in results:
            lds_kb = cfg.lds_bytes / 1024
            print(
                f"{cfg.label:<35} | {ms:>8.2f} | {tflops:>8.1f} | {pct:>6.1f}% | {lds_kb:>4.0f}KB"
            )

        if failed:
            print(f"\nFailed configs ({len(failed)}):")
            for label, err in failed:
                print(f"  {label}: {err}")

        # Sanity check: best config should achieve >20% peak
        assert results, "No configs succeeded"
        best_pct = results[0][3]
        assert best_pct > 20.0, (
            f"Best config only achieved {best_pct:.1f}% of peak. "
            f"Likely a measurement error -- investigate before trusting."
        )
