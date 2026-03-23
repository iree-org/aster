"""Pytest suite for the multi-CU fp16 GEMM kernel."""

import numpy as np
import pytest

from gemm_fp16_lds import GEMMConfig, run_gemm


class TestGEMMFP16LDS:
    """Test generalized GEMM: C = A @ B^T with cooperative loading and CTA swizzle."""

    @pytest.mark.parametrize(
        "m,n,k,m_tile,n_tile,k_tile,num_waves,num_m_waves,swizzle",
        [
            # Single-block cases (one tile = full matrix, swizzle=1).
            # K_TILE >= 64 (K_T >= 2) is the minimum recommended granularity.
            (16, 16, 64, 16, 16, 64, 1, 1, 1),  # 1 block, 1 wave, 1x1 MFMA tile, K_T=2
            (32, 32, 64, 32, 32, 64, 1, 1, 1),  # 1 block, 1 wave, 2x2 MFMA tiles, K_T=2
            (32, 32, 64, 32, 32, 64, 2, 2, 1),  # 1 block, 2 waves, 2D 2x1 decomp
            (16, 16, 128, 16, 16, 64, 1, 1, 1),  # 1 block, 1 wave, K=128, K_T=2
            (32, 32, 128, 32, 32, 128, 1, 1, 1),  # 1 block, 1 wave, K_TILE=128 (K_T=4)
            # Multi-block cases.
            (64, 32, 64, 16, 16, 64, 1, 1, 1),  # 8 blocks, row-major (swizzle=1)
            (64, 32, 64, 16, 16, 64, 1, 1, 4),  # 8 blocks, CTA swizzle=4
            (64, 64, 64, 32, 32, 64, 2, 2, 2),  # 4 blocks, 2 waves, 2D 2x1 + swizzle
            # 2D wave distribution: 4 waves arranged as 2x2 grid.
            (64, 64, 64, 64, 64, 64, 4, 2, 1),  # 1 block, 4 waves, 2D 2x2 decomp
            # Regression test: config that exposed flat-compute LDS redundancy.
            (128, 64, 64, 128, 64, 32, 4, 2, 1),  # 1 block, 4 waves, 2D 2x2, regression
            # Regression tests: K/k_tile divisible by 4 exposed OOB early-load bug.
            (16, 16, 256, 16, 16, 64, 1, 1, 1),  # 1 block, 1 wave, K=256, 4 iters
            (16, 16, 512, 16, 16, 64, 1, 1, 1),  # 1 block, 1 wave, K=512, 8 iters
            (32, 32, 256, 32, 32, 64, 4, 2, 1),  # 1 block, 4 waves, K=256, 4 iters
        ],
        ids=[
            "1b-1w-1x1-k64-kt64",
            "1b-1w-2x2-k64-kt64",
            "1b-2w-2x2-k64-kt64-2d",
            "1b-1w-1x1-k128-kt64",
            "1b-1w-2x2-k128-kt128",
            "8b-1w-row-major",
            "8b-1w-swizzle4",
            "4b-2w-2d-swizzle2",
            "1b-4w-2d-2x2",
            "1b-4w-2d-regression",
            "1b-1w-1x1-k256-kt64-oob-regression",
            "1b-1w-1x1-k512-kt64-oob-regression",
            "1b-4w-2d-2x2-k256-kt64-oob-regression",
        ],
    )
    def test_gemm_fp16_lds(
        self,
        m: int,
        n: int,
        k: int,
        m_tile: int,
        n_tile: int,
        k_tile: int,
        num_waves: int,
        num_m_waves: int,
        swizzle: int,
    ):
        """GEMM output must match numpy reference within f16 tolerance."""
        cfg = GEMMConfig(
            m, n, k, m_tile, n_tile, k_tile, num_waves, num_m_waves, swizzle
        )

        np.random.seed(42)
        A = (np.random.randn(cfg.m, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n, cfg.k) * 0.1).astype(np.float16)

        C_output = run_gemm(cfg, A, B)
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()

        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
