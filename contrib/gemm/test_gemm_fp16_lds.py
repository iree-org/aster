"""Pytest suite for the multi-CU fp16 GEMM kernel."""

import numpy as np
import pytest

from gemm_fp16_lds import GEMMConfig, run_gemm


class TestGEMMFP16LDS:
    """Test generalized GEMM: C = A @ B^T with cooperative loading and CTA swizzle."""

    @pytest.mark.parametrize(
        "m,n,k,m_tile,n_tile,k_tile,num_waves,swizzle,num_m_waves",
        [
            # Single-block cases (one tile = full matrix, swizzle=1).
            # K_TILE >= 64 (K_T >= 2) is the minimum recommended granularity.
            (16, 16, 64, 16, 16, 64, 1, 1, 0),  # 1 block, 1 wave, 1x1 MFMA tile, K_T=2
            (32, 32, 64, 32, 32, 64, 1, 1, 0),  # 1 block, 1 wave, 2x2 MFMA tiles, K_T=2
            (32, 32, 64, 32, 32, 64, 2, 1, 0),  # 1 block, 2 waves, cooperative B load
            (16, 16, 128, 16, 16, 64, 1, 1, 0),  # 1 block, 1 wave, K=128, K_T=2
            (32, 32, 128, 32, 32, 128, 1, 1, 0),  # 1 block, 1 wave, K_TILE=128 (K_T=4)
            # Multi-block cases.
            (64, 32, 64, 16, 16, 64, 1, 1, 0),  # 8 blocks, row-major (swizzle=1)
            (64, 32, 64, 16, 16, 64, 1, 4, 0),  # 8 blocks, CTA swizzle=4
            (
                64,
                64,
                64,
                32,
                32,
                64,
                2,
                2,
                0,
            ),  # 4 blocks, 2 waves, cooperative B + swizzle
            # 2D wave partition: num_m_waves x num_n_waves grid within the workgroup.
            (64, 64, 64, 64, 64, 64, 4, 1, 2),  # 1 block, 4 waves, 2x2 compute grid
        ],
        ids=[
            "1b-1w-1x1-k64-kt64",
            "1b-1w-2x2-k64-kt64",
            "1b-2w-2x2-k64-kt64-coop",
            "1b-1w-1x1-k128-kt64",
            "1b-1w-2x2-k128-kt128",
            "8b-1w-row-major",
            "8b-1w-swizzle4",
            "4b-2w-coop-swizzle2",
            "1b-4w-2x2-wave-grid",
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
        swizzle: int,
        num_m_waves: int,
    ):
        """GEMM output must match numpy reference within f16 tolerance."""
        cfg = GEMMConfig(
            m, n, k, m_tile, n_tile, k_tile, num_waves, swizzle, num_m_waves
        )

        np.random.seed(42)
        A = (np.random.randn(cfg.m, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n, cfg.k) * 0.1).astype(np.float16)

        C_output = run_gemm(cfg, A, B)
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()

        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
