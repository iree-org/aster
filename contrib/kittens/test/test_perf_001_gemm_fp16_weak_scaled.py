"""Test: Correctness gate for weak-scaled constexpr GEMM.

Verifies multi-WG, multi-wave, multi-tile GEMM at K=128 against numpy reference.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import tempfile

from aster.pass_pipelines import TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    get_mlir_file,
    get_kittens_library_paths,
    constexpr_substitutions,
    MCPU,
    WAVEFRONT_SIZE,
)

KERNEL_NAME = "gemm_f16_weak_scaled"
MLIR_FILE = "test_perf_001_gemm_fp16_weak_scaled.mlir"


@dataclass
class WeakScaleConfig:
    """A single point in the sweep grid."""

    m_wg: int  # workgroups along M
    n_wg: int  # workgroups along N
    m_waves: int  # waves per WG along M
    n_waves: int  # waves per WG along N
    m_tiles: int
    n_tiles: int
    k_tiles: int
    num_stages: int
    k: int

    @property
    def num_workgroups(self):
        return self.m_wg * self.n_wg

    @property
    def num_waves(self):
        return self.m_waves * self.n_waves

    @property
    def num_threads(self):
        return self.num_waves * 64

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

    # Note: this is a rather gross approximation, A and B could be pipelined differently.
    # TODO: let the resource allocation pass figure this out.
    @property
    def lds_bytes(self):
        """LDS per pipeline stage: num_waves * (M_T * K_T + K_T * N_T) * (16 * 16 * 2)."""
        return (
            self.num_stages
            * self.num_waves
            * (self.m_tiles * self.k_tiles + self.k_tiles * self.n_tiles)
            * 512
        )

    @property
    def label(self):
        tile_str = f"_t{self.m_tiles}x{self.n_tiles}x{self.k_tiles}"
        return (
            f"m{self.m_dim}xn{self.n_dim}xk{self.k}"
            f"_wg{self.m_wg}x{self.n_wg}_w{self.m_waves}x{self.n_waves}"
            f"{tile_str}_s{self.num_stages}"
        )


def _make_substitutions(cfg):
    """Build template substitutions dict for a WeakScaleConfig."""
    subs = constexpr_substitutions(cfg.m_tiles, cfg.n_tiles, cfg.k, cfg.num_stages)
    subs["{{M_WG}}"] = str(cfg.m_wg)
    subs["{{N_WG}}"] = str(cfg.n_wg)
    subs["{{M_WAVES}}"] = str(cfg.m_waves)
    subs["{{N_WAVES}}"] = str(cfg.n_waves)
    subs["{{A_LDS_BYTES}}"] = str(cfg.m_waves * cfg.m_tiles * cfg.k_tiles * 512)
    subs["{{B_LDS_BYTES}}"] = str(cfg.n_waves * cfg.n_tiles * cfg.k_tiles * 512)
    subs["{{STRIDE_C}}"] = str(cfg.n_dim * 4)  # f32 = 4 bytes
    # shared_memory_size must be 0: all LDS is managed by alloc_lds/dealloc_lds.
    # The LDS allocator uses shared_memory_size as startPos, so any non-zero value
    # wastes that many bytes of LDS (offsets start after the pre-reserved region).
    subs["{{SHARED_MEM}}"] = "0"
    subs["{{NUM_THREADS}}"] = str(cfg.num_threads)
    subs["{{NUM_BLOCKS}}"] = str(cfg.num_workgroups)
    subs["{{K_T}}"] = str(cfg.k_tiles)
    subs["{{A_TILES_PER_SLICE}}"] = str(cfg.m_waves * cfg.m_tiles)
    subs["{{B_TILES_PER_SLICE}}"] = str(cfg.n_waves * cfg.n_tiles)
    return subs


def compile_weak_scaled_gemm(cfg, output_hsaco_path):
    """Compile a weak-scaled GEMM config to HSACO.

    Returns (hsaco_path, asm_str). This is the CPU-only compilation step (no GPU
    needed). Safe to run in parallel across configs.
    """
    from aster import ir, utils
    from aster.testing import compile_mlir_file_to_asm

    subs = _make_substitutions(cfg)

    def preprocess(content):
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    ctx = ir.Context()
    ctx.__enter__()
    try:
        asm, _ = compile_mlir_file_to_asm(
            get_mlir_file(MLIR_FILE),
            KERNEL_NAME,
            TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
            ctx,
            library_paths=get_kittens_library_paths(),
            preprocess=preprocess,
        )
        path = utils.assemble_to_hsaco(
            asm,
            target=MCPU,
            wavefront_size=WAVEFRONT_SIZE,
            output_path=output_hsaco_path,
        )
        assert path is not None, "assemble_to_hsaco returned None"
        return path, asm
    finally:
        ctx.__exit__(None, None, None)


def execute_weak_scaled_hsaco(
    cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False
):
    """Execute a pre-compiled HSACO for a weak-scaled GEMM config.

    Returns (C_output, times_ns). Must run sequentially on GPU. Skips (pytest.skip) if
    the target GPU is not available.

    Uses aster.hip (MLIR/LLVM-free) for rocprofv3 compatibility. Set skip_gpu_check=True
    when running under rocprofv3 (rocminfo hangs because rocprofv3 intercepts child
    processes).
    """
    from aster.hip import system_has_gpu, execute_hsaco

    if not skip_gpu_check and not system_has_gpu(MCPU):
        pytest.skip(f"GPU {MCPU} not available, skip execution")

    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)

    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=KERNEL_NAME,
        input_arrays=[A.flatten(), B.flatten()],
        output_arrays=[C_output],
        grid_dim=(cfg.num_workgroups, 1, 1),
        block_dim=(cfg.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


class TestWeakScaleCorrectness:
    """Correctness gate: must pass before perf sweep runs."""

    @pytest.mark.parametrize(
        "m_wg,n_wg",
        # note: most minor dimension must be power of 2 to delinearize from 1-D
        # as aster does not yet support general divisions.
        # alternatively the kernel could use block_id x and block_id y
        [(1, 1), (19, 4)],
        ids=["wg1x1", "wg4x19"],
    )
    @pytest.mark.parametrize(
        "m_waves,n_waves",
        [(1, 1), (2, 2), (2, 4)],
        ids=["waves_1x1", "waves_2x2", "waves_2x4"],
    )
    @pytest.mark.parametrize(
        "m_tiles,n_tiles,k_tiles",
        [(1, 1, 1), (2, 2, 1), (1, 4, 2), (2, 1, 3), (1, 2, 4)],
        ids=["tiles_1x1x1", "tiles_2x2x1", "tiles_1x4x2", "tiles_2x1x3", "tiles_1x2x4"],
    )
    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    def test_correctness(
        self, m_wg, n_wg, m_waves, n_waves, m_tiles, n_tiles, k_tiles, num_stages
    ):
        """Constexpr GEMM verified against numpy.

        K = 4 * k_tiles * 16.
        """
        k = 4 * k_tiles * 16
        cfg = WeakScaleConfig(
            m_wg,
            n_wg,
            m_waves,
            n_waves,
            m_tiles,
            n_tiles,
            k_tiles,
            num_stages,
            k,
        )
        np.random.seed(42)
        A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            compile_weak_scaled_gemm(cfg, tmp.name)  # asm unused in correctness tests
            C_output, _ = execute_weak_scaled_hsaco(cfg, tmp.name, 1, A, B)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
