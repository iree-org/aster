"""Test: Correctness gate for weak-scaled constexpr GEMM (16x16x16 MFMA + dwordx4).

Verifies multi-WG, multi-wave, multi-tile GEMM at K=128 against numpy reference.
Uses v_mfma_f32_16x16x16_f16: 16x16 output tiles, K=16 per MFMA.
Global loads use dwordx4 (16x32 transfer tiles): 2x bandwidth vs dwordx2.

Tiles are specified per-workgroup (m_tiles_wg, n_tiles_wg). Per-wave tile counts
are derived: m_tiles = m_tiles_wg // m_waves.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import tempfile

from aster.pass_pipelines import TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE

from kittens_helpers import (
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
    constexpr_substitutions_16x32,
    MCPU,
    LDS_SIZE,
    WAVEFRONT_SIZE,
)

KERNEL_NAME = "gemm_f16_weak_scaled"
MLIR_FILE = "test_perf_001_gemm_fp16_weak_scaled.mlir"
K_LOOP_HELPERS_FILE = "gemm_f16_16x32_k_loop_helpers.mlir"


@dataclass
class WeakScaleConfig:
    """A single point in the sweep grid.

    Tiles are specified per-workgroup (m_tiles_wg, n_tiles_wg) and waves are
    independent.  Per-wave tile counts are derived: m_tiles = m_tiles_wg // m_waves.
    Constraint: m_tiles_wg % m_waves == 0 and n_tiles_wg % n_waves == 0.
    """

    m_wg: int  # workgroups along M
    n_wg: int  # workgroups along N
    m_waves: int  # waves per WG along M
    n_waves: int  # waves per WG along N
    m_tiles_wg: int  # tiles per workgroup along M
    n_tiles_wg: int  # tiles per workgroup along N
    k_tiles: int
    num_stages: int
    k: int

    def __post_init__(self):
        assert (
            self.m_tiles_wg % self.m_waves == 0
        ), f"m_tiles_wg={self.m_tiles_wg} not divisible by m_waves={self.m_waves}"
        assert (
            self.n_tiles_wg % self.n_waves == 0
        ), f"n_tiles_wg={self.n_tiles_wg} not divisible by n_waves={self.n_waves}"

    @property
    def m_tiles(self):
        """Per-wave tiles along M (derived from m_tiles_wg // m_waves)."""
        return self.m_tiles_wg // self.m_waves

    @property
    def n_tiles(self):
        """Per-wave tiles along N (derived from n_tiles_wg // n_waves)."""
        return self.n_tiles_wg // self.n_waves

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
        """Total M = M_WG * M_TILES_WG * 16."""
        return self.m_wg * self.m_tiles_wg * 16

    @property
    def n_dim(self):
        """Total N = N_WG * N_TILES_WG * 16."""
        return self.n_wg * self.n_tiles_wg * 16

    @property
    def total_flops(self):
        """2*M*N*K for the full output matrix."""
        return 2 * self.m_dim * self.n_dim * self.k

    @property
    def lds_bytes(self):
        """LDS per pipeline stage: num_stages * (m_tiles_wg * k_tiles + n_tiles_wg * k_tiles) * 1024."""
        return (
            self.num_stages
            * (self.m_tiles_wg * self.k_tiles + self.n_tiles_wg * self.k_tiles)
            * 1024
        )

    @property
    def label(self):
        tile_str = f"_twg{self.m_tiles_wg}x{self.n_tiles_wg}x{self.k_tiles}"
        return (
            f"m{self.m_dim}xn{self.n_dim}xk{self.k}"
            f"_wg{self.m_wg}x{self.n_wg}_w{self.m_waves}x{self.n_waves}"
            f"{tile_str}_s{self.num_stages}"
        )


def _load_k_loop_helpers():
    """Read the shared K-loop helper functions MLIR fragment."""
    helpers_path = get_mlir_file(K_LOOP_HELPERS_FILE)
    with open(helpers_path) as f:
        return f.read()


def _make_substitutions(cfg):
    """Build template substitutions dict for a WeakScaleConfig."""
    subs = {"{{K_LOOP_HELPERS}}": _load_k_loop_helpers()}
    subs.update(
        constexpr_substitutions_16x32(cfg.m_tiles, cfg.n_tiles, cfg.k, cfg.num_stages)
    )
    subs["{{M_WG}}"] = str(cfg.m_wg)
    subs["{{N_WG}}"] = str(cfg.n_wg)
    subs["{{M_WAVES}}"] = str(cfg.m_waves)
    subs["{{N_WAVES}}"] = str(cfg.n_waves)
    subs["{{M_TILES_WG}}"] = str(cfg.m_tiles_wg)
    subs["{{N_TILES_WG}}"] = str(cfg.n_tiles_wg)
    subs["{{A_LDS_BYTES}}"] = str(cfg.m_tiles_wg * cfg.k_tiles * 1024)
    subs["{{B_LDS_BYTES}}"] = str(cfg.n_tiles_wg * cfg.k_tiles * 1024)
    subs["{{STRIDE_C}}"] = str(cfg.n_dim * 4)  # f32 = 4 bytes
    subs["{{SHARED_MEM}}"] = "0"
    subs["{{NUM_THREADS}}"] = str(cfg.num_threads)
    subs["{{NUM_BLOCKS}}"] = str(cfg.num_workgroups)
    subs["{{K_T}}"] = str(cfg.k_tiles)
    subs["{{A_TILES_PER_SLICE}}"] = str(cfg.m_tiles_wg)
    subs["{{B_TILES_PER_SLICE}}"] = str(cfg.n_tiles_wg)
    return subs


def compile_weak_scaled_gemm(cfg, output_hsaco_path, print_ir_after_all=False):
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
            library_paths=get_kittens_16x16_lds_library_paths(),
            preprocess=preprocess,
            print_ir_after_all=print_ir_after_all,
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
        [(1, 1), (2, 2), (2, 4), (4, 4)],
        ids=["waves_1x1", "waves_2x2", "waves_2x4", "waves_4x4"],
    )
    @pytest.mark.parametrize(
        "m_tiles_wg,n_tiles_wg",
        [(2, 2), (4, 4), (4, 8)],
        ids=["twg_2x2", "twg_4x4", "twg_4x8"],
    )
    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    def test_correctness(
        self, m_wg, n_wg, m_waves, n_waves, m_tiles_wg, n_tiles_wg, num_stages
    ):
        """Constexpr GEMM verified against numpy."""
        k = 128
        k_tiles = 1
        if m_tiles_wg % m_waves != 0 or n_tiles_wg % n_waves != 0:
            pytest.skip(
                f"twg {m_tiles_wg}x{n_tiles_wg} not divisible by waves {m_waves}x{n_waves}"
            )
        cfg = WeakScaleConfig(
            m_wg,
            n_wg,
            m_waves,
            n_waves,
            m_tiles_wg,
            n_tiles_wg,
            k_tiles,
            num_stages,
            k,
        )
        # Per-wave tile product > 16 requires too many registers.
        if cfg.m_tiles * cfg.n_tiles > 16:
            pytest.skip(
                f"per-wave tiles {cfg.m_tiles}x{cfg.n_tiles} product > 16 for {cfg.label}"
            )
        # Avoid unfeasible LDS sizes
        if cfg.lds_bytes >= LDS_SIZE:
            pytest.skip(f"LDS {cfg.lds_bytes} >= {LDS_SIZE}")
        np.random.seed(42)
        A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)
        with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
            compile_weak_scaled_gemm(cfg, tmp.name)
            C_output, _ = execute_weak_scaled_hsaco(cfg, tmp.name, 1, A, B)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single weak-scaled GEMM config",
    )
    parser.add_argument("--m-wg", type=int, required=True, help="Workgroups along M")
    parser.add_argument("--n-wg", type=int, required=True, help="Workgroups along N")
    parser.add_argument(
        "--m-waves", type=int, required=True, help="Waves per WG along M"
    )
    parser.add_argument(
        "--n-waves", type=int, required=True, help="Waves per WG along N"
    )
    parser.add_argument(
        "--m-tiles-wg", type=int, required=True, help="Tiles per workgroup along M"
    )
    parser.add_argument(
        "--n-tiles-wg", type=int, required=True, help="Tiles per workgroup along N"
    )
    parser.add_argument(
        "--k-tiles", type=int, required=True, help="Tiles per wave along K"
    )
    parser.add_argument("--stages", type=int, required=True, help="Pipeline stages")
    parser.add_argument("--k", type=int, required=True, help="K dimension")
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Kernel launches (default: 5)",
    )
    parser.add_argument(
        "--hsaco",
        type=str,
        default=None,
        help="Path to pre-compiled HSACO (skips compilation)",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile HSACO and exit (requires --hsaco for output path)",
    )
    parser.add_argument(
        "--print-ir-after-all",
        action="store_true",
        help="Print IR after each pass",
    )
    a = parser.parse_args()

    cfg = WeakScaleConfig(
        a.m_wg,
        a.n_wg,
        a.m_waves,
        a.n_waves,
        a.m_tiles_wg,
        a.n_tiles_wg,
        a.k_tiles,
        a.stages,
        a.k,
    )

    from aster.hip import parse_asm_kernel_resources

    print(f"Config: {cfg.label}")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    print(f"  per-wave tiles: {cfg.m_tiles}x{cfg.n_tiles}, LDS={cfg.lds_bytes}")

    if a.compile_only:
        if not a.hsaco:
            print("Error: --compile-only requires --hsaco <output_path>")
            raise SystemExit(1)
        _, asm = compile_weak_scaled_gemm(
            cfg, a.hsaco, print_ir_after_all=a.print_ir_after_all
        )
        resources = parse_asm_kernel_resources(asm, kernel_name=KERNEL_NAME)
        res = resources.get(KERNEL_NAME)
        if res:
            print(f"  resources: {res}")
        print(f"  Compiled: {a.hsaco}")
    else:
        np.random.seed(42)
        A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)

        if a.hsaco:
            C_output, times_ns = execute_weak_scaled_hsaco(
                cfg, a.hsaco, a.iterations, A, B, skip_gpu_check=True
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
                _, asm = compile_weak_scaled_gemm(
                    cfg, tmp.name, print_ir_after_all=a.print_ir_after_all
                )
                resources = parse_asm_kernel_resources(asm, kernel_name=KERNEL_NAME)
                res = resources.get(KERNEL_NAME)
                if res:
                    print(f"  resources: {res}")
                C_output, times_ns = execute_weak_scaled_hsaco(
                    cfg, tmp.name, a.iterations, A, B
                )

        measured = times_ns[2:]
        min_ns = min(measured)
        min_ms = min_ns / 1e6
        tflops = cfg.total_flops / min_ns * 1e-3

        print(f"\nAll iterations (ms): {[f'{t/1e6:.2f}' for t in times_ns]}")
        print(f"Measured (post-warmup): {[f'{t/1e6:.2f}' for t in measured]}")
        print(f"Min: {min_ms:.2f} ms  {tflops:.1f} TFLOPS")

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)
        print("PASS")
