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

from aster.pass_pipelines import (
    TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
    make_constexpr_pipelining_pass_pipeline,
)

from kittens_helpers import (
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
    constexpr_substitutions_16x32,
    MCPU,
    LDS_SIZE,
    WAVEFRONT_SIZE,
)

# Keyed by (a_path, load_type). a_path: "lds" or "direct". load_type: "flat" or "buffer".
KERNEL_NAMES = {
    "lds": "gemm_f16_weak_scaled",
    "direct": "gemm_f16_direct_a",
}
MLIR_FILES = {
    ("lds", "flat"): "test_perf_001_gemm_fp16_weak_scaled.mlir",
    ("lds", "buffer"): "test_perf_001_gemm_fp16_weak_scaled.mlir",
    ("direct", "flat"): "test_perf_002_gemm_fp16_direct_a.mlir",
}
# Both flat and buffer use the same helpers: after PR #418 the _buf helpers were
# unified into the flat helpers file via !aster_utils.any type-erasure.
K_LOOP_HELPERS_FILES = {
    "flat": "gemm_16x32_f16_k_loop_helpers.mlir",
    "buffer": "gemm_16x32_f16_k_loop_helpers.mlir",
}


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
    load_type: str = "flat"  # "flat" or "buffer"
    a_path: str = "lds"  # "lds" or "direct" (bpermute, A bypasses LDS)
    num_wg_per_cu: int = 1  # target workgroups per CU for register budget
    lcm_unroll: bool = True  # LCM-based kernel loop unrolling
    unroll_factor_multiplier: int = 1  # extra unroll on top of LCM
    epilogue_peeling: bool = True  # fully unroll cleanup loop after LCM unrolling
    _label_suffix: str = ""

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
    def use_buffer(self):
        return self.load_type == "buffer"

    @property
    def direct_a(self):
        return self.a_path == "direct"

    @property
    def kernel_name(self):
        return KERNEL_NAMES[self.a_path]

    @property
    def estimated_agprs(self):
        """Coarse AGPR estimate: 4 AGPRs per 16x16 output tile per wave."""
        return self.m_tiles * self.n_tiles * 4

    @property
    def estimated_vgprs(self):
        """Coarse VGPR estimate: pipeline buffers + overhead.

        Each transfer tile is dwordx4 (4 VGPRs). Per stage we load
        m_tiles*k_tiles A tiles and n_tiles*k_tiles B tiles per wave.
        Add overhead for: LDS read buffers (~same as load buffers for one
        stage), loop counters, addresses, base pointers, bpermute scratch.
        Calibrated against actual compiler output (e.g. 242 VGPRs for
        m_tiles=4 n_tiles=6 k_tiles=2 stages=2).
        """
        a_bufs = self.m_tiles * self.k_tiles * self.num_stages * 4
        b_bufs = self.n_tiles * self.k_tiles * self.num_stages * 4
        # LDS read buffers: one stage worth of tiles (direct-A skips LDS for A)
        a_lds_read = 0 if self.direct_a else self.m_tiles * self.k_tiles * 4
        lds_read = a_lds_read + self.n_tiles * self.k_tiles * 4
        # Fixed overhead for addresses/loop vars.
        # direct-A adds bpermute scratch VGPRs.
        structural = a_bufs + b_bufs + lds_read
        overhead = 30 if self.direct_a else 10
        return structural + overhead

    @property
    def lds_bytes(self):
        """LDS per pipeline stage.

        Direct-A uses LDS only for B.
        """
        a_tiles = 0 if self.direct_a else self.m_tiles_wg * self.k_tiles
        b_tiles = self.n_tiles_wg * self.k_tiles
        return self.num_stages * (a_tiles + b_tiles) * 1024

    @property
    def label(self):
        tile_str = f"_twg{self.m_tiles_wg}x{self.n_tiles_wg}x{self.k_tiles}"
        occ = f"_occ{self.num_wg_per_cu}" if self.num_wg_per_cu > 1 else ""
        lcm = "" if self.lcm_unroll else "_nolcm"
        um = (
            f"_um{self.unroll_factor_multiplier}"
            if self.unroll_factor_multiplier > 1
            else ""
        )
        peel = "" if self.epilogue_peeling else "_nopeel"
        return (
            f"m{self.m_dim}xn{self.n_dim}xk{self.k}"
            f"_wg{self.m_wg}x{self.n_wg}_w{self.m_waves}x{self.n_waves}"
            f"{tile_str}_s{self.num_stages}{occ}{lcm}{um}{peel}{self._label_suffix}"
        )


def _load_k_loop_helpers(load_type="flat", a_path="lds"):
    """Read the shared K-loop helper functions MLIR fragment."""
    helpers_path = get_mlir_file(K_LOOP_HELPERS_FILES[load_type])
    with open(helpers_path) as f:
        helpers = f.read()
    if a_path == "direct":
        direct_path = get_mlir_file("gemm_16x32_f16_k_loop_helpers_direct_a.mlir")
        with open(direct_path) as f:
            helpers += "\n" + f.read()
    return helpers


def _make_substitutions(cfg):
    """Build template substitutions dict for a WeakScaleConfig."""
    subs = {"{{K_LOOP_HELPERS}}": _load_k_loop_helpers(cfg.load_type, cfg.a_path)}
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


def compile_gemm(
    cfg,
    output_hsaco_path,
    print_ir_after_all=False,
    num_vgprs=256,
    num_agprs=256,
    unroll_factor_multiplier=1,
    epilogue_peeling=True,
):
    """Compile a GEMM config to HSACO.

    Returns (hsaco_path, asm_str). Handles a_path (lds/direct) and load_type
    (flat/buffer) via cfg fields.
    """
    from aster import ir, utils
    from aster.testing import compile_mlir_file_to_asm

    subs = _make_substitutions(cfg)

    def preprocess(content):
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    mlir_key = (cfg.a_path, cfg.load_type)
    mlir_file = MLIR_FILES[mlir_key]
    lib_paths = get_kittens_16x16_lds_library_paths(
        use_buffer=cfg.use_buffer, direct_a=cfg.direct_a
    )

    lcm_unroll = getattr(cfg, "lcm_unroll", True)
    if (
        num_vgprs != 256
        or num_agprs != 256
        or unroll_factor_multiplier > 1
        or not lcm_unroll
        or not epilogue_peeling
    ):
        pipeline = make_constexpr_pipelining_pass_pipeline(
            lcm_unroll=lcm_unroll,
            num_vgprs=num_vgprs,
            num_agprs=num_agprs,
            unroll_factor_multiplier=unroll_factor_multiplier,
            epilogue_peeling=epilogue_peeling,
        )
    else:
        pipeline = TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE

    ctx = ir.Context()
    ctx.__enter__()
    try:
        asm, _ = compile_mlir_file_to_asm(
            get_mlir_file(mlir_file),
            cfg.kernel_name,
            pipeline,
            ctx,
            library_paths=lib_paths,
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


def execute_gemm_hsaco(cfg, hsaco_path, num_iterations, A, B, skip_gpu_check=False):
    """Execute a pre-compiled HSACO for a GEMM config.

    Returns (C_output, times_ns).

    Uses aster.hip (MLIR/LLVM-free) for rocprofv3 compatibility. Set skip_gpu_check=True
    when running under rocprofv3 (rocminfo hangs because rocprofv3 intercepts child
    processes).

    Skips (pytest.skip) if target GPU unavailable.
    """
    from aster.hip import system_has_gpu, execute_hsaco

    if not skip_gpu_check and not system_has_gpu(MCPU):
        pytest.skip(f"GPU {MCPU} not available, skip execution")

    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)

    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=cfg.kernel_name,
        input_arrays=[A.flatten(), B.flatten()],
        output_arrays=[C_output],
        grid_dim=(cfg.num_workgroups, 1, 1),
        block_dim=(cfg.num_threads, 1, 1),
        num_iterations=num_iterations,
    )
    return C_output, times_ns


class TestWeakScaleCorrectness:
    """Correctness gate: must pass before perf sweep runs."""

    # Problem sizes > 4000 in each dimension.
    # M = m_wg * m_tiles_wg * 16, N = n_wg * n_tiles_wg * 16.
    # n_wg must be power of 2 (delinearize from 1-D block ID).
    @pytest.mark.parametrize(
        "m_wg,n_wg,m_tiles_wg,n_tiles_wg,m_waves,n_waves",
        [
            # 2048x2048: 32 WG x 4 tiles/WG x 16 = 2048
            (32, 32, 4, 4, 2, 2),  # 2x2 waves, 2x2 tiles/wave
            (32, 32, 4, 4, 4, 4),  # 4x4 waves, 1x1 tiles/wave
            (16, 16, 8, 8, 4, 4),  # 4x4 waves, 2x2 tiles/wave
            (16, 16, 8, 8, 2, 2),  # 2x2 waves, 4x4 tiles/wave
            # 2048x4096: rectangular
            (32, 64, 4, 4, 2, 2),
            (32, 64, 4, 4, 4, 4),
        ],
        ids=[
            "2kx2k_wg32_twg4_w2x2",
            "2kx2k_wg32_twg4_w4x4",
            "2kx2k_wg16_twg8_w4x4",
            "2kx2k_wg16_twg8_w2x2",
            "2kx4k_wg32x64_twg4_w2x2",
            "2kx4k_wg32x64_twg4_w4x4",
        ],
    )
    @pytest.mark.parametrize("num_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("load_type", ["flat", "buffer"], ids=["flat", "buffer"])
    @pytest.mark.parametrize("a_path", ["lds", "direct"], ids=["lds", "direct"])
    def test_correctness(
        self,
        m_wg,
        n_wg,
        m_tiles_wg,
        n_tiles_wg,
        m_waves,
        n_waves,
        num_stages,
        load_type,
        a_path,
    ):
        """Constexpr GEMM verified against numpy."""
        if (a_path, load_type) not in MLIR_FILES:
            pytest.skip(f"({a_path}, {load_type}) not yet implemented")
        k = 128
        k_tiles = 1
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
            load_type=load_type,
            a_path=a_path,
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
            compile_gemm(cfg, tmp.name)
            C_output, _ = execute_gemm_hsaco(cfg, tmp.name, 1, A, B)

        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a single weak-scaled GEMM config",
    )
    parser.add_argument(
        "--m-wg", type=int, default=1, help="Workgroups along M (default: 1)"
    )
    parser.add_argument(
        "--n-wg", type=int, default=1, help="Workgroups along N (default: 1)"
    )
    parser.add_argument(
        "--m-waves", type=int, default=1, help="Waves per WG along M (default: 1)"
    )
    parser.add_argument(
        "--n-waves", type=int, default=1, help="Waves per WG along N (default: 1)"
    )
    parser.add_argument(
        "--m-tiles-wg",
        type=int,
        default=1,
        help="Tiles per workgroup along M (default: 1)",
    )
    parser.add_argument(
        "--n-tiles-wg",
        type=int,
        default=1,
        help="Tiles per workgroup along N (default: 1)",
    )
    parser.add_argument(
        "--k-tiles", type=int, default=1, help="Tiles per wave along K (default: 1)"
    )
    parser.add_argument(
        "--stages", type=int, default=2, help="Pipeline stages (default: 2)"
    )
    parser.add_argument(
        "--k-scaling-factor",
        type=int,
        default=4,
        help="K scaling factor (K = factor * k_tiles * 32, default: 4)",
    )
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
    parser.add_argument(
        "--print-asm",
        action="store_true",
        help="Print generated assembly to stdout",
    )
    buf_group = parser.add_mutually_exclusive_group()
    buf_group.add_argument(
        "--use-buffer",
        action="store_true",
        help="Use buffer_load/buffer_store (MUBUF) instead of global_load/global_store (flat)",
    )
    buf_group.add_argument(
        "--use-flat",
        action="store_true",
        help="Use global_load/global_store (flat) instead of buffer_load/buffer_store",
    )
    parser.add_argument(
        "--direct-a",
        action="store_true",
        help="A operand via bpermute (LDS bypass) instead of LDS",
    )
    a = parser.parse_args()
    load_type = "buffer" if a.use_buffer else "flat"
    a_path = "direct" if a.direct_a else "lds"
    k = a.k_scaling_factor * a.k_tiles * 32

    cfg = WeakScaleConfig(
        a.m_wg,
        a.n_wg,
        a.m_waves,
        a.n_waves,
        a.m_tiles_wg,
        a.n_tiles_wg,
        a.k_tiles,
        a.stages,
        k,
        load_type=load_type,
        a_path=a_path,
    )

    from aster.hip import parse_asm_kernel_resources

    a_mode = f" direct-A" if cfg.direct_a else ""

    print(f"Config: {cfg.label} ({cfg.load_type}{a_mode})")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    print(f"  per-wave tiles: {cfg.m_tiles}x{cfg.n_tiles}, LDS={cfg.lds_bytes}")

    if a.compile_only:
        if not a.hsaco:
            print("Error: --compile-only requires --hsaco <output_path>")
            raise SystemExit(1)
        _, asm = compile_gemm(cfg, a.hsaco, print_ir_after_all=a.print_ir_after_all)
        resources = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name)
        res = resources.get(cfg.kernel_name)
        if res:
            print(f"  resources: {res}")
        if a.print_asm:
            print(asm)
        print(f"  Compiled: {a.hsaco}")
    else:
        np.random.seed(42)
        A = (np.random.randn(cfg.m_dim, cfg.k) * 0.1).astype(np.float16)
        B = (np.random.randn(cfg.n_dim, cfg.k) * 0.1).astype(np.float16)

        if a.hsaco:
            C_output, times_ns = execute_gemm_hsaco(
                cfg, a.hsaco, a.iterations, A, B, skip_gpu_check=True
            )
        else:
            with tempfile.NamedTemporaryFile(suffix=".hsaco", delete=True) as tmp:
                _, asm = compile_gemm(
                    cfg, tmp.name, print_ir_after_all=a.print_ir_after_all
                )
                resources = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name)
                res = resources.get(cfg.kernel_name)
                if res:
                    print(f"  resources: {res}")
                if a.print_asm:
                    print(asm)
                C_output, times_ns = execute_gemm_hsaco(
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
