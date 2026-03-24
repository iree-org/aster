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

from aster.pass_pipelines import make_default_pass_pipeline

from kittens_helpers import (
    get_mlir_file,
    get_kittens_16x16_lds_library_paths,
    constexpr_substitutions_16x32,
    shuffle_weight,
    MCPU,
    LDS_SIZE,
    WAVEFRONT_SIZE,
)

# Keyed by (b_path, load_type). b_path: "lds", "direct_b", or "direct_ab".
# load_type: "flat" or "buffer".
KERNEL_NAMES = {
    "lds": "gemm_f16_weak_scaled",
    "direct_b": "gemm_f16_direct_b",
    "direct_ab": "gemm_f16_direct_ab",
}
MLIR_FILES = {
    ("lds", "flat"): "test_perf_001_gemm_fp16_weak_scaled.mlir",
    ("lds", "buffer"): "test_perf_001_gemm_fp16_weak_scaled.mlir",
    ("direct_b", "flat"): "test_perf_001_gemm_fp16_direct_b.mlir",
    ("direct_b", "buffer"): "test_perf_001_gemm_fp16_direct_b.mlir",
    ("direct_ab", "flat"): "test_perf_001_gemm_fp16_direct_ab.mlir",
    ("direct_ab", "buffer"): "test_perf_001_gemm_fp16_direct_ab.mlir",
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
    a_stages: int
    k: int
    load_type: str = "flat"  # "flat" or "buffer"
    b_path: str = "lds"  # "lds" or "direct_b" (bpermute, B bypasses LDS)
    b_stages: int = 0  # 0 = same as a_stages; >0 = independent B pipeline depth
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
    def k_scaling_factor(self):
        """K = k_scaling_factor * k_tiles * 32."""
        return self.k // (self.k_tiles * 32)

    @property
    def direct_b(self):
        return self.b_path in ("direct_b", "direct_ab")

    @property
    def direct_a(self):
        return self.b_path == "direct_ab"

    def _coop_2d_split(self, spatial_tiles):
        """Split NUM_WAVES into (waves_spatial, waves_k) for 2-D cooperative loading."""
        waves_s = min(spatial_tiles, self.num_waves)
        waves_k = max(1, self.num_waves // waves_s)
        coop_s = -(-spatial_tiles // waves_s)
        coop_k = -(-self.k_tiles // waves_k)
        return waves_s, waves_k, coop_s, coop_k

    @property
    def coop_a_split(self):
        """(waves_m, waves_k, coop_m, coop_k) for 2-D cooperative A loading."""
        return self._coop_2d_split(self.m_tiles_wg)

    @property
    def coop_b_split(self):
        """(waves_n, waves_k, coop_n, coop_k) for 2-D cooperative B loading."""
        return self._coop_2d_split(self.n_tiles_wg)

    @property
    def coop_a_mk_count(self):
        """Total tiles per wave for A: coop_m * coop_k."""
        _, _, coop_m, coop_k = self.coop_a_split
        return coop_m * coop_k

    @property
    def coop_b_nk_count(self):
        """Total tiles per wave for B: coop_n * coop_k."""
        _, _, coop_n, coop_k = self.coop_b_split
        return coop_n * coop_k

    @property
    def padded_m_tiles(self):
        """LDS-padded A tile count: COOP_A * NUM_WAVES (absorbs excess waves)."""
        return self.coop_a_count * self.num_waves

    @property
    def padded_n_tiles(self):
        """LDS-padded B tile count: COOP_B * NUM_WAVES (absorbs excess waves)."""
        return self.coop_b_count * self.num_waves

    @property
    def kernel_name(self):
        return KERNEL_NAMES[self.b_path]

    @property
    def estimated_agprs(self):
        """Coarse AGPR estimate: 4 AGPRs per 16x16 output tile per wave."""
        return self.m_tiles * self.n_tiles * 4

    @property
    def effective_b_stages(self):
        """Effective B pipeline depth."""
        return self.b_stages if self.b_stages > 0 else self.a_stages

    @property
    def pipeline_depth(self):
        """Combined pipeline depth = max(a_stages, effective_b_stages)."""
        return max(self.a_stages, self.effective_b_stages)

    @property
    def estimated_vgprs(self):
        """Coarse VGPR estimate: pipeline buffers + overhead.

        Each path holds (depth * tiles_per_wave * k_tiles * 4) VGPRs for
        in-flight loads. A uses a_stages depth, B uses effective_b_stages.
        A also needs LDS read buffers (1 stage worth).
        direct_b adds overhead for preshuffle address computation.
        """
        # A global load buffers: 2-D cooperative share, a_stages deep.
        a_load_bufs = self.coop_a_mk_count * self.a_stages * 4
        # A LDS read buffers: per-wave M_T tiles (consumed immediately).
        a_lds_read = self.m_tiles * self.k_tiles * 4

        if self.direct_b:
            # B global load buffers: b_stages deep (more depth = more in flight).
            b_load_bufs = self.n_tiles * self.k_tiles * self.effective_b_stages * 4
            # B has no LDS read buffers, but split vx4 -> 2x vx2 per tile.
            b_split = self.n_tiles * self.k_tiles * 4
            overhead = 30
        else:
            # B through LDS: 2-D cooperative share, a_stages deep.
            b_load_bufs = self.coop_b_nk_count * self.a_stages * 4
            b_split = self.n_tiles * self.k_tiles * 4  # LDS read buffers
            overhead = 10

        return a_load_bufs + a_lds_read + b_load_bufs + b_split + overhead

    @property
    def lds_bytes(self):
        """LDS bytes.

        A uses a_stages buffers. B uses LDS only if not direct_b.
        """
        a_tiles = self.m_tiles_wg * self.k_tiles
        b_tiles = 0 if self.direct_b else self.n_tiles_wg * self.k_tiles
        return self.a_stages * (a_tiles + b_tiles) * 1024

    @property
    def simd_occupancy(self):
        """Waves per SIMD (= num_wg_per_cu * ceil(num_waves / NUM_SIMDS))."""
        import math

        return self.num_wg_per_cu * math.ceil(self.num_waves / 4)

    @property
    def label(self):
        tile_str = f"_twg{self.m_tiles_wg}x{self.n_tiles_wg}x{self.k_tiles}"
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
            f"{tile_str}_s{self.a_stages}"
            f"{f'_bs{self.b_stages}' if self.b_stages > 0 else ''}"
            f"{lcm}{um}{peel}{self._label_suffix}"
        )


def _load_k_loop_helpers(load_type="flat", b_path="lds"):
    """Read the shared K-loop helper functions MLIR fragment."""
    helpers_path = get_mlir_file(K_LOOP_HELPERS_FILES[load_type])
    with open(helpers_path) as f:
        helpers = f.read()
    if b_path == "direct_b":
        direct_b_path = get_mlir_file("gemm_16x32_f16_k_loop_helpers_direct_b.mlir")
        with open(direct_b_path) as f:
            helpers += "\n" + f.read()
    elif b_path == "direct_ab":
        direct_ab_path = get_mlir_file("gemm_16x32_f16_k_loop_helpers_direct_ab.mlir")
        with open(direct_ab_path) as f:
            helpers += "\n" + f.read()
    return helpers


def _make_substitutions(cfg):
    """Build template substitutions dict for a WeakScaleConfig."""
    subs = {"{{K_LOOP_HELPERS}}": _load_k_loop_helpers(cfg.load_type, cfg.b_path)}
    effective_b_stages = cfg.b_stages if cfg.b_stages > 0 else None
    subs.update(
        constexpr_substitutions_16x32(
            cfg.m_tiles, cfg.n_tiles, cfg.k, cfg.a_stages, b_stages=effective_b_stages
        )
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
    subs["{{NUM_WAVES}}"] = str(cfg.num_waves)
    # 2-D cooperative split: (waves_m, waves_k) for A, (waves_n, waves_k) for B
    a_wm, a_wk, a_cm, a_ck = cfg.coop_a_split
    b_wn, b_wk, b_cn, b_ck = cfg.coop_b_split
    subs["{{COOP_A_WAVES_M}}"] = str(a_wm)
    subs["{{COOP_A_WAVES_K}}"] = str(a_wk)
    subs["{{COOP_A_M}}"] = str(a_cm)
    subs["{{COOP_A_K}}"] = str(a_ck)
    subs["{{MAX_COOP_A_M_START}}"] = str(max(0, cfg.m_tiles_wg - a_cm))
    subs["{{MAX_COOP_A_K_START}}"] = str(max(0, cfg.k_tiles - a_ck))
    subs["{{COOP_B_WAVES_N}}"] = str(b_wn)
    subs["{{COOP_B_WAVES_K}}"] = str(b_wk)
    subs["{{COOP_B_N}}"] = str(b_cn)
    subs["{{COOP_B_K}}"] = str(b_ck)
    subs["{{MAX_COOP_B_N_START}}"] = str(max(0, cfg.n_tiles_wg - b_cn))
    subs["{{MAX_COOP_B_K_START}}"] = str(max(0, cfg.k_tiles - b_ck))
    # Preshuffle layout parameters (f16: BK=32, 64 lanes, 16 bytes/lane).
    subs["{{STRIDE_N0_BYTES}}"] = str((cfg.k // 32) * 1024)
    subs["{{STRIDE_M0_BYTES}}"] = str((cfg.k // 32) * 1024)  # same formula as N
    subs["{{N_BLOCKS}}"] = str(cfg.n_dim // 16)
    subs["{{M_BLOCKS}}"] = str(cfg.m_dim // 16)
    subs["{{K_BLOCKS}}"] = str(cfg.k // 32)
    return subs


def compile_gemm(
    cfg,
    output_hsaco_path,
    print_ir_after_all=False,
    print_asm=False,
    num_vgprs=256,
    num_agprs=256,
    unroll_factor_multiplier=1,
    epilogue_peeling=True,
):
    """Compile a GEMM config to HSACO.

    Returns (hsaco_path, asm_str). Handles b_path (lds/direct) and load_type
    (flat/buffer) via cfg fields.
    """
    from aster import ir
    from aster.compiler.core import compile_mlir_file_to_asm, assemble_to_hsaco

    subs = _make_substitutions(cfg)

    def preprocess(content):
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    mlir_key = (cfg.b_path, cfg.load_type)
    mlir_file = MLIR_FILES[mlir_key]
    lib_paths = get_kittens_16x16_lds_library_paths(use_buffer=cfg.use_buffer)

    lcm_unroll = getattr(cfg, "lcm_unroll", True)
    pipeline = make_default_pass_pipeline(
        lcm_unroll=lcm_unroll,
        num_vgprs=num_vgprs,
        num_agprs=num_agprs,
        unroll_factor_multiplier=unroll_factor_multiplier,
        epilogue_peeling=epilogue_peeling,
    )

    ctx = ir.Context()
    ctx.__enter__()
    try:
        from aster.compiler.core import PrintOptions

        asm, _ = compile_mlir_file_to_asm(
            get_mlir_file(mlir_file),
            cfg.kernel_name,
            pipeline,
            ctx,
            library_paths=lib_paths,
            preprocess=preprocess,
            print_opts=PrintOptions.from_flags(
                print_ir_after_all=print_ir_after_all,
                print_asm=print_asm,
            ),
        )
        path = assemble_to_hsaco(
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

    Automatically preshuffles B when cfg.direct_b is True. Callers pass the original
    row-major B -- the preshuffle is applied here so there is a single code path for
    both test and bench.

    Skips (pytest.skip) if target GPU unavailable.
    """
    from aster.execution.core import execute_hsaco, InputArray, OutputArray
    from aster.execution.utils import system_has_gpu

    if not skip_gpu_check and not system_has_gpu(MCPU):
        pytest.skip(f"GPU {MCPU} not available, skip execution")

    # Preshuffle B for direct_b (single point of truth).
    A_gpu = shuffle_weight(A) if cfg.direct_a else A
    B_gpu = shuffle_weight(B) if cfg.direct_b else B

    C_output = np.zeros(cfg.m_dim * cfg.n_dim, dtype=np.float32)

    times_ns = execute_hsaco(
        hsaco_path=hsaco_path,
        kernel_name=cfg.kernel_name,
        arguments=[
            InputArray(A_gpu.flatten()),
            InputArray(B_gpu.flatten()),
            OutputArray(C_output),
        ],
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
            # Divisible: M_TILES_WG % NUM_WAVES == 0
            (32, 32, 4, 4, 2, 2),  # 2x2 waves, 2x2 tiles/wave
            (16, 16, 8, 8, 2, 2),  # 2x2 waves, 4x4 tiles/wave
            (32, 64, 4, 4, 2, 2),
            # OOB: M_TILES_WG % NUM_WAVES != 0 (excess waves load tile-0 region)
            (64, 64, 2, 2, 2, 2),  # 4 waves, 2 tiles -> coop_a=1, 2 waves OOB
            (32, 32, 6, 4, 2, 2),  # 4 waves, 6 A-tiles -> coop_a=2, wave3 OOB
            (32, 32, 4, 6, 2, 2),  # 4 waves, 6 B-tiles -> coop_b=2, wave3 OOB
            (32, 32, 6, 6, 2, 2),  # 4 waves, 6x6 -> both OOB
            (32, 64, 4, 4, 2, 4),  # 8 waves, 4 tiles -> coop=1, 4 waves OOB
        ],
        ids=[
            "div_2kx2k_twg4_w2x2",
            "div_2kx2k_twg8_w2x2",
            "div_2kx4k_twg4_w2x2",
            "oob_2kx2k_twg2_w2x2",
            "oob_6x4_twg6x4_w2x2",
            "oob_4x6_twg4x6_w2x2",
            "oob_6x6_twg6x6_w2x2",
            "oob_2kx4k_twg4_w2x4",
        ],
    )
    @pytest.mark.parametrize("a_stages", [2, 3], ids=["2stage", "3stage"])
    @pytest.mark.parametrize("load_type", ["flat", "buffer"], ids=["flat", "buffer"])
    @pytest.mark.parametrize("b_path", ["lds", "direct_b"], ids=["lds", "direct_b"])
    def test_correctness(
        self,
        m_wg,
        n_wg,
        m_tiles_wg,
        n_tiles_wg,
        m_waves,
        n_waves,
        a_stages,
        load_type,
        b_path,
    ):
        """Constexpr GEMM verified against numpy."""
        if (b_path, load_type) not in MLIR_FILES:
            pytest.skip(f"({b_path}, {load_type}) not yet implemented")
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
            a_stages,
            k,
            load_type=load_type,
            b_path=b_path,
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
    b_path = "direct_b" if a.direct_b else "lds"
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
        b_path=b_path,
    )

    from aster.compiler.metadata import parse_asm_kernel_resources

    a_mode = " direct-B" if cfg.direct_b else ""

    print(f"Config: {cfg.label} ({cfg.load_type}{a_mode})")
    print(f"  M={cfg.m_dim}, N={cfg.n_dim}, K={cfg.k}")
    print(f"  workgroups={cfg.num_workgroups}, threads={cfg.num_threads}")
    print(f"  per-wave tiles: {cfg.m_tiles}x{cfg.n_tiles}, LDS={cfg.lds_bytes}")

    if a.compile_only:
        if not a.hsaco:
            print("Error: --compile-only requires --hsaco <output_path>")
            raise SystemExit(1)
        _, asm = compile_gemm(
            cfg, a.hsaco, print_ir_after_all=a.print_ir_after_all, print_asm=a.print_asm
        )
        resources = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name)
        res = resources.get(cfg.kernel_name)
        if res:
            print(f"  resources: {res}")
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
                    cfg,
                    tmp.name,
                    print_ir_after_all=a.print_ir_after_all,
                    print_asm=a.print_asm,
                )
                resources = parse_asm_kernel_resources(asm, kernel_name=cfg.kernel_name)
                res = resources.get(cfg.kernel_name)
                if res:
                    print(f"  resources: {res}")
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
