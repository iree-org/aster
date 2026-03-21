"""Multi-CU GEMM: C[M x N] = A[M x K] @ B[N x K]^T.

Dispatched over a flat grid of (M // M_TILE) * (N // N_TILE) workgroups.  Each
workgroup contains num_waves wavefronts that cooperatively load and compute one
M_TILE x N_TILE output tile using a hybrid work distribution:
  - Loading (flat): wave wid loads a flat range of (K_T * M_T_LD) A-tiles and
    (K_T * N_T_LD) B-tiles, where M_T_LD = M_TILE // 16, N_T_LD = N_TILE // 16.
  - A single s_barrier synchronizes all waves.
  - Compute (2D): waves are arranged as a num_m_waves x num_n_waves grid.
    Wave wid maps to (wm = wid // num_n_waves, wn = wid % num_n_waves).
    Wave (wm, wn) reads A rows [wm*m_t_per_wave, (wm+1)*m_t_per_wave) from LDS,
    B cols [wn*n_t_per_wave, (wn+1)*n_t_per_wave) from LDS, and computes the
    m_t_per_wave x n_t_per_wave output tiles for its rectangle.

A CTA swizzle groups `swizzle` consecutive flat block IDs onto the same N-block so
that those blocks reuse the same B tile from L2 before moving to the next N-block.
swizzle=1 is the default and gives standard row-major dispatch (no swizzle).

Constraints:
  - k % k_tile == 0,  k_tile % 32 == 0
  - m % m_tile == 0,  m_tile % mfma_tile == 0  (mfma_tile=16 for 16x16, 32 for 32x32)
  - n % n_tile == 0,  n_tile % mfma_tile == 0
  - (m_t_ld * k_t) % num_waves == 0,  (n_t_ld * k_t) % num_waves == 0  (loading)
  - num_waves % num_m_waves == 0  (num_n_waves = num_waves // num_m_waves)
  - m_t % num_m_waves == 0,  n_t % num_n_waves == 0  (2D compute decomposition)
  - (m_t_per_wave * n_t_per_wave * k_inner) is a power of 2  (compute unroll)
  - (m // m_tile) % swizzle == 0
"""

import os
from dataclasses import dataclass
from typing import List

import numpy as np

from aster.pass_pipelines import make_default_pass_pipeline
from aster.execution.helpers import compile_and_run
from mlir_kernels.common import get_library_paths

MCPU = "gfx942"
WAVEFRONT_SIZE = 64
_DIR = os.path.dirname(__file__)
_KITTENS_LIB = os.path.join(_DIR, "..", "kittens", "library")
_MLIR_KERNELS_LIB = os.path.join(_DIR, "..", "..", "mlir_kernels", "library")
_HELPERS_FILE = os.path.join(
    _DIR, "..", "kittens", "test", "gemm_16x32_f16_k_loop_helpers.mlir"
)
_HELPERS_FILE_32x32 = os.path.join(_DIR, "gemm_32x32_f16_k_loop_helpers.mlir")


@dataclass
class GEMMConfig:
    """Configuration for the multi-CU generalized GEMM."""

    m: int
    n: int
    k: int
    m_tile: int
    n_tile: int
    k_tile: int
    num_waves: int
    num_m_waves: int = 1
    swizzle: int = 1
    mfma_variant: str = (
        "16x16"  # "16x16" uses v_mfma_f32_16x16x16_f16, "32x32" uses v_mfma_f32_32x32x8_f16
    )

    def __post_init__(self):
        assert self.mfma_variant in (
            "16x16",
            "32x32",
        ), f"mfma_variant must be '16x16' or '32x32', got {self.mfma_variant!r}"
        assert (
            self.m % self.m_tile == 0
        ), f"m={self.m} not divisible by m_tile={self.m_tile}"
        assert (
            self.n % self.n_tile == 0
        ), f"n={self.n} not divisible by n_tile={self.n_tile}"
        assert (
            self.k % self.k_tile == 0
        ), f"k={self.k} not divisible by k_tile={self.k_tile}"
        assert self.k_tile % 32 == 0, f"k_tile={self.k_tile} not divisible by 32"
        assert (self.m_t_ld * self.k_t) % self.num_waves == 0, (
            f"m_t_ld * k_t = {self.m_t_ld * self.k_t} not divisible by"
            f" num_waves={self.num_waves}"
        )
        assert (self.n_t_ld * self.k_t) % self.num_waves == 0, (
            f"n_t_ld * k_t = {self.n_t_ld * self.k_t} not divisible by"
            f" num_waves={self.num_waves}"
        )
        assert (
            self.num_waves % self.num_m_waves == 0
        ), f"num_waves={self.num_waves} not divisible by num_m_waves={self.num_m_waves}"
        assert (
            self.m_t % self.num_m_waves == 0
        ), f"m_t={self.m_t} not divisible by num_m_waves={self.num_m_waves}"
        assert (
            self.n_t % self.num_n_waves == 0
        ), f"n_t={self.n_t} not divisible by num_n_waves={self.num_n_waves}"
        _q = self.m_t_per_wave * self.n_t_per_wave * self.k_inner
        assert _q > 0 and (_q & (_q - 1)) == 0, (
            f"m_t_per_wave * n_t_per_wave * k_inner = {_q} must be a power of 2"
            f" (m_t_per_wave={self.m_t_per_wave}, n_t_per_wave={self.n_t_per_wave},"
            f" k_inner={self.k_inner})"
        )
        assert (
            self.m_wg % self.swizzle == 0
        ), f"m_wg={self.m_wg} not divisible by swizzle={self.swizzle}"

    @property
    def mfma_tile(self) -> int:
        """Output tile side length for the selected MFMA variant (16 or 32)."""
        return 32 if self.mfma_variant == "32x32" else 16

    @property
    def m_wg(self) -> int:
        """Workgroup grid size in M."""
        return self.m // self.m_tile

    @property
    def n_wg(self) -> int:
        """Workgroup grid size in N."""
        return self.n // self.n_tile

    @property
    def k_t(self) -> int:
        """K tiles per outer loop step."""
        return self.k_tile // 32

    @property
    def k_tiles(self) -> int:
        """Total 16x32 K tiles (= K / 32)."""
        return self.k // 32

    @property
    def m_t(self) -> int:
        """M MFMA tiles per workgroup (in mfma_tile units)."""
        return self.m_tile // self.mfma_tile

    @property
    def n_t(self) -> int:
        """N MFMA tiles per workgroup (in mfma_tile units)."""
        return self.n_tile // self.mfma_tile

    @property
    def m_t_ld(self) -> int:
        """LDS 16-row slot count per workgroup (= m_tile // 16; equals m_t for 16x16, 2*m_t for 32x32)."""
        return self.m_tile // 16

    @property
    def n_t_ld(self) -> int:
        """LDS 16-row slot count per workgroup (= n_tile // 16; equals n_t for 16x16, 2*n_t for 32x32)."""
        return self.n_tile // 16

    @property
    def k_inner(self) -> int:
        """MFMA k-steps per K_TILE: K_TILE // K_MMA (2 for 16x16, 4 for 32x32)."""
        return self.k_tile // (8 if self.mfma_variant == "32x32" else 16)

    @property
    def num_n_waves(self) -> int:
        """N-dimension wave count = num_waves // num_m_waves."""
        return self.num_waves // self.num_m_waves

    @property
    def m_t_per_wave(self) -> int:
        """M MFMA tiles per wave = m_t // num_m_waves."""
        return self.m_t // self.num_m_waves

    @property
    def n_t_per_wave(self) -> int:
        """N MFMA tiles per wave = n_t // num_n_waves."""
        return self.n_t // self.num_n_waves

    @property
    def tiles_per_wave(self) -> int:
        """Output MMA tiles per wave = m_t_per_wave * n_t_per_wave."""
        return self.m_t_per_wave * self.n_t_per_wave

    @property
    def a_per_wave(self) -> int:
        """A 16-row tiles loaded per wave across all K_T steps = M_T_LD * K_T / num_waves."""
        return self.m_t_ld * self.k_t // self.num_waves

    @property
    def b_per_wave(self) -> int:
        """B 16-row tiles loaded per wave across all K_T steps = N_T_LD * K_T / num_waves."""
        return self.n_t_ld * self.k_t // self.num_waves

    @property
    def num_blocks(self) -> int:
        """Total number of workgroups dispatched."""
        return self.m_wg * self.n_wg

    @property
    def stride_ab(self) -> int:
        """Row stride for A and B in bytes (f16)."""
        return self.k * 2

    @property
    def stride_c(self) -> int:
        """Row stride for the full C matrix in bytes (f32)."""
        return self.n * 4

    @property
    def a_lds_bytes(self) -> int:
        """Total LDS for A = k_t * m_t_ld * 1024 bytes (16-row slot units; same as k_t * m_t * 1024 for 16x16)."""
        return self.k_t * self.m_t_ld * 1024

    @property
    def b_lds_bytes(self) -> int:
        """Total LDS for B = k_t * n_t_ld * 1024 bytes (16-row slot units; same as k_t * n_t * 1024 for 16x16)."""
        return self.k_t * self.n_t_ld * 1024

    @property
    def total_lds_bytes(self) -> int:
        """Total LDS per workgroup = A_LDS_BYTES + B_LDS_BYTES."""
        return self.a_lds_bytes + self.b_lds_bytes

    @property
    def num_threads(self) -> int:
        return self.num_waves * WAVEFRONT_SIZE


def _get_library_paths() -> List[str]:
    """Library paths for 16x16 MFMA + 16x64_b LDS (dwordx4, XOR swizzle)."""
    return get_library_paths() + [
        os.path.join(_MLIR_KERNELS_LIB, "common", "indexing_ptr.mlir"),
        os.path.join(_KITTENS_LIB, "global_16x64_b.mlir"),
        os.path.join(_KITTENS_LIB, "lds_16x64_b.mlir"),
        os.path.join(_KITTENS_LIB, "lds_mfma_16x64_b.mlir"),
        os.path.join(_KITTENS_LIB, "compute_16x16_f16.mlir"),
    ]


def _load_k_loop_helpers() -> str:
    with open(_HELPERS_FILE) as f:
        return f.read()


_PIPELINE_STAGE_CONFIGS = {
    # num_stages: (STAGE_GLOBAL_LOAD, STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE)
    # The pipeliner requires at least num_stages loop iterations.
    0: (0, 0, 0, 0),
    1: (0, 0, 0, 0),
    2: (0, 1, 1, 1),
    3: (0, 1, 2, 2),
    4: (0, 1, 2, 3),
}


def _make_substitutions(cfg: GEMMConfig) -> dict:
    # Cap pipeline depth so that num_stages * total_lds_bytes ≤ 32768 (half the
    # 64 KB LDS limit), which keeps two workgroups per CU resident and preserves
    # the natural inter-WG latency-hiding that makes this kernel efficient.
    # For example: k_tile=32 (16 KB LDS) → 2-stage; k_tile=64 (32 KB) → 1-stage.
    k_outer_iters = cfg.k // cfg.k_tile
    max_stages_for_2wg = 32768 // cfg.total_lds_bytes
    num_stages = min(4, max_stages_for_2wg, max(1, k_outer_iters))
    stage_gl, stage_dw, stage_dr, stage_c = _PIPELINE_STAGE_CONFIGS[num_stages]
    return {
        "{{K}}": str(cfg.k),
        "{{K_T}}": str(cfg.k_t),
        "{{K_INNER}}": str(cfg.k_inner),
        "{{K_TILES}}": str(cfg.k_tiles),
        "{{M_T}}": str(cfg.m_t),
        "{{N_T}}": str(cfg.n_t),
        "{{NUM_M_WAVES}}": str(cfg.num_m_waves),
        "{{NUM_N_WAVES}}": str(cfg.num_n_waves),
        "{{M_T_PER_WAVE}}": str(cfg.m_t_per_wave),
        "{{N_T_PER_WAVE}}": str(cfg.n_t_per_wave),
        "{{TILES_PER_WAVE}}": str(cfg.tiles_per_wave),
        "{{A_PER_WAVE}}": str(cfg.a_per_wave),
        "{{B_PER_WAVE}}": str(cfg.b_per_wave),
        "{{NUM_WAVES}}": str(cfg.num_waves),
        "{{NUM_THREADS}}": str(cfg.num_threads),
        "{{NUM_BLOCKS}}": str(cfg.num_blocks),
        "{{M_WG}}": str(cfg.m_wg),
        "{{N_WG}}": str(cfg.n_wg),
        "{{SWIZZLE}}": str(cfg.swizzle),
        "{{SWIZZLE_NWG}}": str(cfg.swizzle * cfg.n_wg),
        "{{STRIDE_AB}}": str(cfg.stride_ab),
        "{{STRIDE_C}}": str(cfg.stride_c),
        "{{A_LDS_BYTES}}": str(cfg.a_lds_bytes),
        "{{B_LDS_BYTES}}": str(cfg.b_lds_bytes),
        "{{K_LOOP_HELPERS}}": _load_k_loop_helpers(),
        "{{STAGE_GLOBAL_LOAD}}": str(stage_gl),
        "{{STAGE_DS_WRITE}}": str(stage_dw),
        "{{STAGE_DS_READ}}": str(stage_dr),
        "{{STAGE_COMPUTE}}": str(stage_c),
    }


def run_gemm(
    cfg: GEMMConfig,
    A: np.ndarray,
    B: np.ndarray,
    print_ir_after_all: bool = False,
    num_iterations: int = 1,
    num_warmup: int = 0,
    num_nops: int = 0,
    print_asm: bool = False,
) -> np.ndarray:
    """Compile and run the GEMM kernel; return the flat C output buffer."""
    subs = _make_substitutions(cfg)

    def preprocess(content: str) -> str:
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    C_output = np.zeros(cfg.m * cfg.n, dtype=np.float32)

    times = compile_and_run(
        file_name=os.path.join(_DIR, "gemm_fp16_lds.mlir"),
        kernel_name="gemm_fp16_lds",
        input_data=[A.flatten(), B.flatten()],
        output_data=[C_output],
        pass_pipeline=make_default_pass_pipeline(lcm_unroll=True),
        preprocess=preprocess,
        library_paths=_get_library_paths(),
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        grid_dim=(cfg.num_blocks, 1, 1),
        block_dim=(cfg.num_threads, 1, 1),
        print_ir_after_all=print_ir_after_all,
        num_iterations=num_iterations,
        print_asm=print_asm,
    )
    assert num_iterations > num_warmup, "num_iterations must be greater than num_warmup"
    avg_ns = np.mean(times[num_warmup:])
    flops = 2 * cfg.m * cfg.n * cfg.k
    tflops = flops / (avg_ns * 1e-9) / 1e12
    print(f"avg={avg_ns:.1f} ns  {tflops:.3f} TFLOP/s")
    return C_output


def _get_library_paths_32x32() -> List[str]:
    """Library paths for 32x32 MFMA + 16x64_b LDS (dwordx4, XOR swizzle)."""
    return get_library_paths() + [
        os.path.join(_MLIR_KERNELS_LIB, "common", "indexing_ptr.mlir"),
        os.path.join(_KITTENS_LIB, "global_16x64_b.mlir"),
        os.path.join(_KITTENS_LIB, "lds_16x64_b.mlir"),
        os.path.join(_KITTENS_LIB, "lds_mfma_32x32_f16.mlir"),
        os.path.join(_KITTENS_LIB, "compute_32x32_f16.mlir"),
    ]


def _load_k_loop_helpers_32x32() -> str:
    with open(_HELPERS_FILE_32x32) as f:
        return f.read()


def _make_substitutions_32x32(cfg: GEMMConfig) -> dict:
    # Cap pipeline depth so that num_stages * total_lds_bytes ≤ 32768 (half the
    # 64 KB LDS limit), which keeps two workgroups per CU resident and preserves
    # the natural inter-WG latency-hiding that makes this kernel efficient.
    # For example: k_tile=32 (16 KB LDS) → 2-stage; k_tile=64 (32 KB) → 1-stage.
    k_outer_iters = cfg.k // cfg.k_tile
    max_stages_for_2wg = 32768 // cfg.total_lds_bytes
    num_stages = min(4, max_stages_for_2wg, max(1, k_outer_iters))
    stage_gl, stage_dw, stage_dr, stage_c = _PIPELINE_STAGE_CONFIGS[num_stages]
    return {
        "{{K}}": str(cfg.k),
        "{{K_T}}": str(cfg.k_t),
        "{{K_INNER}}": str(cfg.k_inner),
        "{{K_TILES}}": str(cfg.k_tiles),
        "{{M_T}}": str(cfg.m_t),
        "{{N_T}}": str(cfg.n_t),
        "{{M_T_LD}}": str(cfg.m_t_ld),
        "{{N_T_LD}}": str(cfg.n_t_ld),
        "{{NUM_M_WAVES}}": str(cfg.num_m_waves),
        "{{NUM_N_WAVES}}": str(cfg.num_n_waves),
        "{{M_T_PER_WAVE}}": str(cfg.m_t_per_wave),
        "{{N_T_PER_WAVE}}": str(cfg.n_t_per_wave),
        "{{TILES_PER_WAVE}}": str(cfg.tiles_per_wave),
        "{{A_PER_WAVE}}": str(cfg.a_per_wave),
        "{{B_PER_WAVE}}": str(cfg.b_per_wave),
        "{{NUM_WAVES}}": str(cfg.num_waves),
        "{{NUM_THREADS}}": str(cfg.num_threads),
        "{{NUM_BLOCKS}}": str(cfg.num_blocks),
        "{{M_WG}}": str(cfg.m_wg),
        "{{N_WG}}": str(cfg.n_wg),
        "{{SWIZZLE}}": str(cfg.swizzle),
        "{{SWIZZLE_NWG}}": str(cfg.swizzle * cfg.n_wg),
        "{{STRIDE_AB}}": str(cfg.stride_ab),
        "{{STRIDE_C}}": str(cfg.stride_c),
        "{{A_LDS_BYTES}}": str(cfg.a_lds_bytes),
        "{{B_LDS_BYTES}}": str(cfg.b_lds_bytes),
        "{{K_LOOP_HELPERS_32x32}}": _load_k_loop_helpers_32x32(),
        "{{STAGE_GLOBAL_LOAD}}": str(stage_gl),
        "{{STAGE_DS_WRITE}}": str(stage_dw),
        "{{STAGE_DS_READ}}": str(stage_dr),
        "{{STAGE_COMPUTE}}": str(stage_c),
    }


def run_gemm_32x32(
    cfg: GEMMConfig,
    A: np.ndarray,
    B: np.ndarray,
    print_ir_after_all: bool = False,
    num_iterations: int = 1,
    num_warmup: int = 0,
    print_asm: bool = False,
) -> np.ndarray:
    """Compile and run the 32x32x8 GEMM kernel; return the flat C output buffer."""
    subs = _make_substitutions_32x32(cfg)

    def preprocess(content: str) -> str:
        for pattern, replacement in subs.items():
            content = content.replace(pattern, replacement)
        return content

    C_output = np.zeros(cfg.m * cfg.n, dtype=np.float32)

    times = compile_and_run(
        file_name=os.path.join(_DIR, "gemm_fp16_lds_32x32.mlir"),
        kernel_name="gemm_fp16_lds_32x32",
        input_data=[A.flatten(), B.flatten()],
        output_data=[C_output],
        pass_pipeline=make_default_pass_pipeline(lcm_unroll=True),
        preprocess=preprocess,
        library_paths=_get_library_paths_32x32(),
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        grid_dim=(cfg.num_blocks, 1, 1),
        block_dim=(cfg.num_threads, 1, 1),
        print_ir_after_all=print_ir_after_all,
        num_iterations=num_iterations,
        print_asm=print_asm,
    )
    assert num_iterations > num_warmup, "num_iterations must be greater than num_warmup"
    avg_ns = np.mean(times[num_warmup:])
    flops = 2 * cfg.m * cfg.n * cfg.k
    tflops = flops / (avg_ns * 1e-9) / 1e12
    print(f"avg={avg_ns:.1f} ns  {tflops:.3f} TFLOP/s")
    return C_output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run multi-CU GEMM fp16 LDS kernel.")
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--m-tile", type=int, default=16)
    parser.add_argument("--n-tile", type=int, default=16)
    parser.add_argument("--k-tile", type=int, default=64)
    parser.add_argument("--num-waves", type=int, default=1)
    parser.add_argument("--num-m-waves", type=int, default=1)
    parser.add_argument("--swizzle", type=int, default=1)
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--print-asm", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--num-warmup", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--nops", type=int, default=0)
    parser.add_argument(
        "--mfma-variant",
        choices=["16x16", "32x32"],
        default="16x16",
        help="MFMA variant: 16x16 uses v_mfma_f32_16x16x16_f16, 32x32 uses v_mfma_f32_32x32x8_f16.",
    )
    args = parser.parse_args()

    cfg = GEMMConfig(
        args.m,
        args.n,
        args.k,
        args.m_tile,
        args.n_tile,
        args.k_tile,
        args.num_waves,
        args.num_m_waves,
        args.swizzle,
        mfma_variant=args.mfma_variant,
    )
    variant_tag = "32x32x8" if args.mfma_variant == "32x32" else "16x16x16"
    print(
        f"GEMM {cfg.m}x{cfg.n}x{cfg.k} [{variant_tag}]"
        f"  tile={cfg.m_tile}x{cfg.n_tile}  k_tile={cfg.k_tile}"
        f"  num_waves={cfg.num_waves}  grid={cfg.m_wg}x{cfg.n_wg}"
        f"  swizzle={cfg.swizzle}  blocks={cfg.num_blocks}"
    )
    np.random.seed(42)
    A = (np.random.randn(cfg.m, cfg.k)).astype(np.float16)
    B = (np.random.randn(cfg.n, cfg.k)).astype(np.float16)
    if args.mfma_variant == "32x32":
        C = run_gemm_32x32(
            cfg,
            A,
            B,
            print_ir_after_all=args.print_ir_after_all,
            num_iterations=args.num_iterations,
            num_warmup=args.num_warmup,
            print_asm=args.print_asm,
        )
    else:
        C = run_gemm(
            cfg,
            A,
            B,
            print_ir_after_all=args.print_ir_after_all,
            num_iterations=args.num_iterations,
            num_warmup=args.num_warmup,
            num_nops=args.nops,
            print_asm=args.print_asm,
        )
    print(f"C[0:4] = {C[:4]}")
    if args.verify:
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        rel_err = np.linalg.norm(C - expected) / np.linalg.norm(expected)
        print(f"Relative error: {rel_err:.6e}", flush=True)
        np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)
        print("Verification passed.")
