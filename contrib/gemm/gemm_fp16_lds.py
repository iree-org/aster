"""Multi-CU GEMM: C[M x N] = A[M x K] @ B[N x K]^T.

Dispatched over a flat grid of (M // M_TILE) * (N // N_TILE) workgroups.  Each
workgroup contains num_waves wavefronts that cooperatively load and compute one
M_TILE x N_TILE output tile:
  - Loading is 1D: wave wid loads M_T_PER_WAVE rows of A and N_T_PER_WAVE cols of B.
  - A single s_barrier synchronizes all waves.
  - Compute is 2D: wm = wid // num_n_waves, wn = wid % num_n_waves.
    Wave (wm, wn) reads M_T_PW_C A rows and N_T_PW_C B cols from LDS and computes
    the M_T_PW_C x N_T_PW_C C sub-tile at row wm*M_T_PW_C, col wn*N_T_PW_C.
  - num_m_waves=0 (default) selects 1D mode: num_m_waves=num_waves, num_n_waves=1,
    M_T_PW_C=M_T_PER_WAVE, N_T_PW_C=N_T (all N columns per wave, classic layout).

A CTA swizzle groups `swizzle` consecutive flat block IDs onto the same N-block so
that those blocks reuse the same B tile from L2 before moving to the next N-block.
swizzle=1 is the default and gives standard row-major dispatch (no swizzle).

Constraints:
  - k % k_tile == 0,  k_tile % 32 == 0
  - m % m_tile == 0,  n % n_tile == 0
  - m_tile % (16 * num_waves) == 0
  - n_tile % (16 * num_waves) == 0
  - num_waves % num_m_waves == 0  (num_m_waves=0 is resolved to num_waves)
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
    swizzle: int = 1
    num_m_waves: int = 0  # 0 = auto (= num_waves, 1D mode); else divides num_waves

    def __post_init__(self):
        # Resolve num_m_waves: 0 means all waves in M direction (1D mode).
        if self.num_m_waves == 0:
            self.num_m_waves = self.num_waves
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
        assert (
            self.num_waves % self.num_m_waves == 0
        ), f"num_waves={self.num_waves} not divisible by num_m_waves={self.num_m_waves}"
        assert (
            self.m_tile % (16 * self.num_waves) == 0
        ), f"m_tile={self.m_tile} not divisible by 16 * num_waves={16 * self.num_waves}"
        assert (
            self.n_tile % (16 * self.num_waves) == 0
        ), f"n_tile={self.n_tile} not divisible by 16 * num_waves={16 * self.num_waves}"
        assert (
            self.m_wg % self.swizzle == 0
        ), f"m_wg={self.m_wg} not divisible by swizzle={self.swizzle}"

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
        """M MFMA tiles per workgroup."""
        return self.m_tile // 16

    @property
    def n_t(self) -> int:
        """N MFMA tiles per workgroup."""
        return self.n_tile // 16

    @property
    def num_n_waves(self) -> int:
        """Number of wave groups in N direction."""
        return self.num_waves // self.num_m_waves

    @property
    def m_t_per_wave(self) -> int:
        """M MFMA tiles loaded per wave (1D loading assignment)."""
        return self.m_t // self.num_waves

    @property
    def n_t_per_wave(self) -> int:
        """B tiles each wave loads (cooperative 1D B loading)."""
        return self.n_t // self.num_waves

    @property
    def m_t_pw_c(self) -> int:
        """M MFMA tiles per wave for compute (2D partition, = m_t / num_m_waves)."""
        return self.m_t // self.num_m_waves

    @property
    def n_t_pw_c(self) -> int:
        """N MFMA tiles per wave for compute (2D partition, = n_t / num_n_waves)."""
        return self.n_t // self.num_n_waves

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
        """Total LDS for A = K_T * M_T * 1024 bytes (per workgroup)."""
        return self.k_t * self.m_t * 1024

    @property
    def b_lds_bytes(self) -> int:
        """Total LDS for B = K_T * N_T * 1024 bytes (per workgroup)."""
        return self.k_t * self.n_t * 1024

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


def _make_substitutions(cfg: GEMMConfig) -> dict:
    return {
        "{{K}}": str(cfg.k),
        "{{K_T}}": str(cfg.k_t),
        "{{K_TILES}}": str(cfg.k_tiles),
        "{{M_T}}": str(cfg.m_t),
        "{{N_T}}": str(cfg.n_t),
        "{{M_T_PER_WAVE}}": str(cfg.m_t_per_wave),
        "{{N_T_PER_WAVE}}": str(cfg.n_t_per_wave),
        "{{NUM_WAVES}}": str(cfg.num_waves),
        "{{NUM_THREADS}}": str(cfg.num_threads),
        "{{NUM_BLOCKS}}": str(cfg.num_blocks),
        "{{NUM_M_WAVES}}": str(cfg.num_m_waves),
        "{{NUM_N_WAVES}}": str(cfg.num_n_waves),
        "{{M_T_PW_C}}": str(cfg.m_t_pw_c),
        "{{N_T_PW_C}}": str(cfg.n_t_pw_c),
        "{{M_WG}}": str(cfg.m_wg),
        "{{N_WG}}": str(cfg.n_wg),
        "{{SWIZZLE}}": str(cfg.swizzle),
        "{{SWIZZLE_NWG}}": str(cfg.swizzle * cfg.n_wg),
        "{{STRIDE_AB}}": str(cfg.stride_ab),
        "{{STRIDE_C}}": str(cfg.stride_c),
        "{{A_LDS_BYTES}}": str(cfg.a_lds_bytes),
        "{{B_LDS_BYTES}}": str(cfg.b_lds_bytes),
        "{{K_LOOP_HELPERS}}": _load_k_loop_helpers(),
        # Stage annotations: 0..3 for software pipelining (double-buffer LDS).
        # GLOBAL_LOAD=0 → DS_WRITE=1 → DS_READ=2 → COMPUTE=3.
        # Set all to 0 to fall back to non-pipelined execution.
        "{{STAGE_GLOBAL_LOAD}}": "0",
        "{{STAGE_DS_WRITE}}": "1",
        "{{STAGE_DS_READ}}": "2",
        "{{STAGE_COMPUTE}}": "3",
    }


def run_gemm(
    cfg: GEMMConfig,
    A: np.ndarray,
    B: np.ndarray,
    print_ir_after_all: bool = False,
    num_iterations: int = 1,
    num_warmup: int = 0,
    num_nops: int = 0,
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
    parser.add_argument("--swizzle", type=int, default=1)
    parser.add_argument("--num-m-waves", type=int, default=0)
    parser.add_argument("--print-ir-after-all", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--num-warmup", type=int, default=0)
    parser.add_argument("--num-iterations", type=int, default=1)
    parser.add_argument("--nops", type=int, default=0)
    args = parser.parse_args()

    cfg = GEMMConfig(
        args.m,
        args.n,
        args.k,
        args.m_tile,
        args.n_tile,
        args.k_tile,
        args.num_waves,
        args.swizzle,
        args.num_m_waves,
    )
    print(
        f"GEMM {cfg.m}x{cfg.n}x{cfg.k}"
        f"  tile={cfg.m_tile}x{cfg.n_tile}  k_tile={cfg.k_tile}"
        f"  num_waves={cfg.num_waves}  grid={cfg.m_wg}x{cfg.n_wg}"
        f"  swizzle={cfg.swizzle}  blocks={cfg.num_blocks}"
    )
    np.random.seed(42)
    A = (np.random.randn(cfg.m, cfg.k)).astype(np.float16)
    B = (np.random.randn(cfg.n, cfg.k)).astype(np.float16)
    C = run_gemm(
        cfg,
        A,
        B,
        print_ir_after_all=args.print_ir_after_all,
        num_iterations=args.num_iterations,
        num_warmup=args.num_warmup,
        num_nops=args.nops,
    )
    print(f"C[0:4] = {C[:4]}")
    if args.verify:
        expected = (A.astype(np.float32) @ B.astype(np.float32).T).flatten()
        rel_err = np.linalg.norm(C - expected) / np.linalg.norm(expected)
        print(f"Relative error: {rel_err:.6e}", flush=True)
        np.testing.assert_allclose(C, expected, rtol=1e-2, atol=1e-2)
        print("Verification passed.")
