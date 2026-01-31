"""Shared kernel utilities for GEMM/MFMA tests and benchmarks.

This module provides unified config classes, preprocess functions, and verification
functions used by both test and benchmark infrastructure.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import numpy as np

from mlir_kernels.gemm_config import validate_gemm_config

# =============================================================================
# MFMA Constants (16x16x16 operation)
# =============================================================================

MFMA_SIZE_M = 16
MFMA_SIZE_N = 16
MFMA_SIZE_K = 16
MFMA_SIZE = MFMA_SIZE_M * MFMA_SIZE_N * MFMA_SIZE_K

# LDS limit (64KB)
LDS_SIZE_LIMIT = 65536


# =============================================================================
# GEMM Config
# =============================================================================


@dataclass
class GEMMConfig:
    """Configuration for GEMM 16x16x16 kernels.

    Used by both tests and benchmarks. Inheriting from BaseConfig is optional
    (benchmarks do, tests may not).
    """

    m: int  # Problem size M
    n: int  # Problem size N
    k: int  # Problem size K
    m_tile: int = 16  # Tile size in M dimension
    n_tile: int = 16  # Tile size in N dimension
    k_tile: int = 16  # Tile size in K dimension
    num_waves: int = 1  # Number of wavefronts per workgroup
    mlir_file: str = ""
    kernel_name: str = "test_matmul_kernel"
    wavefront_size: int = 64
    pass_pipeline: str = ""
    mcpu: str = "gfx942"

    # Peak performance specs (for efficiency calculations)
    shader_clock_mhz: float = 2100.0
    peak_gbps: float = 5300.0
    peak_tflops: float = 1307.0

    def __post_init__(self):
        """Validate configuration using shared validation logic."""
        is_valid, error = validate_gemm_config(
            self.m,
            self.n,
            self.k,
            self.m_tile,
            self.n_tile,
            self.k_tile,
            self.num_waves,
        )
        if not is_valid:
            raise ValueError(error)

    @property
    def num_blocks_m(self) -> int:
        """Number of blocks in M dimension."""
        return self.m // self.m_tile

    @property
    def num_blocks_n(self) -> int:
        """Number of blocks in N dimension."""
        return self.n // self.n_tile

    @property
    def num_workgroups(self) -> int:
        """Total number of workgroups (blocks)."""
        return self.num_blocks_m * self.num_blocks_n

    @property
    def num_threads(self) -> int:
        """Number of threads per workgroup."""
        return self.wavefront_size * self.num_waves

    @property
    def size_a(self) -> int:
        """Bytes per element for A (f16)."""
        return np.dtype(np.float16).itemsize

    @property
    def size_b(self) -> int:
        """Bytes per element for B (f16)."""
        return np.dtype(np.float16).itemsize

    @property
    def size_c(self) -> int:
        """Bytes per element for C (f32)."""
        return np.dtype(np.float32).itemsize

    @property
    def total_bytes(self) -> int:
        """Compute total bytes read/written."""
        bytes_a = self.m * self.k * self.size_a
        bytes_b = self.k * self.n * self.size_b
        bytes_c = self.m * self.n * self.size_c
        return bytes_a + bytes_b + bytes_c

    @property
    def lds_a_size(self) -> int:
        """LDS size for A tile."""
        return self.m_tile * self.k_tile * self.size_a

    @property
    def lds_b_size(self) -> int:
        """LDS size for B tile."""
        return self.n_tile * self.k_tile * self.size_b

    @property
    def lds_total_size(self) -> int:
        """Total LDS size for A and B tiles."""
        return self.lds_a_size + self.lds_b_size

    @property
    def total_flops(self) -> int:
        """Total FLOPs for the matmul (2*M*N*K)."""
        return 2 * self.m * self.n * self.k

    @property
    def loop_size_d_mmnnkk(self) -> int:
        """MFMA loop count for scheduling."""
        mnkt = self.m_tile * self.n_tile * self.k_tile
        mnkt_mfma = mnkt // MFMA_SIZE
        return mnkt_mfma // self.num_waves


# =============================================================================
# MFMA Config (block-based, m/n/k are block counts not sizes)
# =============================================================================


@dataclass
class MFMAConfig:
    """Configuration for MFMA 16x16x16 block-based kernels.

    Here m, n, k are the number of 16x16 blocks, not element counts.
    """

    m: int  # Number of 16x16 blocks in M dimension
    n: int  # Number of 16x16 blocks in N dimension
    k: int  # Number of 16x16 blocks in K dimension
    num_workgroups: int = 1
    num_waves: int = 1
    mlir_file: str = ""
    kernel_name: str = "test_matmul_kernel"
    wavefront_size: int = 64
    pass_pipeline: str = ""
    mcpu: str = "gfx942"

    # Peak performance specs
    shader_clock_mhz: float = 2100.0
    peak_gbps: float = 5300.0
    peak_tflops: float = 1307.0

    @property
    def num_threads(self) -> int:
        """Number of threads per workgroup."""
        return self.wavefront_size * self.num_waves

    @property
    def batch(self) -> int:
        """Batch size (each workgroup/wave needs its own data)."""
        return self.num_workgroups * self.num_waves

    @property
    def size_a(self) -> int:
        """Bytes per element for A (f16)."""
        return np.dtype(np.float16).itemsize

    @property
    def size_b(self) -> int:
        """Bytes per element for B (f16)."""
        return np.dtype(np.float16).itemsize

    @property
    def size_c(self) -> int:
        """Bytes per element for C (f32)."""
        return np.dtype(np.float32).itemsize

    @property
    def total_bytes(self) -> int:
        """Compute total bytes read/written."""
        bytes_a = self.m * self.k * MFMA_SIZE_M * MFMA_SIZE_K * self.size_a
        bytes_b = self.k * self.n * MFMA_SIZE_K * MFMA_SIZE_N * self.size_b
        bytes_c = self.m * self.n * MFMA_SIZE_M * MFMA_SIZE_N * self.size_c
        return (bytes_a + bytes_b + bytes_c) * self.batch

    @property
    def lds_a_size(self) -> int:
        """LDS size for A matrix."""
        return (
            self.m * self.k * MFMA_SIZE_M * MFMA_SIZE_K * self.size_a * self.num_waves
        )

    @property
    def lds_b_size(self) -> int:
        """LDS size for B matrix."""
        return (
            self.k * self.n * MFMA_SIZE_K * MFMA_SIZE_N * self.size_b * self.num_waves
        )

    @property
    def lds_total_size(self) -> int:
        """Total LDS size for A and B matrices."""
        return self.lds_a_size + self.lds_b_size

    @property
    def total_flops(self) -> int:
        """Total FLOPs for the matmul."""
        flops_per_wave = self.m * self.n * self.k * 2 * MFMA_SIZE
        return flops_per_wave * self.batch


# =============================================================================
# Preprocess Functions
# =============================================================================


def make_gemm_preprocess(config: GEMMConfig) -> Callable[[str], str]:
    """Create a preprocess function for GEMM MLIR templates."""

    def preprocess(x: str) -> str:
        x = x.replace("{{SIZE_M}}", str(config.m))
        x = x.replace("{{SIZE_N}}", str(config.n))
        x = x.replace("{{SIZE_K}}", str(config.k))
        x = x.replace("{{TILE_SIZE_M}}", str(config.m_tile))
        x = x.replace("{{TILE_SIZE_N}}", str(config.n_tile))
        x = x.replace("{{TILE_SIZE_K}}", str(config.k_tile))
        x = x.replace("{{NUM_BLOCKS}}", str(config.num_workgroups))
        x = x.replace("{{NUM_THREADS}}", str(config.num_threads))
        x = x.replace("{{LDS_SIZE}}", str(config.lds_total_size))
        x = x.replace("{{SIZE_K_BY_TILE_SIZE_K}}", str(config.k // config.k_tile))
        x = x.replace("{{LOOP_SIZE_D_MMNNKK}}", str(config.loop_size_d_mmnnkk))
        return x

    return preprocess


def make_mfma_preprocess(config: MFMAConfig) -> Callable[[str], str]:
    """Create a preprocess function for MFMA MLIR templates."""

    def preprocess(x: str) -> str:
        x = x.replace("{{SIZE_M}}", str(config.m))
        x = x.replace("{{SIZE_N}}", str(config.n))
        x = x.replace("{{SIZE_K}}", str(config.k))
        x = x.replace("{{LDS_B_SHIFT}}", str(config.lds_a_size // config.num_waves))
        x = x.replace("{{LDS_SIZE}}", str(config.lds_total_size // config.num_waves))
        return x

    return preprocess


# =============================================================================
# Verification Functions
# =============================================================================


def make_gemm_verify_fn(
    config: GEMMConfig,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    scale_with_k: bool = False,
) -> Callable:
    """Create a verification function for GEMM kernels.

    Args:
        config: GEMM configuration
        rtol: Relative tolerance (default: 1e-4, or scaled if scale_with_k)
        atol: Absolute tolerance (default: 1e-4, or scaled if scale_with_k)
        scale_with_k: If True, scale tolerance with sqrt(k) for large problems

    Expects B in transposed layout (n, k) and output C as flat (m*n,).
    """
    m, n, k = config.m, config.n, config.k

    # Default tolerances
    if scale_with_k:
        # For benchmarks with large k: f16 has ~1e-3 precision, error accumulates
        mean = 0.75  # typical value range
        _rtol = rtol if rtol is not None else 1e-3 * np.sqrt(k)
        _atol = atol if atol is not None else 1e-3 * k * mean
    else:
        _rtol = rtol if rtol is not None else 1e-4
        _atol = atol if atol is not None else 1e-4

    def verify_fn(input_args: List[np.ndarray], output_args: List[np.ndarray]) -> None:
        a_flat = np.array(input_args[0])
        a = a_flat.reshape(m, k)
        b_flat = np.array(input_args[1])
        b = b_flat.reshape(n, k).T  # B stored as (n, k), need (k, n)
        c_flat = np.array(output_args[0], dtype=np.float32)
        c = c_flat.reshape(m, n)

        ref = np.matmul(a.astype(np.float32), b.astype(np.float32))

        if not np.allclose(c, ref, rtol=_rtol, atol=_atol):
            diff = np.abs(c.astype(np.float32) - ref)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            max_diff = diff[max_idx]
            rel_error = np.linalg.norm(diff) / np.linalg.norm(ref)
            raise AssertionError(
                f"GEMM kernel failed!\n"
                f"Max diff: {max_diff} at {max_idx}, c={c[max_idx]}, ref={ref[max_idx]}\n"
                f"Relative error: {rel_error}, rtol: {_rtol}, atol: {_atol}\n"
                f"c shape: {c.shape}, ref shape: {ref.shape}"
            )

    return verify_fn


def make_mfma_verify_fn(config: MFMAConfig) -> Callable:
    """Create a verification function for MFMA block-based kernels."""
    m, n, k = config.m, config.n, config.k
    batch = config.batch

    def verify_fn(input_args: List[np.ndarray], output_args: List[np.ndarray]) -> None:
        a_flat = np.array(input_args[0])
        a_blocks = a_flat.reshape(batch, m, k, MFMA_SIZE_M, MFMA_SIZE_K)

        b_flat = np.array(input_args[1])
        b_blocks = b_flat.reshape(batch, k, n, MFMA_SIZE_K, MFMA_SIZE_N)

        c_flat = np.array(output_args[0])
        c_blocks = c_flat.reshape(batch, m, n, MFMA_SIZE_M, MFMA_SIZE_N)

        # Compute reference using block matrix multiplication
        ref = np.zeros((batch, m, n, MFMA_SIZE_M, MFMA_SIZE_N), dtype=np.float32)
        for b_idx in range(batch):
            for i in range(m):
                for j in range(n):
                    for l in range(k):
                        a_block = a_blocks[b_idx, i, l]
                        b_block = b_blocks[b_idx, l, j]
                        ref[b_idx, i, j] += np.matmul(
                            a_block.astype(np.float32), b_block.astype(np.float32)
                        )

        if not np.allclose(c_blocks, ref, rtol=1e-5, atol=1e-5):
            diff = np.abs(c_blocks - ref)
            max_diff = np.max(diff)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            raise AssertionError(
                f"MFMA kernel failed! Max diff: {max_diff} at index {max_idx}\n"
                f"c shape: {c_blocks.shape}, ref shape: {ref.shape}\n"
                f"c_blocks:\n{c_blocks}\nref:\n{ref}"
            )

    return verify_fn


# =============================================================================
# Data Generation
# =============================================================================


def generate_gemm_data(
    config: GEMMConfig,
    random_data: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate input/output data for GEMM kernel.

    Args:
        config: GEMM configuration
        random_data: If True, use random values; if False, use well-conditioned values

    Returns:
        (a_data, b_data, c_data) arrays
    """
    dt_a, dt_b, dt_c = np.float16, np.float16, np.float32

    if random_data:
        a_data = np.random.randn(config.m, config.k).astype(dt_a)
        b_data = np.random.randn(config.k, config.n).astype(dt_b)
    else:
        # Well-conditioned values in [-1.0, -0.5] ∪ [0.5, 1.0]
        mean = 0.75
        a_data = (
            np.random.uniform(mean * 2 / 3, mean * 4 / 3, (config.m, config.k))
            * np.random.choice([-1, 1], (config.m, config.k))
        ).astype(dt_a)
        b_data = (
            np.random.uniform(mean * 2 / 3, mean * 4 / 3, (config.k, config.n))
            * np.random.choice([-1, 1], (config.k, config.n))
        ).astype(dt_b)

    c_data = np.zeros(config.m * config.n, dtype=dt_c)
    return a_data, b_data, c_data


def generate_mfma_data(
    config: MFMAConfig,
    a_val: float = 1.0,
    b_val: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate input/output data for MFMA kernel.

    Args:
        config: MFMA configuration
        a_val, b_val: Constant values to fill A and B matrices

    Returns:
        (a_data, b_data, c_data) arrays
    """
    dt_a, dt_b, dt_c = np.float16, np.float16, np.float32

    a_size = config.batch * config.m * config.k * MFMA_SIZE_M * MFMA_SIZE_K
    b_size = config.batch * config.k * config.n * MFMA_SIZE_K * MFMA_SIZE_N
    c_size = config.batch * config.m * config.n * MFMA_SIZE_M * MFMA_SIZE_N

    a_data = np.full(a_size, a_val, dtype=dt_a)
    b_data = np.full(b_size, b_val, dtype=dt_b)
    c_data = np.zeros(c_size, dtype=dt_c)

    return a_data, b_data, c_data
