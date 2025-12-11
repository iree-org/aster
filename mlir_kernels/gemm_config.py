"""Shared GEMM configuration validation for dword4 16x16x16 f16f16f32 kernels."""

from typing import Tuple, Optional


def is_power_of_two(n: int) -> bool:
    """Check if a number is a power of two."""
    return n > 0 and (n & (n - 1)) == 0


def validate_gemm_config(
    m: int,
    n: int,
    k: int,
    m_tile: int,
    n_tile: int,
    k_tile: int,
    num_waves: int,
) -> Tuple[bool, Optional[str]]:
    """Validate GEMM configuration for dword4 16x16x16 f16f16f32 kernels.

    Returns:
        (is_valid, error_reason) - error_reason is None if valid
    """
    # Problem size must be divisible by tile size
    if m % m_tile != 0:
        return False, "M must be a multiple of m_tile"
    if n % n_tile != 0:
        return False, "N must be a multiple of n_tile"
    if k % k_tile != 0:
        return False, "K must be a multiple of k_tile"

    # Tile sizes must be multiples of 16 (MFMA operation size)
    if m_tile % 16 != 0:
        return False, "m_tile must be a multiple of 16"
    if n_tile % 16 != 0:
        return False, "n_tile must be a multiple of 16"
    if k_tile % 16 != 0:
        return False, "k_tile must be a multiple of 16"

    # Minimum tile sizes
    if m_tile < 16:
        return False, "m_tile must be at least 16"
    if n_tile < 16:
        return False, "n_tile must be at least 16"
    if k_tile < 16:
        return False, "k_tile must be at least 16"

    # Tile size / 16 must be power of 2 (for efficient indexing)
    if not is_power_of_two(m_tile // 16):
        return False, "m_tile / 16 must be a power of 2"
    if not is_power_of_two(n_tile // 16):
        return False, "n_tile / 16 must be a power of 2"
    if not is_power_of_two(k_tile // 16):
        return False, "k_tile / 16 must be a power of 2"

    # Wave constraints
    if num_waves <= 0:
        return False, "Number of waves must be positive"
    if not is_power_of_two(num_waves):
        return False, "Number of waves must be a power of 2"

    # Work distribution: MFMA blocks must be evenly divisible by waves
    if (m_tile // 16) * (n_tile // 16) % num_waves != 0:
        return False, "m_tile/16 * n_tile/16 must be divisible by num_waves"
    if (m_tile // 16) * (k_tile // 16) % num_waves != 0:
        return False, "m_tile/16 * k_tile/16 must be divisible by num_waves"
    if (k_tile // 16) * (n_tile // 16) % num_waves != 0:
        return False, "k_tile/16 * n_tile/16 must be divisible by num_waves"

    return True, None
