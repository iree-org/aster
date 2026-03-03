"""Shared infrastructure for kittens test suite."""

import os
from dataclasses import dataclass
from typing import List

import math

import numpy as np
import pytest

from aster.testing import compile_and_run as _compile_and_run
from aster.pass_pipelines import (
    TEST_SCF_PIPELINING_PASS_PIPELINE,
    TEST_CONSTEXPR_PIPELINING_PASS_PIPELINE,
    FUTURE_SROA_PASS_PIPELINE,
)
from mlir_kernels.common import get_library_paths

# Test configuration
MCPU = "gfx942"
WAVEFRONT_SIZE = 64


def get_kittens_library_paths() -> List[str]:
    """Get paths to all required library files including kittens."""
    base_paths = get_library_paths()
    kittens_dir = os.path.join(os.path.dirname(__file__), "..", "library")
    kittens_paths = [
        os.path.join(kittens_dir, "global_16x16_f16.mlir"),
        os.path.join(kittens_dir, "lds_16x16_f16.mlir"),
        os.path.join(kittens_dir, "tiles_16x16_fp8.mlir"),
        os.path.join(kittens_dir, "lds_16x16_fp8.mlir"),
    ]
    return base_paths + kittens_paths


def get_mlir_file(file_name: str) -> str:
    """Get path to a test MLIR file in the kittens test directory."""
    return os.path.join(os.path.dirname(__file__), file_name)


def run_kittens_kernel(
    mlir_file,
    kernel_name,
    input_args=None,
    output_args=None,
    pass_pipeline=None,
    template_substitutions=None,
    grid_dim=(1, 1, 1),
    block_dim=(64, 1, 1),
    num_iterations=1,
    print_ir_after_all=False,
):
    """Compile an MLIR file to HSACO and execute the kernel on GPU."""
    preprocess = None
    if template_substitutions:
        subs = template_substitutions

        def preprocess(content):
            for pattern, replacement in subs.items():
                content = content.replace(pattern, replacement)
            return content

    return _compile_and_run(
        file_name=mlir_file,
        kernel_name=kernel_name,
        input_data=input_args,
        output_data=output_args,
        pass_pipeline=pass_pipeline,
        preprocess=preprocess,
        library_paths=get_kittens_library_paths(),
        mcpu=MCPU,
        wavefront_size=WAVEFRONT_SIZE,
        grid_dim=grid_dim,
        block_dim=block_dim,
        num_iterations=num_iterations,
        print_ir_after_all=print_ir_after_all,
    )


def _make_gemm_inputs(K):
    """Create random f16 test matrices for GEMM: A[16xK], B[16xK]."""
    np.random.seed(42)
    A = (np.random.randn(16, K) * 0.05).astype(np.float16)
    B = (np.random.randn(16, K) * 0.05).astype(np.float16)
    return A, B


PIPELINE_STAGE_CONFIGS = {
    # num_stages: (STAGE_LOAD, STAGE_SYNC, STAGE_COMPUTE)
    # Used by 3-stage templates (test_014 through test_019).
    2: (0, 1, 1),
    3: (0, 1, 2),
    4: (0, 2, 3),
    5: (0, 3, 4),
}

PIPELINE_STAGE_CONFIGS_4 = {
    # num_stages: (STAGE_GLOBAL_LOAD, STAGE_DS_WRITE, STAGE_DS_READ, STAGE_COMPUTE)
    # 4-stage split: separates global load from DS write for better pipelining.
    # For 2/3-stage, DS_WRITE == GLOBAL_LOAD (combined load + store can't split).
    # For 4+, all 4 stages are distinct.
    2: (0, 0, 1, 1),
    3: (0, 0, 1, 2),
    4: (0, 1, 2, 3),
    5: (0, 1, 3, 4),
}


def pipelined_substitutions(k, num_stages):
    """Build template substitutions for pipelined GEMM tests (xor_swizzle only)."""
    k_tiles = k // 16
    stride_ab = k * 2
    stage_load, stage_sync, stage_compute = PIPELINE_STAGE_CONFIGS[num_stages]
    return {
        "{{K}}": str(k),
        "{{K_TILES}}": str(k_tiles),
        "{{STRIDE_AB}}": str(stride_ab),
        "{{STAGE_LOAD}}": str(stage_load),
        "{{STAGE_SYNC}}": str(stage_sync),
        "{{STAGE_COMPUTE}}": str(stage_compute),
    }


# ---------------------------------------------------------------------------
# FP8 E4M3FNUZ conversion utilities (bias=8, CDNA3)
#
# CRITICAL: CDNA3 (gfx942) uses FP8 E4M3FNUZ format (bias=8), NOT OCP E4M3 (bias=7).
# Value = 2^(E-8) * (1 + M/8) for E>0, or 2^(-7) * (M/8) for E=0.
# NaN = 0x80 (negative zero is NaN). No negative zero exists.
# Reference: commit efc47ab4 ("Add support for CDNA3 FP8 MFMA (16x16x32) (#308)")
# ---------------------------------------------------------------------------


def float_to_fp8_e4m3fnuz(values: np.ndarray) -> np.ndarray:
    """Convert float32 values to FP8 E4M3FNUZ format (uint8).

    FP8 E4M3FNUZ (AMD CDNA3):
      - 1 sign + 4 exponent + 3 mantissa, exponent bias = 8
      - Normal: (-1)^S * 2^(E-8) * (1 + M/8) for E in [1..15]
      - Subnormal: (-1)^S * 2^(-7) * (M/8) for E=0, M>0
      - Zero: 0x00
      - NaN: 0x80 (negative zero is NaN, not -0)
      - Max representable: 2^7 * (1 + 7/8) = 240.0
    """
    f32 = values.astype(np.float32).flatten()
    result = np.zeros(len(f32), dtype=np.uint8)

    for i, val in enumerate(f32):
        if np.isnan(val):
            result[i] = 0x80  # FNUZ NaN
            continue

        sign = 0
        if val < 0:
            sign = 1
            val = -val

        if val == 0.0:
            # FNUZ: +0 is 0x00, -0 is NaN (0x80). Treat -0 as +0.
            result[i] = 0x00
            continue

        # Clamp to max representable: 2^7 * (1 + 7/8) = 240.0
        val = min(val, 240.0)

        # FNUZ E4M3: bias = 8
        # Normal range: E in [1..15] -> real exponent [-7..7]
        # Min normal: 2^(-7) * 1.0 = 2^(-7) ~ 0.0078125
        # Subnormal: E=0, value = 2^(-7) * (M/8) for M in [1..7]
        # Min subnormal: 2^(-7) * (1/8) = 2^(-10) ~ 0.000977
        exp = int(math.floor(math.log2(val)))

        if exp < -10:
            # Too small, round to zero
            result[i] = 0x00
        elif exp < -7:
            # Subnormal: E=0, value = 2^(-7) * (M/8)
            m = int(round(val / (2.0**-7) * 8.0))
            m = max(0, min(m, 7))
            if m == 0:
                result[i] = 0x00
            else:
                result[i] = (sign << 7) | m
        else:
            biased_exp = exp + 8  # bias = 8
            frac = val / (2.0**exp) - 1.0
            m = int(round(frac * 8.0))
            if m >= 8:
                m = 0
                biased_exp += 1
            if biased_exp > 15:
                biased_exp = 15
                m = 7
            if biased_exp < 1:
                # Should not happen given exp >= -7, but clamp
                biased_exp = 1
                m = 0
            result[i] = (sign << 7) | (biased_exp << 3) | m

    return result


def fp8_e4m3fnuz_to_float(values: np.ndarray) -> np.ndarray:
    """Convert FP8 E4M3FNUZ (uint8) back to float32.

    FNUZ format (bias=8):
      - 0x80 = NaN
      - E=0: subnormal, value = (-1)^S * 2^(-7) * (M/8)
      - E>0: normal, value = (-1)^S * 2^(E-8) * (1 + M/8)
    """
    flat = values.flatten()
    result = np.zeros(len(flat), dtype=np.float32)

    for i, byte in enumerate(flat):
        if byte == 0x80:
            result[i] = np.nan
            continue

        sign = int((byte >> 7) & 1)
        exp = int((byte >> 3) & 0xF)
        mantissa = int(byte & 0x7)
        sign_mul = -1.0 if sign else 1.0

        if exp == 0:
            # Subnormal: value = (-1)^S * 2^(-7) * (M/8)
            result[i] = sign_mul * (2.0**-7) * (mantissa / 8.0)
        else:
            # Normal: value = (-1)^S * 2^(E-8) * (1 + M/8)
            result[i] = sign_mul * (2.0 ** (exp - 8)) * (1.0 + mantissa / 8.0)

    return result


def _fp8_template_subs(k):
    """Build template substitutions for FP8 GEMM kernels."""
    assert k % 32 == 0, f"K must be divisible by 32, got {k}"
    return {
        "{{K}}": str(k),
        "{{K_TILES}}": str(k // 32),
        "{{STRIDE_AB}}": str(k * 1),  # 1 byte per fp8 element
    }


def _make_fp8_inputs(M, K, seed=42):
    """Create random FP8 test matrices: A[MxK], B quantized to FNUZ."""
    np.random.seed(seed)
    A_f32 = (np.random.randn(M, K) * 0.5).astype(np.float32)
    A_fp8 = float_to_fp8_e4m3fnuz(A_f32)
    return A_fp8


def constexpr_substitutions(m_tiles, n_tiles, k, num_stages):
    """Build scalar-only template substitutions for constexpr multi-tile GEMM.

    The template (test_gemm_constexpr.mlir) uses only scalar substitutions.
    All structural complexity is handled by the compiler pipeline:
      constexpr-expansion -> sroa -> mem2reg -> promote-loop-carried-memrefs
    """
    mn = m_tiles * n_tiles
    k_tiles = k // 16
    stride_ab = k * 2
    stride_c = n_tiles * 16 * 4
    shared_mem = (m_tiles + n_tiles) * 512
    stage_load, stage_sync, stage_compute = PIPELINE_STAGE_CONFIGS[num_stages]
    stage_gl, stage_dw, stage_dr, stage_c = PIPELINE_STAGE_CONFIGS_4[num_stages]

    return {
        "{{M_T}}": str(m_tiles),
        "{{N_T}}": str(n_tiles),
        "{{MN}}": str(mn),
        "{{M_DIM}}": str(m_tiles * 16),
        "{{N_DIM}}": str(n_tiles * 16),
        "{{K}}": str(k),
        "{{K_TILES}}": str(k_tiles),
        "{{STRIDE_AB}}": str(stride_ab),
        "{{STRIDE_C}}": str(stride_c),
        "{{SHARED_MEM}}": str(shared_mem),
        # 3-stage names (backward compat with test_019)
        "{{STAGE_LOAD}}": str(stage_load),
        "{{STAGE_SYNC}}": str(stage_sync),
        # 4-stage names (perf_001 split template)
        "{{STAGE_GLOBAL_LOAD}}": str(stage_gl),
        "{{STAGE_DS_WRITE}}": str(stage_dw),
        "{{STAGE_DS_READ}}": str(stage_dr),
        "{{STAGE_COMPUTE}}": str(stage_c),
    }
