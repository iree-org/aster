"""Unit tests for kittens FP8 GEMM kernel.

CRITICAL: CDNA3 (gfx942) uses FP8 E4M3FNUZ format (bias=8), NOT OCP E4M3 (bias=7).
Value = 2^(E-8) * (1 + M/8) for E>0, or 2^(-7) * (M/8) for E=0.
NaN = 0x80 (negative zero is NaN). No negative zero exists.

Reference: commit efc47ab4 ("Add support for CDNA3 FP8 MFMA (16x16x32) (#308)")
"""

import numpy as np
import pytest

from aster.pass_pipelines import TEST_SCF_PIPELINING_PASS_PIPELINE

from test_lib import run_kittens_kernel, get_mlir_file

# Test configuration
MCPU = "gfx942"


# ---------------------------------------------------------------------------
# FP8 E4M3FNUZ conversion utilities (bias=8, CDNA3)
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
        import math

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


# ---------------------------------------------------------------------------
# Host-side conversion tests (no GPU)
# ---------------------------------------------------------------------------


class TestFP8E4M3FNUZConversion:
    """Test FP8 E4M3FNUZ conversion utilities (host-side, no GPU)."""

    def test_known_values(self):
        """Key FNUZ constants should encode correctly."""
        # These are the exact values verified on CDNA3 hardware
        assert float_to_fp8_e4m3fnuz(np.array([1.0]))[0] == 0x40
        assert float_to_fp8_e4m3fnuz(np.array([1.5]))[0] == 0x44
        assert float_to_fp8_e4m3fnuz(np.array([2.0]))[0] == 0x48
        assert float_to_fp8_e4m3fnuz(np.array([0.5]))[0] == 0x38

    def test_roundtrip(self):
        """Values with exact FNUZ representations should roundtrip perfectly."""
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, -1.0, -2.0], dtype=np.float32)
        fp8 = float_to_fp8_e4m3fnuz(values)
        recovered = fp8_e4m3fnuz_to_float(fp8)
        np.testing.assert_allclose(recovered, values, rtol=0, atol=0)

    def test_zero(self):
        """Zero should encode as 0x00."""
        fp8 = float_to_fp8_e4m3fnuz(np.array([0.0], dtype=np.float32))
        assert fp8[0] == 0x00
        recovered = fp8_e4m3fnuz_to_float(fp8)
        assert recovered[0] == 0.0

    def test_nan(self):
        """NaN should encode as 0x80 (FNUZ: negative zero is NaN)."""
        fp8 = float_to_fp8_e4m3fnuz(np.array([np.nan], dtype=np.float32))
        assert fp8[0] == 0x80
        recovered = fp8_e4m3fnuz_to_float(fp8)
        assert np.isnan(recovered[0])

    def test_max_value(self):
        """Max FNUZ E4M3 value is 240.0 (same as OCP)."""
        fp8 = float_to_fp8_e4m3fnuz(np.array([240.0], dtype=np.float32))
        recovered = fp8_e4m3fnuz_to_float(fp8)
        assert recovered[0] == 240.0

    def test_clamp_overflow(self):
        """Values > 240 should clamp to 240."""
        fp8 = float_to_fp8_e4m3fnuz(np.array([500.0], dtype=np.float32))
        recovered = fp8_e4m3fnuz_to_float(fp8)
        assert recovered[0] == 240.0

    def test_negative(self):
        """Negative values should preserve sign."""
        fp8 = float_to_fp8_e4m3fnuz(np.array([-1.0], dtype=np.float32))
        recovered = fp8_e4m3fnuz_to_float(fp8)
        assert recovered[0] == -1.0

    def test_not_ocp(self):
        """Verify FNUZ encoding differs from OCP for key values.

        OCP E4M3 (bias=7): 1.0 = 0x38 FNUZ E4M3 (bias=8): 1.0 = 0x40
        """
        fp8_one = float_to_fp8_e4m3fnuz(np.array([1.0]))[0]
        assert fp8_one != 0x38, "Got OCP encoding (0x38) instead of FNUZ (0x40)"
        assert fp8_one == 0x40


# ---------------------------------------------------------------------------
# FP8 GEMM kernel tests (GPU)
# ---------------------------------------------------------------------------


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


class TestKittensGEMMFP8:
    """Test FP8 GEMM with scf.for K-loop: C = A @ B^T.

    Uses v_mfma_f32_16x16x32_fp8_fp8 with E4M3FNUZ format on CDNA3. K must be divisible
    by 32 (FP8 MFMA processes 32 elements per iteration).
    """

    @pytest.mark.parametrize("k", [64, 128, 256])
    def test_gemm_fp8_16x16xK(self, k):
        """FP8 GEMM should compute C = A @ B^T within tolerance vs reference."""
        A_fp8 = _make_fp8_inputs(16, k, seed=42 + k)
        B_fp8 = _make_fp8_inputs(16, k, seed=137 + k)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_fp8_16x16xK.mlir"),
            kernel_name="gemm_fp8_16x16xK",
            input_args=[A_fp8, B_fp8],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=_fp8_template_subs(k),
        )

        # Reference: dequantize FNUZ fp8 -> f32, then matmul
        A_ref = fp8_e4m3fnuz_to_float(A_fp8).reshape(16, k)
        B_ref = fp8_e4m3fnuz_to_float(B_fp8).reshape(16, k)
        expected = (A_ref @ B_ref.T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMMFP8_2Wave:
    """Test 2-wave FP8 GEMM: C[32x16] = A[32xK] @ B[16xK]^T.

    2x1 wave grid: wave 0 computes C[0:16, 0:16], wave 1 computes C[16:32, 0:16].
    Both waves share the same B matrix; each loads its own A rows.
    """

    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_fp8_2wave(self, k):
        """2-wave FP8 GEMM should compute C = A @ B^T correctly."""
        A_fp8 = _make_fp8_inputs(32, k, seed=42 + k)
        B_fp8 = _make_fp8_inputs(16, k, seed=137 + k)
        C_output = np.zeros(32 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_fp8_2wave.mlir"),
            kernel_name="gemm_fp8_2wave",
            input_args=[A_fp8, B_fp8],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(128, 1, 1),
            template_substitutions=_fp8_template_subs(k),
        )

        A_ref = fp8_e4m3fnuz_to_float(A_fp8).reshape(32, k)
        B_ref = fp8_e4m3fnuz_to_float(B_fp8).reshape(16, k)
        expected = (A_ref @ B_ref.T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMMFP8_4Wave:
    """Test 4-wave FP8 GEMM: C[32x32] = A[32xK] @ B[32xK]^T.

    2x2 wave grid:
      Wave 0: C[0:16, 0:16],   Wave 1: C[0:16, 16:32]
      Wave 2: C[16:32, 0:16],  Wave 3: C[16:32, 16:32]
    Each wave loads its own A rows and B rows based on grid position.
    """

    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_fp8_4wave(self, k):
        """4-wave FP8 GEMM should compute C = A @ B^T correctly."""
        A_fp8 = _make_fp8_inputs(32, k, seed=42 + k)
        B_fp8 = _make_fp8_inputs(32, k, seed=137 + k)
        C_output = np.zeros(32 * 32, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_fp8_4wave.mlir"),
            kernel_name="gemm_fp8_4wave",
            input_args=[A_fp8, B_fp8],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            block_dim=(256, 1, 1),
            template_substitutions=_fp8_template_subs(k),
        )

        A_ref = fp8_e4m3fnuz_to_float(A_fp8).reshape(32, k)
        B_ref = fp8_e4m3fnuz_to_float(B_fp8).reshape(32, k)
        expected = (A_ref @ B_ref.T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


class TestKittensGEMMFP8_LDS1Buf:
    """Test FP8 GEMM with single-buffer LDS: C = A @ B^T.

    Global -> LDS -> Register -> MFMA pipeline.
    Baseline LDS implementation to establish correctness (no latency hiding).
    """

    @pytest.mark.parametrize("k", [64, 128])
    def test_gemm_fp8_lds_1buf(self, k):
        """FP8 LDS GEMM should match reference."""
        A_fp8 = _make_fp8_inputs(16, k, seed=42 + k)
        B_fp8 = _make_fp8_inputs(16, k, seed=137 + k)
        C_output = np.zeros(16 * 16, dtype=np.float32)

        run_kittens_kernel(
            mlir_file=get_mlir_file("test_gemm_fp8_16x16xK_lds_1buf.mlir"),
            kernel_name="gemm_fp8_16x16xK_lds_1buf",
            input_args=[A_fp8, B_fp8],
            output_args=[C_output],
            pass_pipeline=TEST_SCF_PIPELINING_PASS_PIPELINE,
            template_substitutions=_fp8_template_subs(k),
        )

        A_ref = fp8_e4m3fnuz_to_float(A_fp8).reshape(16, k)
        B_ref = fp8_e4m3fnuz_to_float(B_fp8).reshape(16, k)
        expected = (A_ref @ B_ref.T).flatten()
        np.testing.assert_allclose(C_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":

    def run_test(name, test_fn, *args, **kwargs):
        """Run a test, handling pytest.skip gracefully."""
        print(f"\n--- {name} ---")
        try:
            test_fn(*args, **kwargs)
            print("  PASS")
        except pytest.skip.Exception as e:
            print(f"  SKIP: {e}")

    # Host-side conversion tests
    conv = TestFP8E4M3FNUZConversion()
    run_test("known_values", conv.test_known_values)
    run_test("roundtrip", conv.test_roundtrip)
    run_test("zero", conv.test_zero)
    run_test("nan", conv.test_nan)
    run_test("max_value", conv.test_max_value)
    run_test("clamp_overflow", conv.test_clamp_overflow)
    run_test("negative", conv.test_negative)
    run_test("not_ocp", conv.test_not_ocp)

    # GPU GEMM tests
    gemm = TestKittensGEMMFP8()
    run_test("gemm_fp8_k64", gemm.test_gemm_fp8_16x16xK, k=64)
    run_test("gemm_fp8_k128", gemm.test_gemm_fp8_16x16xK, k=128)
    run_test("gemm_fp8_k256", gemm.test_gemm_fp8_16x16xK, k=256)

    # 2-wave GPU GEMM tests
    gemm_2w = TestKittensGEMMFP8_2Wave()
    run_test("gemm_fp8_2wave_k64", gemm_2w.test_gemm_fp8_2wave, k=64)
    run_test("gemm_fp8_2wave_k128", gemm_2w.test_gemm_fp8_2wave, k=128)

    # 4-wave GPU GEMM tests
    gemm_4w = TestKittensGEMMFP8_4Wave()
    run_test("gemm_fp8_4wave_k64", gemm_4w.test_gemm_fp8_4wave, k=64)
    run_test("gemm_fp8_4wave_k128", gemm_4w.test_gemm_fp8_4wave, k=128)

    # LDS 1-buffer GPU GEMM tests
    gemm_lds = TestKittensGEMMFP8_LDS1Buf()
    run_test("gemm_fp8_lds_1buf_k64", gemm_lds.test_gemm_fp8_lds_1buf, k=64)
    run_test("gemm_fp8_lds_1buf_k128", gemm_lds.test_gemm_fp8_lds_1buf, k=128)
