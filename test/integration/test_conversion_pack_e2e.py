"""End-to-end tests for conversion and pack operations.

Tests VOP1 type conversions (f16<->f32, i32<->f32, u32<->f32) and VOP3 pack operations
(v_pack_b32_f16, v_cvt_pk_fp8_f32, v_cvt_pk_bf8_f32) by running GPU kernels and
verifying output bytes in Python.

Parametrized over multiple GPU targets (gfx942/cdna3, gfx950/cdna4).
"""

import re
import struct
import numpy as np
import pytest

from aster.testing import compile_and_run
from aster.pass_pipelines import DEFAULT_SROA_PASS_PIPELINE

TARGET_CONFIGS = [
    ("gfx942", "cdna3"),
    ("gfx950", "cdna4"),
]

WAVEFRONT_SIZE = 64
TOTAL_LANES = 64
MLIR_FILE = "conversion-pack-e2e.mlir"


def _retarget(target, isa):
    """Return a preprocess function that rewrites the MLIR target and ISA."""

    def preprocess(mlir_src):
        s = re.sub(r"#amdgcn\.target<\w+>", f"#amdgcn.target<{target}>", mlir_src)
        s = re.sub(r"#amdgcn\.isa<\w+>", f"#amdgcn.isa<{isa}>", s)
        return s

    return preprocess


def _run(mcpu, isa, kernel_name, input_data, output_data, verify_fn):
    compile_and_run(
        MLIR_FILE,
        kernel_name,
        input_data=input_data,
        output_data=output_data,
        pass_pipeline=DEFAULT_SROA_PASS_PIPELINE,
        mcpu=mcpu,
        wavefront_size=WAVEFRONT_SIZE,
        block_dim=(TOTAL_LANES, 1, 1),
        verify_fn=verify_fn,
        library_paths=[],
        skip_on_cross_compile=True,
        preprocess=_retarget(mcpu, isa),
    )


# ---------------------------------------------------------------------------
# Helpers for f16 encoding/decoding via struct
# ---------------------------------------------------------------------------


def f32_to_f16_bits(f32_val):
    """Convert a float32 value to its IEEE 754 half-precision bit pattern (uint16)."""
    return int(np.float16(f32_val).view(np.uint16))


def f16_bits_to_f32(bits):
    """Convert IEEE 754 half-precision bits (uint16) to a float32 value."""
    return float(np.uint16(bits).view(np.float16))


def f32_to_bits(f32_val):
    """Convert float32 to its IEEE 754 bit pattern as uint32."""
    return struct.unpack("<I", struct.pack("<f", f32_val))[0]


def bits_to_f32(bits):
    """Convert uint32 IEEE 754 bits to float32."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# ---------------------------------------------------------------------------
# FP8 (E4M3) and BF8 (E5M2) helpers
#
# CDNA3 (gfx942): FNUZ variant - bias is +1 vs OCP, NaN=0x80, no neg zero, no inf
# CDNA4 (gfx950): OCP variant  - standard bias, has neg zero and inf/NaN encodings
#
# FP8 E4M3: 1 sign + 4 exp + 3 mantissa, bias=7 (OCP) or 8 (FNUZ)
# BF8 E5M2: 1 sign + 5 exp + 2 mantissa, bias=15 (OCP) or 16 (FNUZ)
# ---------------------------------------------------------------------------

_ISA_IS_FNUZ = {"cdna3": True, "cdna4": False}


def f32_to_fp8_e4m3(val, fnuz=True):
    """Reference conversion from f32 to FP8 E4M3 (round-to-nearest-even).

    Args:
        val: float32 input value
        fnuz: True for FNUZ (CDNA3, bias=8), False for OCP (CDNA4, bias=7)
    """
    bias = 8 if fnuz else 7
    max_stored_exp = 15  # 4-bit exponent, all normal
    # E4M3 FNUZ max: 2^(15-8)*1.875=240, OCP max: 2^(15-7)*1.75=448 (exp=15 reserved for NaN in OCP)
    if fnuz:
        max_magnitude = 240.0
    else:
        max_magnitude = 448.0

    bits = f32_to_bits(float(val))
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0xFF:
        if fnuz:
            return 0x80  # NaN (all specials map here)
        else:
            # OCP E4M3: NaN = S_1111_111, no inf
            return (sign << 7) | 0x7F

    magnitude = abs(float(val))

    if magnitude == 0.0:
        return 0x00 if fnuz else (sign << 7)  # FNUZ has no negative zero

    if magnitude > max_magnitude:
        if fnuz:
            return (sign << 7) | 0x7F
        else:
            return (sign << 7) | 0x7E  # OCP: saturate to max (0x7F is NaN)

    f32_exp = exp - 127
    fp8_exp = f32_exp + bias

    if fp8_exp <= 0:
        shift = 1 - fp8_exp
        full_mantissa = (1 << 23) | frac
        total_shift = 20 + shift
        if total_shift >= 24:
            return 0x00 if fnuz else (sign << 7)
        rounded = (full_mantissa + (1 << (total_shift - 1))) >> total_shift
        if rounded > 7:
            rounded = 7
        if rounded == 0:
            return 0x00 if fnuz else (sign << 7)
        return (sign << 7) | rounded
    elif fp8_exp > max_stored_exp:
        if fnuz:
            return (sign << 7) | 0x7F
        else:
            return (sign << 7) | 0x7E
    else:
        round_bit = 1 << 19
        guard_bit = frac & round_bit
        remainder = frac & (round_bit - 1)
        mantissa_3 = frac >> 20

        if guard_bit:
            if remainder > 0 or (mantissa_3 & 1):
                mantissa_3 += 1
                if mantissa_3 > 7:
                    mantissa_3 = 0
                    fp8_exp += 1
                    if fp8_exp > max_stored_exp:
                        if fnuz:
                            return (sign << 7) | 0x7F
                        else:
                            return (sign << 7) | 0x7E

        return (sign << 7) | (fp8_exp << 3) | mantissa_3


def f32_to_bf8_e5m2(val, fnuz=True):
    """Reference conversion from f32 to BF8 E5M2 (round-to-nearest-even).

    Args:
        val: float32 input value
        fnuz: True for FNUZ (CDNA3, bias=16), False for OCP (CDNA4, bias=15)
    """
    bias = 16 if fnuz else 15
    if fnuz:
        max_stored_exp = 31  # all exponents are normal/subnormal
        max_magnitude = 57344.0  # 2^(31-16)*1.75
    else:
        max_stored_exp = 30  # exp=31 reserved for inf/NaN in OCP
        max_magnitude = 57344.0  # 2^(30-15)*1.75 = same max

    bits = f32_to_bits(float(val))
    sign = (bits >> 31) & 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF

    if exp == 0xFF:
        if fnuz:
            return 0x80
        else:
            if frac:
                return (sign << 7) | 0x7F  # NaN
            else:
                return (sign << 7) | 0x7C  # inf

    magnitude = abs(float(val))

    if magnitude == 0.0:
        return 0x00 if fnuz else (sign << 7)

    if magnitude > max_magnitude:
        if fnuz:
            return (sign << 7) | 0x7F
        else:
            return (sign << 7) | 0x7C  # OCP: overflow to inf

    f32_exp = exp - 127
    bf8_exp = f32_exp + bias

    if bf8_exp <= 0:
        shift = 1 - bf8_exp
        full_mantissa = (1 << 23) | frac
        total_shift = 21 + shift
        if total_shift >= 24:
            return 0x00 if fnuz else (sign << 7)
        rounded = (full_mantissa + (1 << (total_shift - 1))) >> total_shift
        if rounded > 3:
            rounded = 3
        if rounded == 0:
            return 0x00 if fnuz else (sign << 7)
        return (sign << 7) | rounded
    elif bf8_exp > max_stored_exp:
        if fnuz:
            return (sign << 7) | 0x7F
        else:
            return (sign << 7) | 0x7C
    else:
        round_bit = 1 << 20
        guard_bit = frac & round_bit
        remainder = frac & (round_bit - 1)
        mantissa_2 = frac >> 21

        if guard_bit:
            if remainder > 0 or (mantissa_2 & 1):
                mantissa_2 += 1
                if mantissa_2 > 3:
                    mantissa_2 = 0
                    bf8_exp += 1
                    if bf8_exp > max_stored_exp:
                        if fnuz:
                            return (sign << 7) | 0x7F
                        else:
                            return (sign << 7) | 0x7C

        return (sign << 7) | (bf8_exp << 2) | mantissa_2


# ===========================================================================
# VOP1 Conversion Tests
# ===========================================================================


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtF32F16:
    """v_cvt_f32_f16: read f16 bits from low 16 of dword, produce f32."""

    def test_basic_values(self, mcpu, isa):
        # Store f16 bit patterns in the low 16 bits of each dword.
        # Values: 0.0, 1.0, -1.0, 0.5, 65504.0 (max f16), tiny subnormal
        test_f32_vals = [0.0, 1.0, -1.0, 0.5, 65504.0, 5.96046448e-08]
        src_dwords = np.zeros(TOTAL_LANES, dtype=np.uint32)
        for i, v in enumerate(test_f32_vals):
            src_dwords[i] = f32_to_f16_bits(v)
        # Fill rest with lane index as f16
        for i in range(len(test_f32_vals), TOTAL_LANES):
            src_dwords[i] = f32_to_f16_bits(float(i))

        src = src_dwords.view(np.int32)
        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_f32 = outputs[0].view(np.float32)
            for i in range(TOTAL_LANES):
                expected = f16_bits_to_f32(src_dwords[i] & 0xFFFF)
                assert result_f32[i] == np.float32(
                    expected
                ), f"lane {i}: expected {expected}, got {result_f32[i]}"

        _run(mcpu, isa, "cvt_f32_f16_kernel", [src], [dst], verify)


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtF16F32:
    """v_cvt_f16_f32: read f32, produce f16 bits in low 16 of dword."""

    def test_basic_values(self, mcpu, isa):
        test_vals = [0.0, 1.0, -1.0, 0.5, 65504.0, 0.333251953125]
        src = np.zeros(TOTAL_LANES, dtype=np.float32)
        for i, v in enumerate(test_vals):
            src[i] = v
        for i in range(len(test_vals), TOTAL_LANES):
            src[i] = float(i)

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_bits = outputs[0].view(np.uint32)
            src_f32 = inputs[0].view(np.float32)
            for i in range(TOTAL_LANES):
                expected_bits = f32_to_f16_bits(src_f32[i])
                actual_low16 = result_bits[i] & 0xFFFF
                assert actual_low16 == expected_bits, (
                    f"lane {i}: f32={src_f32[i]}, "
                    f"expected f16 bits 0x{expected_bits:04X}, "
                    f"got 0x{actual_low16:04X}"
                )

        _run(mcpu, isa, "cvt_f16_f32_kernel", [src.view(np.int32)], [dst], verify)


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtF32U32:
    """v_cvt_f32_u32: unsigned int -> f32."""

    def test_basic_values(self, mcpu, isa):
        src = np.zeros(TOTAL_LANES, dtype=np.uint32)
        for i in range(TOTAL_LANES):
            src[i] = i * 1000  # 0, 1000, 2000, ...
        src[0] = 0
        src[1] = 1
        src[2] = 0xFFFFFFFF  # max u32
        src[3] = 0x7FFFFFFF  # max i32 as unsigned

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_f32 = outputs[0].view(np.float32)
            src_u32 = inputs[0].view(np.uint32)
            for i in range(TOTAL_LANES):
                expected = np.float32(np.float64(src_u32[i]))
                assert result_f32[i] == expected, (
                    f"lane {i}: u32={src_u32[i]}, expected f32={expected}, "
                    f"got {result_f32[i]}"
                )

        _run(mcpu, isa, "cvt_f32_u32_kernel", [src.view(np.int32)], [dst], verify)


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtF32I32:
    """v_cvt_f32_i32: signed int -> f32."""

    def test_basic_values(self, mcpu, isa):
        src = np.zeros(TOTAL_LANES, dtype=np.int32)
        for i in range(TOTAL_LANES):
            src[i] = i * 100 - 3200  # -3200, -3100, ..., 3100
        src[0] = 0
        src[1] = 1
        src[2] = -1
        src[3] = np.iinfo(np.int32).max
        src[4] = np.iinfo(np.int32).min

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_f32 = outputs[0].view(np.float32)
            src_i32 = inputs[0]
            for i in range(TOTAL_LANES):
                expected = np.float32(np.float64(src_i32[i]))
                assert result_f32[i] == expected, (
                    f"lane {i}: i32={src_i32[i]}, expected f32={expected}, "
                    f"got {result_f32[i]}"
                )

        _run(mcpu, isa, "cvt_f32_i32_kernel", [src], [dst], verify)


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtU32F32:
    """v_cvt_u32_f32: f32 -> unsigned int (truncate toward zero, clamp to [0, 2^32-1])."""

    def test_basic_values(self, mcpu, isa):
        src = np.zeros(TOTAL_LANES, dtype=np.float32)
        for i in range(TOTAL_LANES):
            src[i] = float(i * 1000)
        src[0] = 0.0
        src[1] = 1.0
        src[2] = 1.5  # truncates to 1
        src[3] = 255.9  # truncates to 255
        src[4] = 4294967040.0  # largest f32 <= 2^32-1

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_u32 = outputs[0].view(np.uint32)
            src_f32 = inputs[0].view(np.float32)
            for i in range(TOTAL_LANES):
                val = src_f32[i]
                # AMD hardware: truncate toward zero, clamp to [0, 0xFFFFFFFF]
                if val != val or val < 0.0:  # NaN or negative
                    expected = np.uint32(0)
                elif val >= 4294967296.0:
                    expected = np.uint32(0xFFFFFFFF)
                else:
                    expected = np.uint32(int(val))
                assert result_u32[i] == expected, (
                    f"lane {i}: f32={val}, expected u32={expected}, "
                    f"got {result_u32[i]}"
                )

        _run(mcpu, isa, "cvt_u32_f32_kernel", [src.view(np.int32)], [dst], verify)


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtI32F32:
    """v_cvt_i32_f32: f32 -> signed int (truncate toward zero, clamp)."""

    def test_basic_values(self, mcpu, isa):
        src = np.zeros(TOTAL_LANES, dtype=np.float32)
        for i in range(TOTAL_LANES):
            src[i] = float(i * 100 - 3200)
        src[0] = 0.0
        src[1] = 1.0
        src[2] = -1.0
        src[3] = 1.7  # truncates to 1
        src[4] = -1.7  # truncates to -1
        src[5] = 2147483520.0  # largest f32 < 2^31

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_i32 = outputs[0]
            src_f32 = inputs[0].view(np.float32)
            for i in range(TOTAL_LANES):
                val = src_f32[i]
                if val != val:  # NaN
                    expected = np.int32(0)
                elif val >= 2147483648.0:
                    expected = np.int32(2147483647)
                elif val <= -2147483649.0:
                    expected = np.int32(-2147483648)
                else:
                    expected = np.int32(int(val))
                assert result_i32[i] == expected, (
                    f"lane {i}: f32={val}, expected i32={expected}, "
                    f"got {result_i32[i]}"
                )

        _run(mcpu, isa, "cvt_i32_f32_kernel", [src.view(np.int32)], [dst], verify)


# ===========================================================================
# VOP3 Pack Tests
# ===========================================================================


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestPackB32F16:
    """v_pack_b32_f16: pack two f16 values (in low 16 bits of dwords) into one b32.

    result[15:0] = src0[15:0], result[31:16] = src1[15:0]
    """

    def test_basic_values(self, mcpu, isa):
        # src0 and src1 contain f16 bit patterns in low 16 bits of each dword
        src0 = np.zeros(TOTAL_LANES, dtype=np.uint32)
        src1 = np.zeros(TOTAL_LANES, dtype=np.uint32)
        for i in range(TOTAL_LANES):
            src0[i] = f32_to_f16_bits(float(i))  # f16(0), f16(1), ...
            src1[i] = f32_to_f16_bits(float(i + 100))  # f16(100), f16(101), ...

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_u32 = outputs[0].view(np.uint32)
            s0 = inputs[0].view(np.uint32)
            s1 = inputs[1].view(np.uint32)
            for i in range(TOTAL_LANES):
                lo = s0[i] & 0xFFFF
                hi = s1[i] & 0xFFFF
                expected = lo | (hi << 16)
                assert result_u32[i] == expected, (
                    f"lane {i}: expected 0x{expected:08X}, got 0x{result_u32[i]:08X} "
                    f"(src0=0x{s0[i]:08X}, src1=0x{s1[i]:08X})"
                )

        _run(
            mcpu,
            isa,
            "pack_b32_f16_kernel",
            [src0.view(np.int32), src1.view(np.int32)],
            [dst],
            verify,
        )


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtPkFp8F32:
    """v_cvt_pk_fp8_f32: pack-convert two f32 to two fp8 (E4M3) in low 16 bits.

    CDNA3: E4M3 FNUZ (bias=8), CDNA4: E4M3 OCP (bias=7).
    result[7:0] = fp8(src0), result[15:8] = fp8(src1)
    """

    def test_basic_values(self, mcpu, isa):
        fnuz = _ISA_IS_FNUZ[isa]
        src0 = np.zeros(TOTAL_LANES, dtype=np.float32)
        src1 = np.zeros(TOTAL_LANES, dtype=np.float32)
        for i in range(TOTAL_LANES):
            src0[i] = float(i)
            src1[i] = float(i) * 0.5

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_u32 = outputs[0].view(np.uint32)
            s0 = inputs[0].view(np.float32)
            s1 = inputs[1].view(np.float32)
            for i in range(TOTAL_LANES):
                byte0 = result_u32[i] & 0xFF
                byte1 = (result_u32[i] >> 8) & 0xFF
                expected0 = f32_to_fp8_e4m3(s0[i], fnuz=fnuz)
                expected1 = f32_to_fp8_e4m3(s1[i], fnuz=fnuz)
                assert byte0 == expected0, (
                    f"lane {i} byte0: f32={s0[i]}, "
                    f"expected fp8=0x{expected0:02X}, got 0x{byte0:02X}"
                )
                assert byte1 == expected1, (
                    f"lane {i} byte1: f32={s1[i]}, "
                    f"expected fp8=0x{expected1:02X}, got 0x{byte1:02X}"
                )

        _run(
            mcpu,
            isa,
            "cvt_pk_fp8_f32_kernel",
            [src0.view(np.int32), src1.view(np.int32)],
            [dst],
            verify,
        )


@pytest.mark.parametrize("mcpu,isa", TARGET_CONFIGS, ids=[t for t, _ in TARGET_CONFIGS])
class TestCvtPkBf8F32:
    """v_cvt_pk_bf8_f32: pack-convert two f32 to two bf8 (E5M2) in low 16 bits.

    CDNA3: E5M2 FNUZ (bias=16), CDNA4: E5M2 OCP (bias=15).
    result[7:0] = bf8(src0), result[15:8] = bf8(src1)
    """

    def test_basic_values(self, mcpu, isa):
        fnuz = _ISA_IS_FNUZ[isa]
        src0 = np.zeros(TOTAL_LANES, dtype=np.float32)
        src1 = np.zeros(TOTAL_LANES, dtype=np.float32)
        for i in range(TOTAL_LANES):
            src0[i] = float(i)
            src1[i] = float(i) * 0.5

        dst = np.zeros(TOTAL_LANES, dtype=np.int32)

        def verify(inputs, outputs):
            result_u32 = outputs[0].view(np.uint32)
            s0 = inputs[0].view(np.float32)
            s1 = inputs[1].view(np.float32)
            for i in range(TOTAL_LANES):
                byte0 = result_u32[i] & 0xFF
                byte1 = (result_u32[i] >> 8) & 0xFF
                expected0 = f32_to_bf8_e5m2(s0[i], fnuz=fnuz)
                expected1 = f32_to_bf8_e5m2(s1[i], fnuz=fnuz)
                assert byte0 == expected0, (
                    f"lane {i} byte0: f32={s0[i]}, "
                    f"expected bf8=0x{expected0:02X}, got 0x{byte0:02X}"
                )
                assert byte1 == expected1, (
                    f"lane {i} byte1: f32={s1[i]}, "
                    f"expected bf8=0x{expected1:02X}, got 0x{byte1:02X}"
                )

        _run(
            mcpu,
            isa,
            "cvt_pk_bf8_f32_kernel",
            [src0.view(np.int32), src1.view(np.int32)],
            [dst],
            verify,
        )
