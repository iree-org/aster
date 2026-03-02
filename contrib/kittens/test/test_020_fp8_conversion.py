"""Test: FP8 E4M3FNUZ conversion utilities (host-side, no GPU).

CRITICAL: CDNA3 (gfx942) uses FP8 E4M3FNUZ format (bias=8), NOT OCP E4M3 (bias=7).
Reference: commit efc47ab4 ("Add support for CDNA3 FP8 MFMA (16x16x32) (#308)")
"""

import numpy as np

from kittens_helpers import float_to_fp8_e4m3fnuz, fp8_e4m3fnuz_to_float


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
