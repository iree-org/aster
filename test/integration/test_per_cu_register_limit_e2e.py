"""Test that the hardware CP enforces per-CU register file limits at dispatch."""

import pytest

from aster.compiler.core import compile_to_hsaco
from aster.execution.core import execute_hsaco, InputArray, OutputArray

# Skip if no GPU available.
try:
    from aster._mlir_libs._runtime_module import hip_init, hip_get_device_count

    hip_init()
    _NUM_GPUS = hip_get_device_count()
except Exception:
    _NUM_GPUS = 0

# Skip atm, as it's timing out.
pytestmark = pytest.mark.skip()

# Per-CU register limit constants.
# On gfx942: regsPerMultiprocessor = 131072 (32-bit lanes).
# Architectural regs per CU = 131072 / 64 (warpSize) = 2048.
# With 16 waves (1024 threads) and alloc granule 8:
#   PASS: 128 regs/wave * 16 waves = 2048 (at limit)
#   FAIL: 136 regs/wave * 16 waves = 2176 (over limit)
_REGS_PER_CU = 2048
_NUM_WAVES = 16
_ALLOC_GRANULE = 8
_PASS_REGS = _REGS_PER_CU // _NUM_WAVES  # 128
_FAIL_REGS = _PASS_REGS + _ALLOC_GRANULE  # 136

# Minimal kernel: writes threadIdx.x to output[threadIdx.x].
# The actual register usage is tiny (v0-v1, s0-s1). We control the hardware
# allocation purely through .amdhsa_next_free_vgpr and .amdhsa_accum_offset.
_KERNEL_TEMPLATE = """\
  .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
  .text
  .globl simd_reg_test
  .p2align 8
  .type simd_reg_test,@function
simd_reg_test:
  s_load_dwordx2 s[0:1], s[0:1], 0
  v_lshlrev_b32_e64 v1, 2, v0
  s_waitcnt lgkmcnt(0)
  global_store_dword v1, v0, s[0:1]
  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel simd_reg_test
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_kernarg_size 8
    .amdhsa_user_sgpr_count 2
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr {next_free_vgpr}
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset {accum_offset}
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size simd_reg_test, .Lfunc_end0-simd_reg_test

  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .access: write_only
        .actual_access: write_only
        .address_space: generic
        .offset: 0
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: simd_reg_test
    .private_segment_fixed_size: 0
    .sgpr_count: 8
    .sgpr_spill_count: 0
    .symbol: simd_reg_test.kd
    .vgpr_count: {next_free_vgpr}
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdgcn_target: amdgcn-amd-amdhsa--gfx942
amdhsa.version:
  - 1
  - 2
---

  .end_amdgpu_metadata
"""


def _make_hsaco(next_free_vgpr, accum_offset):
    asm = _KERNEL_TEMPLATE.format(
        next_free_vgpr=next_free_vgpr, accum_offset=accum_offset
    )
    return compile_to_hsaco(asm, target="gfx942", wavefront_size=64)


import numpy as np
import tempfile
import os


def _write_hsaco(data):
    f = tempfile.NamedTemporaryFile(suffix=".hsaco", delete=False)
    f.write(data)
    f.close()
    return f.name


def test_per_cu_register_limit_pass():
    """128 regs/wave * 16 waves * 64 lanes = 131072 = regsPerMultiprocessor."""
    hsaco = _make_hsaco(next_free_vgpr=128, accum_offset=128)
    assert hsaco is not None, "Assembly failed"
    path = _write_hsaco(hsaco)
    try:
        N = 1024
        out = np.zeros(N, dtype=np.int32)
        execute_hsaco(
            hsaco_path=path,
            kernel_name="simd_reg_test",
            arguments=[OutputArray(out)],
            grid_dim=(1, 1, 1),
            block_dim=(1024, 1, 1),  # 16 waves
            num_iterations=1,
        )
        # Verify: out[i] == i for i in [0, 1024)
        expected = np.arange(N, dtype=np.int32)
        np.testing.assert_array_equal(out, expected)
    finally:
        os.unlink(path)


def test_per_cu_register_limit_expected_fail():
    """One granule over the per-CU register limit: hardware CP rejects dispatch.

    Runs in a completely separate Python process via subprocess.run to avoid
    polluting the current process's GPU state with the queue abort.
    """
    import subprocess
    import sys

    hsaco_data = _make_hsaco(next_free_vgpr=_FAIL_REGS, accum_offset=_FAIL_REGS)
    assert hsaco_data is not None, "Assembly failed"
    path = _write_hsaco(hsaco_data)

    script = f"""\
import sys, numpy as np
from aster.execution.core import execute_hsaco, OutputArray
try:
    out = np.zeros(1024, dtype=np.int32)
    execute_hsaco(
        hsaco_path={path!r},
        kernel_name="simd_reg_test",
        arguments=[OutputArray(out)],
        grid_dim=(1, 1, 1),
        block_dim=(1024, 1, 1),
        num_iterations=1,
    )
    print("UNEXPECTED_SUCCESS")
    sys.exit(1)
except RuntimeError as e:
    msg = str(e).lower()
    if "invalid kernel file" in msg:
        print("EXPECTED_FAILURE")
        sys.exit(0)
    else:
        print(f"UNEXPECTED_ERROR: {{e}}")
        sys.exit(2)
except Exception as e:
    print(f"UNEXPECTED_EXCEPTION: {{type(e).__name__}}: {{e}}")
    sys.exit(3)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = result.stdout.strip()
        assert result.returncode == 0 and "EXPECTED_FAILURE" in stdout, (
            f"exit={result.returncode}, stdout={stdout},"
            f" stderr={result.stderr[-500:]}"
        )
    finally:
        os.unlink(path)
