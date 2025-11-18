"""Integration test for GPU kernel parameter passing."""

import pytest
import numpy as np

from aster import utils
from integration_test.test_utils import execute_kernel_and_verify, hsaco_file


@pytest.mark.parametrize("mcpu", ["gfx1201", "gfx942"])
def test_copy_kernel(mcpu, wavefront_size=32):
    """Test parameter passing on GPU."""
    asm = f"""
.amdgcn_target "amdgcn-amd-amdhsa--{mcpu}"
.text
.globl copy_kernel
.p2align 8
.type copy_kernel,@function
copy_kernel:

s_load_dwordx2 s[2:3], s[0:1], 0
s_load_dwordx2 s[4:5], s[0:1], 8
s_waitcnt lgkmcnt(0)

v_mov_b32_e32 v10, s2
v_mov_b32_e32 v11, s3

v_mov_b32_e32 v12, s4
v_mov_b32_e32 v13, s5

; Note: global_load_dwordx2 seems valid for both gfx942 and gfx1201, us it here.
global_load_dwordx2 v[14:15], v[10:11], off
s_waitcnt vmcnt(0)

global_store_dwordx2 v[12:13], v[14:15], off
s_waitcnt vmcnt(0)

s_endpgm

.section .rodata,"a",@progbits
.p2align 6, 0x0
.amdhsa_kernel copy_kernel
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_kernarg_size 16
    .amdhsa_user_sgpr_count 2
    .amdhsa_next_free_sgpr 6
    .amdhsa_next_free_vgpr 16
    {".amdhsa_accum_offset 4" if mcpu == "gfx942" else ""}
.end_amdhsa_kernel

    .amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
amdhsa.kernels:
  - .name:           copy_kernel
    .symbol:         copy_kernel.kd
    .kernarg_segment_align: 8
    .kernarg_segment_size: 16
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .sgpr_count:     6
    .vgpr_count:     16
    .wavefront_size: {wavefront_size}
    .max_flat_workgroup_size: 1024
    .args:
      - .size: 8
        .offset: 0
        .value_kind: global_buffer
        .address_space: global
        .actual_access: read_only
      - .size: 8
        .offset: 8
        .value_kind: global_buffer
        .address_space: global
        .actual_access: write_only
amdhsa.target:   amdgcn-amd-amdhsa--{mcpu}
...

    .end_amdgpu_metadata
"""

    # Prepare test data
    input_data = np.array([1, 2], dtype=np.int32)
    output_data = np.zeros(2, dtype=np.int32)

    def verify_fn(input_args, output_args):
        assert np.array_equal(input_args[0], output_args[0]), "Copy kernel failed!"

    # Assemble to hsaco
    hsaco_path = utils.assemble_to_hsaco(
        asm, target=mcpu, wavefront_size=wavefront_size
    )
    assert hsaco_path is not None, "Failed to assemble kernel to HSACO"

    with hsaco_file(hsaco_path):
        # Skip execution if GPU doesn't match
        if not utils.system_has_mcpu(mcpu=mcpu):
            print(asm)
            pytest.skip(
                f"GPU {mcpu} not available, but cross-compilation to HSACO succeeded"
            )

        # Execute kernel and verify results
        execute_kernel_and_verify(
            hsaco_path=hsaco_path,
            kernel_name="copy_kernel",
            input_args=[input_data],
            output_args=[output_data],
            mcpu=mcpu,
            wavefront_size=wavefront_size,
            verify_fn=verify_fn,
        )


if __name__ == "__main__":
    test_copy_kernel("gfx942")
    test_copy_kernel("gfx1201")
