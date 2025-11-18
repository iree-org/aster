"""Utilities for flushing the Last Level Cache (LLC) on GPU."""

import os
from typing import Tuple, Callable

from aster import utils
from aster._mlir_libs._runtime_module import (
    hip_module_load_data,
    hip_module_get_function,
    hip_module_launch_kernel,
    hip_device_synchronize,
    hip_free,
    hip_malloc,
    hip_module_unload,
    hip_function_free,
)


class FlushLLC:
    """A class that flushes the LLC by running a kernel that reads/writes a large buffer."""

    def __init__(
        self,
        size_mb: int = 256,
        mcpu: str = "gfx942",
        num_workgroups: int = 304,
        wavefront_size: int = 64,
    ):
        """Initialize the FlushLLC instance.

        Args:
            size_mb: Size of the buffer in MB to allocate for flushing (default: 256)
            mcpu: Target GPU architecture (default: gfx942)
            num_workgroups: Number of workgroups to launch (default: 304 for MI300X)
            wavefront_size: Wavefront size (default: 64)
        """
        self.size_mb = size_mb
        self.mcpu = mcpu
        self.wavefront_size = wavefront_size
        self.num_workgroups = num_workgroups

        self.flush_buffer_size = size_mb * 1024 * 1024  # Convert MB to bytes
        self.num_elements = self.flush_buffer_size // 4  # Number of int32 elements
        self.num_threads = wavefront_size  # Threads per workgroup
        self.elements_per_thread = self.num_elements // (
            num_workgroups * self.num_threads
        )

        # State variables
        self.flush_buffer_ptr = None
        self.module = None
        self.function = None
        self.params_tuple = None
        self.initialized = False

        # Generate minimal assembly kernel that reads and writes back the buffer
        # Each thread handles elements_per_thread elements
        # Uses SGPR base + VGPR offset format for global memory ops
        self.flush_kernel_asm = f"""
  .amdgcn_target "amdgcn-amd-amdhsa--{mcpu}"
  .text
  .globl flush_llc_kernel
  .p2align 8
  .type flush_llc_kernel,@function
flush_llc_kernel:
  ; Load buffer pointer from kernarg into s[0:1]
  s_load_dwordx2 s[4:5], s[0:1], 0x0
  s_waitcnt lgkmcnt(0)

  ; Calculate global thread ID: workgroup_id * num_threads + local_id
  ; s6 = workgroup_id (from hardware), v0 = local thread id (from hardware)
  s_mul_i32 s2, s6, {self.num_threads}
  v_add_u32 v1, s2, v0

  ; Calculate base offset for this thread: thread_id * elements_per_thread * 4
  ; Move literal to SGPR first (LLVM MC doesn't support inline literals for v_mul_lo_u32)
  s_mov_b32 s3, {self.elements_per_thread * 4}
  v_mul_lo_u32 v2, v1, s3

  ; v2 now contains the starting byte offset for this thread
  ; Loop over elements_per_thread elements, reading and writing each one
  s_mov_b32 s2, {self.elements_per_thread}
.Lloop:
  ; Load dword from global memory: vdst, voffset, saddr
  global_load_dword v3, v2, s[4:5]
  s_waitcnt vmcnt(0)

  ; Increment offset by 4 bytes
  v_add_u32 v2, v2, 4

  ; Decrement counter and loop
  s_sub_u32 s2, s2, 1
  s_cmp_gt_u32 s2, 0
  s_cbranch_scc1 .Lloop

  s_endpgm
  .section .rodata,"a",@progbits
  .p2align 6, 0x0
  .amdhsa_kernel flush_llc_kernel
    .amdhsa_user_sgpr_count 6
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_next_free_vgpr 4
    .amdhsa_next_free_sgpr 8
    .amdhsa_accum_offset 4
  .end_amdhsa_kernel
  .text
.Lfunc_end0:
  .size flush_llc_kernel, .Lfunc_end0-flush_llc_kernel

  .amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count: 0
    .args:
      - .address_space: global
        .offset: 0
        .size: 8
        .value_kind: global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language: Assembler
    .max_flat_workgroup_size: 1024
    .name: flush_llc_kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 8
    .sgpr_spill_count: 0
    .symbol: flush_llc_kernel.kd
    .vgpr_count: 4
    .vgpr_spill_count: 0
    .wavefront_size: {wavefront_size}
amdhsa.version:
  - 1
  - 2
---
  .end_amdgpu_metadata
"""

    def initialize(self) -> None:
        """Initialize the flush kernel and buffer (called once on first flush)."""
        if self.initialized:
            return

        # Compile the kernel to HSACO
        hsaco_path = utils.assemble_to_hsaco(
            self.flush_kernel_asm, target=self.mcpu, wavefront_size=self.wavefront_size
        )
        if hsaco_path is None:
            raise RuntimeError("Failed to assemble flush LLC kernel")

        # Load the module and function
        with open(hsaco_path, "rb") as f:
            hsaco_binary = f.read()
        self.module = hip_module_load_data(hsaco_binary)
        self.function = hip_module_get_function(self.module, b"flush_llc_kernel")

        # Clean up temp file
        os.unlink(hsaco_path)

        # Allocate the GPU buffer
        self.flush_buffer_ptr = hip_malloc(self.flush_buffer_size)

        # Create kernel args (just the buffer pointer)
        ptr_value = utils.unwrap_pointer_from_capsule(self.flush_buffer_ptr)
        self.params_tuple = utils.create_kernel_args_capsule([ptr_value])

        self.initialized = True

    def flush_llc(self) -> None:
        """Flush the LLC by running a kernel that reads/writes the buffer."""
        self.initialize()

        # Launch the flush kernel
        hip_module_launch_kernel(
            self.function,
            self.num_workgroups,
            1,
            1,
            self.num_threads,
            1,
            1,
            self.params_tuple[0],
        )
        hip_device_synchronize()

    def cleanup(self) -> None:
        """Clean up the flush buffer and module."""
        if self.flush_buffer_ptr is not None:
            hip_free(self.flush_buffer_ptr)
            self.flush_buffer_ptr = None
        if self.function is not None:
            hip_function_free(self.function)
            self.function = None
        if self.module is not None:
            hip_module_unload(self.module)
            self.module = None
        self.initialized = False


def create_flush_llc_fn(
    size_mb: int = 256,
    mcpu: str = "gfx942",
    wavefront_size: int = 64,
    num_workgroups: int = 304,
) -> Tuple[Callable[[], None], Callable[[], None]]:
    """Create a function that flushes the LLC by running a kernel that reads/writes a large buffer.

    Args:
        size_mb: Size of the buffer in MB to allocate for flushing (default: 256)
        mcpu: Target GPU architecture (default: gfx942)
        wavefront_size: Wavefront size (default: 64)
        num_workgroups: Number of workgroups to launch (default: 304 for MI300X)

    Returns:
        A tuple of (flush_function, cleanup_function)
    """
    flush_llc_obj = FlushLLC(
        size_mb=size_mb,
        mcpu=mcpu,
        wavefront_size=wavefront_size,
        num_workgroups=num_workgroups,
    )
    return flush_llc_obj.flush_llc, flush_llc_obj.cleanup


if __name__ == "__main__":
    flush_llc = FlushLLC(
        size_mb=256, mcpu="gfx942", wavefront_size=64, num_workgroups=304
    )
    flush_llc.initialize()
