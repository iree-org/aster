// RUN: aster-opt %s \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce \
// RUN: | FileCheck %s
//
// RUN: aster-opt %s \
// RUN:   --inline \
// RUN: | aster-opt \
// RUN:   --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-amdgcn-expand-md-ops)))" \
// RUN: | aster-opt \
// RUN:   --amdgcn-reg-alloc --symbol-dce | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s --check-prefix=ASM

// Buffer copy kernels using MUBUF with IDXEN addressing mode (structured buffer
// mode). Each lane uses its thread index as the buffer element index.
//
// Two kernels with different strides:
//   buffer_idxen_copy_kernel:        stride=4   (dword elements)
//   buffer_idxen_stride1024_kernel:  stride=1024 (large elements)
//
// In structured mode (stride > 0), num_records is an element count and the OOB
// check is: index >= num_records (per-element, not per-byte).
//
// Arguments (all pointers, ordered as input_args + output_args for
// execute_kernel_and_verify compatibility):
//   arg0: src buffer pointer (8 bytes)       -- input
//   arg1: params buffer pointer (8 bytes)    -- input: [num_records, soffset]
//   arg2: dst buffer pointer (8 bytes)       -- output

// CHECK-LABEL: amdgcn.module
//   CHECK-NOT:   load_kernargs

// ASM-LABEL: buffer_idxen_copy_kernel:
//       ASM:   s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], 0
//       ASM:   buffer_load_dword{{.*}}idxen
//       ASM:   buffer_store_dword{{.*}}idxen
//       ASM:   s_endpgm

// ASM-LABEL: buffer_idxen_stride1024_kernel:
//       ASM:   buffer_load_dword{{.*}}idxen
//       ASM:   buffer_store_dword{{.*}}idxen
//       ASM:   s_endpgm
amdgcn.module @buffer_idxen_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // Load kernel args: three pointers, then dereference params to get scalars.
  func.func private @load_kernargs()
      -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
          !amdgcn.sgpr, !amdgcn.sgpr) {
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr_range<[? + 2]>
    %params_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr_range<[? + 2]>
    %dst_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr_range<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // params layout: [num_records (i32), soffset (i32)]
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32

    %nrec_dest = amdgcn.alloca : !amdgcn.sgpr
    %num_records, %t0 = amdgcn.load s_load_dword dest %nrec_dest addr %params_ptr
      offset c(%c0)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    %soff_dest = amdgcn.alloca : !amdgcn.sgpr
    %soffset, %t1 = amdgcn.load s_load_dword dest %soff_dest addr %params_ptr
      offset c(%c4)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr_range<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return %src_ptr, %dst_ptr, %num_records, %soffset
      : !amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
        !amdgcn.sgpr, !amdgcn.sgpr
  }

  // --- Kernel 1: stride=4 (dword elements, minimal structured mode) ----------

  amdgcn.kernel @buffer_idxen_copy_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {

    %src_ptr, %dst_ptr, %num_records, %soffset =
      func.call @load_kernargs()
        : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
                 !amdgcn.sgpr, !amdgcn.sgpr)

    // stride=4: each element is one dword. address = base + index * 4.
    %c4_stride = arith.constant 4 : i32
    %src_rsrc = amdgcn.make_buffer_rsrc %src_ptr, %num_records, %c4_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
    %dst_rsrc = amdgcn.make_buffer_rsrc %dst_ptr, %num_records, %c4_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>

    %vindex = amdgcn.alloca : !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load buffer_load_dword_idxen dest %load_dest addr %src_rsrc
      offset u(%soffset) + d(%vindex) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %tok_st = amdgcn.store buffer_store_dword_idxen data %loaded addr %dst_rsrc
      offset u(%soffset) + d(%vindex) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

  // --- Kernel 2: stride=1024 (large elements, proves stride math works) ------
  //
  // With stride=1024 and buffer_load_dword, each lane reads 4 bytes starting at
  // byte offset index*1024. This creates a scattered access pattern that only
  // produces correct results if the hardware is actually multiplying by stride.

  amdgcn.kernel @buffer_idxen_stride1024_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {

    %src_ptr, %dst_ptr, %num_records, %soffset =
      func.call @load_kernargs()
        : () -> (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr_range<[? + 2]>,
                 !amdgcn.sgpr, !amdgcn.sgpr)

    // stride=1024: each element is 1024 bytes. address = base + index * 1024.
    // Only the first 4 bytes of each 1024-byte element are read/written.
    %c1024_stride = arith.constant 1024 : i32
    %src_rsrc = amdgcn.make_buffer_rsrc %src_ptr, %num_records, %c1024_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>
    %dst_rsrc = amdgcn.make_buffer_rsrc %dst_ptr, %num_records, %c1024_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!amdgcn.sgpr_range<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr_range<[? + 4]>

    %vindex = amdgcn.alloca : !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load buffer_load_dword_idxen dest %load_dest addr %src_rsrc
      offset u(%soffset) + d(%vindex) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    %tok_st = amdgcn.store buffer_store_dword_idxen data %loaded addr %dst_rsrc
      offset u(%soffset) + d(%vindex) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr_range<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }
}
