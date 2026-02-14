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

// Buffer copy kernel using MUBUF buffer_load/store with hardware bounds checking
// via buffer descriptors (make_buffer_rsrc). Each lane copies one dword from src
// to dst. OOB lanes (beyond num_records bytes) read 0 and writes are silently
// dropped.
//
// The buffer descriptor uses stride=0 (raw mode) where num_records is a byte
// count and the OOB check is: voffset + soffset < num_records.
//
// Arguments (all pointers, ordered as input_args + output_args for
// execute_kernel_and_verify compatibility):
//   arg0: src buffer pointer (8 bytes)       -- input
//   arg1: params buffer pointer (8 bytes)    -- input: [src_num_bytes, dst_num_bytes, soffset]
//   arg2: dst buffer pointer (8 bytes)       -- output

// CHECK-LABEL: amdgcn.module
//   CHECK-NOT:   load_kernargs

// ASM-LABEL: buffer_copy_kernel:
//       ASM:   s_load_dwordx2 s[{{[0-9]+}}:{{[0-9]+}}], s[0:1], 0
//   ASM-NOT:   s_or_b32
//       ASM:   buffer_load_dword
//       ASM:   buffer_store_dword
//       ASM:   s_endpgm
amdgcn.module @buffer_copy_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {

  // Load the three pointer args from the kernarg segment, then dereference
  // the params pointer to get the scalar values (num_records are byte counts).
  func.func private @load_kernargs()
      -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
          !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) {
    // Load the three 64-bit pointers from kernarg segment
    // Order: src(0), params(1), dst(2) -- matches input_args + output_args
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %params_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %dst_ptr = amdgcn.load_arg 2 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Dereference params_ptr to load [src_num_elements, dst_num_elements, soffset]
    %c0 = arith.constant 0 : i32
    %c4 = arith.constant 4 : i32
    %c8 = arith.constant 8 : i32

    %snelems_dest = amdgcn.alloca : !amdgcn.sgpr
    %src_nelems, %t0 = amdgcn.load s_load_dword dest %snelems_dest addr %params_ptr
      offset c(%c0)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    %dnelems_dest = amdgcn.alloca : !amdgcn.sgpr
    %dst_nelems, %t1 = amdgcn.load s_load_dword dest %dnelems_dest addr %params_ptr
      offset c(%c4)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    %soff_dest = amdgcn.alloca : !amdgcn.sgpr
    %soffset, %t2 = amdgcn.load s_load_dword dest %soff_dest addr %params_ptr
      offset c(%c8)
      : dps(!amdgcn.sgpr) ins(!amdgcn.sgpr<[? + 2]>, i32)
        -> !amdgcn.read_token<constant>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return %src_ptr, %dst_ptr, %src_nelems, %dst_nelems, %soffset
      : !amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
        !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr
  }

  amdgcn.kernel @buffer_copy_kernel arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {

    // Load kernel arguments (3 pointers; scalars come from params buffer)
    %src_ptr, %dst_ptr, %src_nelems, %dst_nelems, %soffset =
      func.call @load_kernargs()
        : () -> (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr<[? + 2]>,
                 !amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr)

    // Build buffer descriptors with stride=0 (raw mode).
    // num_records is a byte count; the hardware checks voffset + soffset < num_records.
    // flags = 0x20000 sets DATA_FORMAT=4 (32-bit), required for buffer ops on GFX9.
    %c0_stride = arith.constant 0 : i32
    %src_rsrc = amdgcn.make_buffer_rsrc %src_ptr, %src_nelems, %c0_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr<[? + 4]>
    %dst_rsrc = amdgcn.make_buffer_rsrc %dst_ptr, %dst_nelems, %c0_stride,
      cache_swizzle = false, swizzle_enable = false, flags = 131072
      : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> !amdgcn.sgpr<[? + 4]>

    // v0 = threadidx.x (hardware-initialized)
    %threadidx_x = amdgcn.alloca : !amdgcn.vgpr<0>

    // Compute byte offset: threadidx.x << 2 (each lane copies one dword = 4 bytes)
    %voffset_alloc = amdgcn.alloca : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voffset_alloc ins %c2, %threadidx_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr<0>

    %c0 = arith.constant 0 : i32

    // buffer_load_dword: load one dword from src[threadidx.x]
    // OOB lanes (voffset + soffset >= num_records) get 0
    %load_dest = amdgcn.alloca : !amdgcn.vgpr
    %loaded, %tok_ld = amdgcn.load buffer_load_dword dest %load_dest addr %src_rsrc
      offset u(%soffset) + d(%voffset) + c(%c0)
      : dps(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
        -> !amdgcn.read_token<flat>

    // Wait for buffer load
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // buffer_store_dword: store loaded dword to dst[threadidx.x]
    // OOB stores (voffset + soffset >= num_records) are silently dropped
    %tok_st = amdgcn.store buffer_store_dword data %loaded addr %dst_rsrc
      offset u(%soffset) + d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 4]>, !amdgcn.sgpr, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>

    // Wait for buffer store
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }
}
