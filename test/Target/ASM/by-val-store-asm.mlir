// RUN: aster-opt %s \
// RUN:   --pass-pipeline='builtin.module(amdgcn.module(amdgcn.kernel( \
// RUN:     aster-amdgcn-expand-md-ops, \
// RUN:     aster-amdgcn-bufferization, \
// RUN:     amdgcn-to-register-semantics)), \
// RUN:   amdgcn-backend, \
// RUN:   amdgcn-remove-test-inst, \
// RUN:   amdgcn-hazards{v_nops=0 s_nops=0})' \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s

// Verify by_val_arg survives the full pipeline and produces correct ASM.

// CHECK-LABEL: by_val_store:
// CHECK:         s_load_dword s{{[0-9]+}}, s[0:1], 0
// CHECK:         s_load_dwordx2 s[{{[0-9]+:[0-9]+}}], s[0:1], 8
// CHECK:         s_waitcnt lgkmcnt(0)
// CHECK:         v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
// CHECK:         v_lshlrev_b32_e32 v{{[0-9]+}},
// CHECK:         global_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+:[0-9]+}}]
// CHECK:         s_endpgm

// CHECK:       .amdhsa_kernel by_val_store
// CHECK:         .amdhsa_kernarg_size 16
// CHECK:         .amdhsa_user_sgpr_kernarg_segment_ptr 1

// CHECK:       amdhsa.kernels:
// CHECK:         .args:
// CHECK:           - .offset: 0
// CHECK:             .size: 4
// CHECK:             .value_kind: by_value
// CHECK:           - .access: write_only
// CHECK:             .address_space: generic
// CHECK:             .offset: 8
// CHECK:             .size: 8
// CHECK:             .value_kind: global_buffer

amdgcn.module @by_val_store_mod target = #amdgcn.target<gfx942> {
  amdgcn.kernel @by_val_store arguments <[
    #amdgcn.by_val_arg<size = 4, alignment = 4, type = i32>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> {
    %val = amdgcn.load_arg 0 : !amdgcn.sgpr
    %out_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %v_a = amdgcn.alloca : !amdgcn.vgpr
    %v_val = amdgcn.vop1.vop1 #amdgcn.inst<v_mov_b32_e32> %v_a, %val
      : (!amdgcn.vgpr, !amdgcn.sgpr) -> !amdgcn.vgpr

    %tid_x = amdgcn.thread_id x : !amdgcn.vgpr
    %c2 = arith.constant 2 : i32
    %c0 = arith.constant 0 : i32
    %voff_a = amdgcn.alloca : !amdgcn.vgpr
    %voffset = amdgcn.vop2 v_lshlrev_b32_e32 outs %voff_a ins %c2, %tid_x
      : !amdgcn.vgpr, i32, !amdgcn.vgpr

    %tok = amdgcn.store global_store_dword data %v_val addr %out_ptr
      offset d(%voffset) + c(%c0)
      : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, i32)
        -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }
}
