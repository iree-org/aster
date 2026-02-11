// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// CHECK-LABEL:Module: buffer_test
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx942"
// CHECK:    .text
// CHECK:    .globl test_buffer_load_store
// CHECK:    .p2align 8
// CHECK:    .type test_buffer_load_store,@function
// CHECK:  test_buffer_load_store:
// CHECK:    buffer_load_dword v1, v0, s[0:3], s4 offen
// CHECK:    buffer_load_dwordx2 v[2:3], v0, s[0:3], s4 offen offset: 64
// CHECK:    buffer_load_dwordx3 v[4:6], v0, s[0:3], s4 offen offset: 96
// CHECK:    buffer_load_dwordx4 v[8:11], v0, s[0:3], s4 offen offset: 128
// CHECK:    buffer_store_dword v1, v0, s[0:3], s4 offen
// CHECK:    buffer_store_dwordx2 v[2:3], v0, s[0:3], s4 offen offset: 64
// CHECK:    buffer_store_dwordx3 v[4:6], v0, s[0:3], s4 offen offset: 96
// CHECK:    buffer_store_dwordx4 v[8:11], v0, s[0:3], s4 offen offset: 128
// CHECK:    s_endpgm
amdgcn.module @buffer_test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_buffer_load_store {
    // Buffer descriptor (4 SGPRs, s[0:3])
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %rsrc = amdgcn.make_register_range %s0, %s1, %s2, %s3
      : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %soffset = amdgcn.alloca : !amdgcn.sgpr<4>
    %vaddr = amdgcn.alloca : !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %c96 = arith.constant 96 : i32
    %c128 = arith.constant 128 : i32

    // buffer_load_dword v1, v0, s[0:3], s4 offen
    %ld1_dest = amdgcn.alloca : !amdgcn.vgpr<1>
    %ld1, %lt1 = amdgcn.load buffer_load_dword dest %ld1_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0)
      : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_load_dwordx2 v[2:3], v0, s[0:3], s4 offen offset: 64
    %ld2a = amdgcn.alloca : !amdgcn.vgpr<2>
    %ld2b = amdgcn.alloca : !amdgcn.vgpr<3>
    %ld2_dest = amdgcn.make_register_range %ld2a, %ld2b
      : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    %ld2, %lt2 = amdgcn.load buffer_load_dwordx2 dest %ld2_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c64)
      : dps(!amdgcn.vgpr_range<[2 : 4]>) ins(!amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_load_dwordx3 v[4:6], v0, s[0:3], s4 offen offset: 96
    %ld3a = amdgcn.alloca : !amdgcn.vgpr<4>
    %ld3b = amdgcn.alloca : !amdgcn.vgpr<5>
    %ld3c = amdgcn.alloca : !amdgcn.vgpr<6>
    %ld3_dest = amdgcn.make_register_range %ld3a, %ld3b, %ld3c
      : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>
    %ld3, %lt3 = amdgcn.load buffer_load_dwordx3 dest %ld3_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c96)
      : dps(!amdgcn.vgpr_range<[4 : 7]>) ins(!amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_load_dwordx4 v[8:11], v0, s[0:3], s4 offen offset: 128
    %ld4a = amdgcn.alloca : !amdgcn.vgpr<8>
    %ld4b = amdgcn.alloca : !amdgcn.vgpr<9>
    %ld4c = amdgcn.alloca : !amdgcn.vgpr<10>
    %ld4d = amdgcn.alloca : !amdgcn.vgpr<11>
    %ld4_dest = amdgcn.make_register_range %ld4a, %ld4b, %ld4c, %ld4d
      : !amdgcn.vgpr<8>, !amdgcn.vgpr<9>, !amdgcn.vgpr<10>, !amdgcn.vgpr<11>
    %ld4, %lt4 = amdgcn.load buffer_load_dwordx4 dest %ld4_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c128)
      : dps(!amdgcn.vgpr_range<[8 : 12]>) ins(!amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_store_dword v0, v1, s[0:3], s4 offen
    %st1 = amdgcn.store buffer_store_dword data %ld1 addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0)
      : ins(!amdgcn.vgpr<1>, !amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    // buffer_store_dwordx2 v0, v[2:3], s[0:3], s4 offen offset: 64
    %st2 = amdgcn.store buffer_store_dwordx2 data %ld2 addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c64)
      : ins(!amdgcn.vgpr_range<[2 : 4]>, !amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    // buffer_store_dwordx3 v[4:6], v0, s[0:3], s4 offen offset: 96
    %st3 = amdgcn.store buffer_store_dwordx3 data %ld3 addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c96)
      : ins(!amdgcn.vgpr_range<[4 : 7]>, !amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    // buffer_store_dwordx4 v[8:11], v0, s[0:3], s4 offen offset: 128
    %st4 = amdgcn.store buffer_store_dwordx4 data %ld4 addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c128)
      : ins(!amdgcn.vgpr_range<[8 : 12]>, !amdgcn.sgpr_range<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.end_kernel
  }
}
