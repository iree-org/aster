// RUN: aster-translate %s --mlir-to-asm | FileCheck %s

// Test cache modifier (glc, slc, scc) emission on buffer and global memory ops.

// CHECK-LABEL:Module: cache_mod_buffer_test
// CHECK:  test_buffer_cache_modifiers:
// CHECK:    buffer_load_dword v1, v0, s[0:3], s4 offen glc
// CHECK:    buffer_load_dwordx2 v[2:3], v0, s[0:3], s4 offen slc
// CHECK:    buffer_load_dwordx4 v[4:7], v0, s[0:3], s4 offen glc slc scc
// CHECK:    buffer_store_dword v1, v0, s[0:3], s4 offen scc
// CHECK:    s_endpgm
amdgcn.module @cache_mod_buffer_test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_buffer_cache_modifiers {
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %rsrc = amdgcn.make_register_range %s0, %s1, %s2, %s3
      : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %soffset = amdgcn.alloca : !amdgcn.sgpr<4>
    %vaddr = amdgcn.alloca : !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    // buffer_load_dword with glc only
    %ld1_dest = amdgcn.alloca : !amdgcn.vgpr<1>
    %lt1 = amdgcn.load buffer_load_dword dest %ld1_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0) {glc = true}
      : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_load_dwordx2 with slc only
    %ld2a = amdgcn.alloca : !amdgcn.vgpr<2>
    %ld2b = amdgcn.alloca : !amdgcn.vgpr<3>
    %ld2_dest = amdgcn.make_register_range %ld2a, %ld2b
      : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    %lt2 = amdgcn.load buffer_load_dwordx2 dest %ld2_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0) {slc = true}
      : dps(!amdgcn.vgpr<[2 : 4]>) ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_load_dwordx4 with all three modifiers
    %ld4a = amdgcn.alloca : !amdgcn.vgpr<4>
    %ld4b = amdgcn.alloca : !amdgcn.vgpr<5>
    %ld4c = amdgcn.alloca : !amdgcn.vgpr<6>
    %ld4d = amdgcn.alloca : !amdgcn.vgpr<7>
    %ld4_dest = amdgcn.make_register_range %ld4a, %ld4b, %ld4c, %ld4d
      : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
    %lt3 = amdgcn.load buffer_load_dwordx4 dest %ld4_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0) {glc = true, slc = true, scc = true}
      : dps(!amdgcn.vgpr<[4 : 8]>) ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    // buffer_store_dword with scc only
    %st1 = amdgcn.store buffer_store_dword data %ld1_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0) {scc = true}
      : ins(!amdgcn.vgpr<1>, !amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.end_kernel
  }
}

// CHECK-LABEL:Module: cache_mod_global_test
// CHECK:  test_global_cache_modifiers:
// CHECK:    global_load_dword v1, v[0:1], off glc
// CHECK:    global_load_dwordx2 v[2:3], v[0:1], off slc
// CHECK:    global_load_dwordx4 v[4:7], v[0:1], off glc slc scc
// CHECK:    global_store_dword v[0:1], v1, off scc
// CHECK:    s_endpgm
amdgcn.module @cache_mod_global_test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_global_cache_modifiers {
    %va = amdgcn.alloca : !amdgcn.vgpr<0>
    %vb = amdgcn.alloca : !amdgcn.vgpr<1>
    %addr = amdgcn.make_register_range %va, %vb
      : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %c0 = arith.constant 0 : i32

    // global_load_dword with glc
    %ld1_dest = amdgcn.alloca : !amdgcn.vgpr<1>
    %lt1 = amdgcn.load global_load_dword dest %ld1_dest addr %addr
      offset c(%c0) {glc = true}
      : dps(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<[0 : 2]>, i32)
        -> !amdgcn.read_token<flat>

    // global_load_dwordx2 with slc
    %ld2a = amdgcn.alloca : !amdgcn.vgpr<2>
    %ld2b = amdgcn.alloca : !amdgcn.vgpr<3>
    %ld2_dest = amdgcn.make_register_range %ld2a, %ld2b
      : !amdgcn.vgpr<2>, !amdgcn.vgpr<3>
    %lt2 = amdgcn.load global_load_dwordx2 dest %ld2_dest addr %addr
      offset c(%c0) {slc = true}
      : dps(!amdgcn.vgpr<[2 : 4]>) ins(!amdgcn.vgpr<[0 : 2]>, i32)
        -> !amdgcn.read_token<flat>

    // global_load_dwordx4 with all modifiers
    %ld4a = amdgcn.alloca : !amdgcn.vgpr<4>
    %ld4b = amdgcn.alloca : !amdgcn.vgpr<5>
    %ld4c = amdgcn.alloca : !amdgcn.vgpr<6>
    %ld4d = amdgcn.alloca : !amdgcn.vgpr<7>
    %ld4_dest = amdgcn.make_register_range %ld4a, %ld4b, %ld4c, %ld4d
      : !amdgcn.vgpr<4>, !amdgcn.vgpr<5>, !amdgcn.vgpr<6>, !amdgcn.vgpr<7>
    %lt3 = amdgcn.load global_load_dwordx4 dest %ld4_dest addr %addr
      offset c(%c0) {glc = true, slc = true, scc = true}
      : dps(!amdgcn.vgpr<[4 : 8]>) ins(!amdgcn.vgpr<[0 : 2]>, i32)
        -> !amdgcn.read_token<flat>

    // global_store_dword with scc
    %st1 = amdgcn.store global_store_dword data %ld1_dest addr %addr
      offset c(%c0) {scc = true}
      : ins(!amdgcn.vgpr<1>, !amdgcn.vgpr<[0 : 2]>, i32)
        -> !amdgcn.write_token<flat>

    amdgcn.end_kernel
  }
}

// CHECK-LABEL:Module: cache_mod_no_modifier_test
// CHECK:  test_no_modifiers:
// CHECK:    buffer_load_dword v1, v0, s[0:3], s4 offen
// CHECK-NOT: glc
// CHECK-NOT: slc
// CHECK-NOT: scc
// CHECK:    global_load_dword v1, v[0:1], off
// CHECK-NOT: glc
// CHECK-NOT: slc
// CHECK-NOT: scc
// CHECK:    s_endpgm
amdgcn.module @cache_mod_no_modifier_test target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  amdgcn.kernel @test_no_modifiers {
    %s0 = amdgcn.alloca : !amdgcn.sgpr<0>
    %s1 = amdgcn.alloca : !amdgcn.sgpr<1>
    %s2 = amdgcn.alloca : !amdgcn.sgpr<2>
    %s3 = amdgcn.alloca : !amdgcn.sgpr<3>
    %rsrc = amdgcn.make_register_range %s0, %s1, %s2, %s3
      : !amdgcn.sgpr<0>, !amdgcn.sgpr<1>, !amdgcn.sgpr<2>, !amdgcn.sgpr<3>
    %soffset = amdgcn.alloca : !amdgcn.sgpr<4>
    %vaddr = amdgcn.alloca : !amdgcn.vgpr<0>
    %c0 = arith.constant 0 : i32

    // No modifiers - backward compat
    %ld1_dest = amdgcn.alloca : !amdgcn.vgpr<1>
    %lt1 = amdgcn.load buffer_load_dword dest %ld1_dest addr %rsrc
      offset u(%soffset) + d(%vaddr) + c(%c0)
      : dps(!amdgcn.vgpr<1>) ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<4>, !amdgcn.vgpr<0>, i32)
        -> !amdgcn.read_token<flat>

    %va = amdgcn.alloca : !amdgcn.vgpr<0>
    %vb = amdgcn.alloca : !amdgcn.vgpr<1>
    %gaddr = amdgcn.make_register_range %va, %vb
      : !amdgcn.vgpr<0>, !amdgcn.vgpr<1>
    %gld_dest = amdgcn.alloca : !amdgcn.vgpr<1>
    %glt = amdgcn.load global_load_dword dest %gld_dest addr %gaddr
      offset c(%c0)
      : dps(!amdgcn.vgpr<1>) ins(!amdgcn.vgpr<[0 : 2]>, i32)
        -> !amdgcn.read_token<flat>

    amdgcn.end_kernel
  }
}
