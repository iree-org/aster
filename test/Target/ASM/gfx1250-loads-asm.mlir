// RUN: aster-opt %s --pass-pipeline='builtin.module(aster-to-amdgcn,amdgcn.module(amdgcn-convert-waits))' | aster-translate --mlir-to-asm | FileCheck %s

// CHECK-LABEL: Module: gfx1250_loads_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// CHECK-LABEL: ds_load_b128:
//  CHECK-NEXT: ds_load_b128 v[0:3], v4
//  CHECK-NEXT: s_endpgm

// CHECK-LABEL: tensor_load_to_lds:
//  CHECK-NEXT: tensor_load_to_lds s[0:3], s[8:15], s[16:19], s[20:23]
//  CHECK-NEXT: s_endpgm

// CHECK-LABEL: s_set_vgpr_msb:
//  CHECK-NEXT: s_set_vgpr_msb 3
//  CHECK-NEXT: s_endpgm

// tensor_load_to_lds completion is waited via s_wait_tensorcnt (TENSOR_CNT).
// CHECK-LABEL: tensor_load_wait:
//  CHECK-NEXT: tensor_load_to_lds s[0:3], s[8:15], s[16:19], s[20:23]
//  CHECK-NEXT: s_wait_tensorcnt 0
//  CHECK-NEXT: s_endpgm

// ds_load_b128 completion is waited via s_wait_dscnt (DS_CNT).
// CHECK-LABEL: ds_load_wait:
//  CHECK-NEXT: ds_load_b128 v[0:3], v4
//  CHECK-NEXT: s_wait_dscnt 0
//  CHECK-NEXT: s_endpgm

// A load_cnt + ds_cnt wait fuses into the combined s_wait_loadcnt_dscnt.
// CHECK-LABEL: fused_wait:
//  CHECK-NEXT: s_wait_loadcnt_dscnt 0
//  CHECK-NEXT: s_endpgm

// global_load_async_to_lds_b32 emits the VFLAT async-to-LDS load.
// CHECK-LABEL: async_load_to_lds:
//  CHECK-NEXT: global_load_async_to_lds_b32 v2, v[0:1], off
//  CHECK-NEXT: s_endpgm

// global_load_async_to_lds completion is waited via s_wait_asynccnt (ASYNC_CNT).
// CHECK-LABEL: async_load_wait:
//  CHECK-NEXT: global_load_async_to_lds_b32 v2, v[0:1], off
//  CHECK-NEXT: s_wait_asynccnt 0
//  CHECK-NEXT: s_endpgm

// The SADDR form takes an SGPR base + VGPR voffset and a const byte offset.
// CHECK-LABEL: async_load_b128_saddr:
//  CHECK-NEXT: global_load_async_to_lds_b128 v2, v0, s[0:1] offset: 16
//  CHECK-NEXT: s_endpgm

amdgcn.module @gfx1250_loads_mod target = #amdgcn.target<gfx1250> {

  amdgcn.kernel @ds_load_b128 attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %dst4 = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
    %addr = lsir.alloca : !amdgcn.vgpr<4>
    %offset = arith.constant 0 : i32
    %tok = amdgcn.ds_load_b128 dest %dst4 addr %addr offset c(%offset)
        : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr<4>) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  amdgcn.kernel @tensor_load_to_lds attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
    %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
    %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
    %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
    %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
        : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
    amdgcn.end_kernel
  }

  amdgcn.kernel @s_set_vgpr_msb attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    amdgcn.s_set_vgpr_msb 3
    amdgcn.end_kernel
  }

  amdgcn.kernel @tensor_load_wait attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
    %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
    %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
    %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
    %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
        : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
    %wf0 = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<tensor> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }

  amdgcn.kernel @ds_load_wait attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %dst4 = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
    %addr = lsir.alloca : !amdgcn.vgpr<4>
    %offset = arith.constant 0 : i32
    %tok = amdgcn.ds_load_b128 dest %dst4 addr %addr offset c(%offset)
        : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr<4>) mods(i32) -> !amdgcn.read_token<shared>
    %wf1 = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<shared> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }

  amdgcn.kernel @fused_wait attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %wf2 = amdgcn.wait_gfx1250 load_cnt 0 ds_cnt 0 -> !amdgcn.fence_token
    amdgcn.end_kernel
  }

  amdgcn.kernel @async_load_to_lds attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %addr = lsir.alloca : !amdgcn.vgpr<[0 : 2]>
    %lds_addr = lsir.alloca : !amdgcn.vgpr<2>
    %offset = arith.constant 0 : i32
    %tok = amdgcn.global_load_async_to_lds_b32 addr %addr lds_addr %lds_addr offset c(%offset)
        : ins(!amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<2>) mods(i32) -> !amdgcn.read_token<async>
    amdgcn.end_kernel
  }

  amdgcn.kernel @async_load_wait attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %addr = lsir.alloca : !amdgcn.vgpr<[0 : 2]>
    %lds_addr = lsir.alloca : !amdgcn.vgpr<2>
    %offset = arith.constant 0 : i32
    %tok = amdgcn.global_load_async_to_lds_b32 addr %addr lds_addr %lds_addr offset c(%offset)
        : ins(!amdgcn.vgpr<[0 : 2]>, !amdgcn.vgpr<2>) mods(i32) -> !amdgcn.read_token<async>
    %wf = amdgcn.wait_gfx1250 deps %tok : !amdgcn.read_token<async> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }

  amdgcn.kernel @async_load_b128_saddr attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %saddr = lsir.alloca : !amdgcn.sgpr<[0 : 2]>
    %voff = lsir.alloca : !amdgcn.vgpr<0>
    %lds_addr = lsir.alloca : !amdgcn.vgpr<2>
    %offset = arith.constant 16 : i32
    %tok = amdgcn.global_load_async_to_lds_b128 addr %saddr lds_addr %lds_addr offset d(%voff) + c(%offset)
        : ins(!amdgcn.sgpr<[0 : 2]>, !amdgcn.vgpr<2>, !amdgcn.vgpr<0>) mods(i32) -> !amdgcn.read_token<async>
    amdgcn.end_kernel
  }

}
