// RUN: aster-opt %s --aster-to-amdgcn | aster-translate --mlir-to-asm | FileCheck %s

// CHECK-LABEL: Module: gfx1250_loads_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// CHECK-LABEL: test_ds_load_b128_asm:
//  CHECK-NEXT: ds_load_b128 v[0:3], v4
//  CHECK-NEXT: s_endpgm

// CHECK-LABEL: test_tensor_load_to_lds_asm:
//  CHECK-NEXT: tensor_load_to_lds s[0:3], s[8:15], s[16:19], s[20:23]
//  CHECK-NEXT: s_endpgm

// CHECK-LABEL: test_s_set_vgpr_msb_asm:
//  CHECK-NEXT: s_set_vgpr_msb 3
//  CHECK-NEXT: s_endpgm

amdgcn.module @gfx1250_loads_mod target = #amdgcn.target<gfx1250> {

  // ds_load_b128: gfx1250 plain DS load.
  amdgcn.kernel @test_ds_load_b128_asm attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %dst4 = lsir.alloca : !amdgcn.vgpr<[0 : 4]>
    %addr = lsir.alloca : !amdgcn.vgpr<4>
    %offset = arith.constant 0 : i32
    %tok = amdgcn.ds_load_b128 dest %dst4 addr %addr offset c(%offset)
        : outs(!amdgcn.vgpr<[0 : 4]>) ins(!amdgcn.vgpr<4>) mods(i32) -> !amdgcn.read_token<shared>
    amdgcn.end_kernel
  }

  // tensor_load_to_lds: VIMAGE tensor engine, gfx1250 only.
  amdgcn.kernel @test_tensor_load_to_lds_asm attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
    %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
    %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
    %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
    %tok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
        : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
    amdgcn.end_kernel
  }

  // s_set_vgpr_msb: SOPP gfx1250 instruction.
  amdgcn.kernel @test_s_set_vgpr_msb_asm attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    amdgcn.s_set_vgpr_msb 3
    amdgcn.end_kernel
  }
}
