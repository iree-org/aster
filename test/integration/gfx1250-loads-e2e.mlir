// RUN: aster-opt %s --verify-roundtrip

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
        : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>, !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<flat>
    amdgcn.end_kernel
  }

  amdgcn.kernel @s_set_vgpr_msb attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    amdgcn.s_set_vgpr_msb 3
    amdgcn.end_kernel
  }
}
