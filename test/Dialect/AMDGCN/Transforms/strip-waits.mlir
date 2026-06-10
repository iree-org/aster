// RUN: aster-opt %s --aster-to-amdgcn --amdgcn-strip-waits | aster-translate --mlir-to-asm | FileCheck %s

// Verify strip-waits erases all wait ops for the StinkyTofu handoff.
// The output should contain the load/store/compute ops but no s_wait*.

// CHECK-LABEL: Module: gfx1250_strip_mod
// CHECK:    .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

// CHECK-LABEL: strip_tensor_wait:
//  CHECK-NEXT: tensor_load_to_lds s[0:3], s[8:15], s[16:19], s[20:23]
//  CHECK-NEXT: ds_load_b128 v[16:19], v32
//  CHECK-NEXT: ds_load_b128 v[20:23], v32 offset: 16
//  CHECK-NEXT: v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], v[0:7]
//  CHECK-NEXT: s_endpgm
//   CHECK-NOT: s_wait

amdgcn.module @gfx1250_strip_mod target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @strip_tensor_wait attributes {normal_forms = [#amdgcn.all_registers_allocated]} {
    %d0 = lsir.alloca : !amdgcn.sgpr<[0 : 4]>
    %d1 = lsir.alloca : !amdgcn.sgpr<[8 : 16]>
    %d2 = lsir.alloca : !amdgcn.sgpr<[16 : 20]>
    %d3 = lsir.alloca : !amdgcn.sgpr<[20 : 24]>
    %ltok = amdgcn.tensor_load_to_lds desc0 %d0 desc1 %d1 desc2 %d2 desc3 %d3
        : ins(!amdgcn.sgpr<[0 : 4]>, !amdgcn.sgpr<[8 : 16]>, !amdgcn.sgpr<[16 : 20]>,
              !amdgcn.sgpr<[20 : 24]>) -> !amdgcn.read_token<tensor>
    %wf0 = amdgcn.wait_gfx1250 deps %ltok : !amdgcn.read_token<tensor> -> !amdgcn.fence_token

    %v32 = lsir.alloca : !amdgcn.vgpr<32>
    %b_lo = lsir.alloca : !amdgcn.vgpr<[16 : 20]>
    %off0 = arith.constant 0 : i32
    %dtok0 = amdgcn.ds_load_b128 dest %b_lo addr %v32 offset c(%off0)
        : outs(!amdgcn.vgpr<[16 : 20]>) ins(!amdgcn.vgpr<32>) mods(i32) -> !amdgcn.read_token<shared>
    %b_hi = lsir.alloca : !amdgcn.vgpr<[20 : 24]>
    %off16 = arith.constant 16 : i32
    %dtok1 = amdgcn.ds_load_b128 dest %b_hi addr %v32 offset c(%off16)
        : outs(!amdgcn.vgpr<[20 : 24]>) ins(!amdgcn.vgpr<32>) mods(i32) -> !amdgcn.read_token<shared>
    %wf1 = amdgcn.wait_gfx1250 deps %dtok0, %dtok1 : !amdgcn.read_token<shared>, !amdgcn.read_token<shared> -> !amdgcn.fence_token

    %acc = lsir.alloca : !amdgcn.vgpr<[0 : 8]>
    %a_frag = lsir.alloca : !amdgcn.vgpr<[8 : 16]>
    %b_frag = lsir.alloca : !amdgcn.vgpr<[16 : 24]>

    amdgcn.v_wmma_f32_16x16x32_bf16 outs(%acc) ins(%a_frag, %b_frag, %acc)
        : outs(!amdgcn.vgpr<[0 : 8]>) ins(!amdgcn.vgpr<[8 : 16]>, !amdgcn.vgpr<[16 : 24]>,
                                          !amdgcn.vgpr<[0 : 8]>)
    amdgcn.end_kernel
  }
}
