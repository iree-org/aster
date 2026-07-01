// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s
// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN:   | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

amdgcn.module @gfx1250_loads_e2e_mod target = #amdgcn.target<gfx1250> {

  // CHECK-LABEL: ds_load_b128:
  //       CHECK: global_load_b128
  //       CHECK: ds_store_b128
  //       CHECK: ds_load_b128
  //       CHECK: global_store_b128
  //       CHECK: s_endpgm
  amdgcn.kernel @ds_load_b128 arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {grid_dims = array<i32: 1, 1, 1>,
                 block_dims = array<i32: 32, 1, 1>,
                 wavefront_size32,
                 shared_memory_size = 512 : i32} {
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %dst_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %c0 = arith.constant 0 : i32

    // Per-lane byte offset = tid * 16 (four dwords per lane).
    %tid_vgpr = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = lsir.from_reg %tid_vgpr : !amdgcn.vgpr -> i32
    %tid_idx = arith.index_cast %tid_i32 : i32 to index
    %off_idx = affine.apply affine_map<(d0) -> (d0 * 16)>(%tid_idx)
    %off_i32 = arith.index_cast %off_idx : index to i32
    %off = lsir.to_reg %off_i32 : i32 -> !amdgcn.vgpr

    %gpr4 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %ld4, %load_tok = amdgcn.global_load_b128 dest %gpr4 addr %src_ptr
                         offset d(%off) + c(%c0)
        : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<flat>
    %load_fence = amdgcn.wait_gfx1250 deps %load_tok
        : !amdgcn.read_token<flat> -> !amdgcn.fence_token

    %store_tok = amdgcn.ds_store_b128 data %ld4 addr %off offset c(%c0)
        : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.write_token<shared>
    %store_fence = amdgcn.wait_gfx1250 deps %store_tok
        : !amdgcn.write_token<shared> -> !amdgcn.fence_token

    %lds_dst4 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %back4, %ds_tok = amdgcn.ds_load_b128 dest %lds_dst4 addr %off offset c(%c0)
        : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<shared> fence_token %store_fence : !amdgcn.fence_token
    %ds_fence = amdgcn.wait_gfx1250 deps %ds_tok
        : !amdgcn.read_token<shared> -> !amdgcn.fence_token

    %out_tok = amdgcn.global_store_b128 data %back4 addr %dst_ptr
                   offset d(%off) + c(%c0)
        : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.write_token<flat>
    %out_fence = amdgcn.wait_gfx1250 deps %out_tok
        : !amdgcn.write_token<flat> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }
}
