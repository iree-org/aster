// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s
// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN:   | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// CHECK: .amdgcn_target "amdgcn-amd-amdhsa--gfx1250"

amdgcn.module @gfx1250_async_load_e2e_mod target = #amdgcn.target<gfx1250> {

  // CHECK-LABEL: async_load_to_lds:
  //       CHECK: global_load_async_to_lds_b32
  //       CHECK: s_endpgm
  amdgcn.kernel @async_load_to_lds arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {grid_dims = array<i32: 1, 1, 1>,
                 block_dims = array<i32: 32, 1, 1>,
                 wavefront_size32,
                 shared_memory_size = 128 : i32} {
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %c0 = arith.constant 0 : i32

    // Per-lane byte offset = tid * 4 (one dword per lane).
    %tid_vgpr = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = lsir.from_reg %tid_vgpr : !amdgcn.vgpr -> i32
    %tid_idx = arith.index_cast %tid_i32 : i32 to index
    %off_idx = affine.apply affine_map<(d0) -> (d0 * 4)>(%tid_idx)
    %off_i32 = arith.index_cast %off_idx : index to i32
    %off = lsir.to_reg %off_i32 : i32 -> !amdgcn.vgpr

    %tok = amdgcn.global_load_async_to_lds_b32 addr %src_ptr lds_addr %off
               offset d(%off) + c(%c0)
        : ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<async>
    amdgcn.end_kernel
  }

  // CHECK-LABEL: async_load_wait:
  //       CHECK: global_load_async_to_lds_b32
  //       CHECK: s_wait_asynccnt 0
  //       CHECK: ds_load_b32
  //       CHECK: s_wait_dscnt 0
  //       CHECK: global_store_b32
  //       CHECK: s_wait_storecnt 0
  //       CHECK: s_endpgm
  amdgcn.kernel @async_load_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {grid_dims = array<i32: 1, 1, 1>,
                 block_dims = array<i32: 32, 1, 1>,
                 wavefront_size32,
                 shared_memory_size = 128 : i32} {
    %src_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %dst_ptr = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>
    %c0 = arith.constant 0 : i32

    // Per-lane byte offset = tid * 4 (one dword per lane).
    %tid_vgpr = amdgcn.thread_id x : !amdgcn.vgpr
    %tid_i32 = lsir.from_reg %tid_vgpr : !amdgcn.vgpr -> i32
    %tid_idx = arith.index_cast %tid_i32 : i32 to index
    %off_idx = affine.apply affine_map<(d0) -> (d0 * 4)>(%tid_idx)
    %off_i32 = arith.index_cast %off_idx : index to i32
    %off = lsir.to_reg %off_i32 : i32 -> !amdgcn.vgpr

    %async_tok = amdgcn.global_load_async_to_lds_b32 addr %src_ptr lds_addr %off
                     offset d(%off) + c(%c0)
        : ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<async>
    %async_fence = amdgcn.wait_gfx1250 deps %async_tok
        : !amdgcn.read_token<async> -> !amdgcn.fence_token
    %lds_dst = amdgcn.alloca : !amdgcn.vgpr
    %ld, %ds_tok = amdgcn.ds_load_b32 dest %lds_dst addr %off offset c(%c0)
        : outs(!amdgcn.vgpr) ins(!amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<shared> fence_token %async_fence : !amdgcn.fence_token
    %ds_fence = amdgcn.wait_gfx1250 deps %ds_tok
        : !amdgcn.read_token<shared> -> !amdgcn.fence_token
    %res_tok = amdgcn.global_store_b32 data %ld addr %dst_ptr
                   offset d(%off) + c(%c0)
        : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.write_token<flat>
    %res_fence = amdgcn.wait_gfx1250 deps %res_tok
        : !amdgcn.write_token<flat> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }

  // CHECK-LABEL: async_load_b128_saddr:
  //       CHECK: global_load_async_to_lds_b128
  //       CHECK: s_wait_asynccnt 0
  //       CHECK: ds_load_b128
  //       CHECK: s_wait_dscnt 0
  //       CHECK: global_store_b128
  //       CHECK: s_wait_storecnt 0
  //       CHECK: s_endpgm
  amdgcn.kernel @async_load_b128_saddr arguments <[
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

    %async_tok = amdgcn.global_load_async_to_lds_b128 addr %src_ptr lds_addr %off
                     offset d(%off) + c(%c0)
        : ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<async>
    %async_fence = amdgcn.wait_gfx1250 deps %async_tok
        : !amdgcn.read_token<async> -> !amdgcn.fence_token
    %lds_dst4 = lsir.alloca : !amdgcn.vgpr<[? + 4]>
    %ld4, %ds_tok = amdgcn.ds_load_b128 dest %lds_dst4 addr %off offset c(%c0)
        : outs(!amdgcn.vgpr<[? + 4]>) ins(!amdgcn.vgpr) mods(i32)
        -> !amdgcn.read_token<shared> fence_token %async_fence : !amdgcn.fence_token
    %ds_fence = amdgcn.wait_gfx1250 deps %ds_tok
        : !amdgcn.read_token<shared> -> !amdgcn.fence_token
    %store_tok = amdgcn.global_store_b128 data %ld4 addr %dst_ptr
                     offset d(%off) + c(%c0)
        : ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32)
        -> !amdgcn.write_token<flat>
    %store_fence = amdgcn.wait_gfx1250 deps %store_tok
        : !amdgcn.write_token<flat> -> !amdgcn.fence_token
    amdgcn.end_kernel
  }
}
