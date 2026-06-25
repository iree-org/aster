// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s
// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN:   | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// CHECK-LABEL: cload:
//       CHECK: global_store_b32
//       CHECK: s_wait_storecnt 0
//       CHECK: s_barrier_signal_isfirst -1
//       CHECK: s_barrier_wait -1
//       CHECK: s_cbranch_scc0 [[JOIN:.*]]
//       CHECK: s_barrier_signal -3
//       CHECK: s_branch [[JOIN]]
//       CHECK: s_barrier_wait -3
//       CHECK: s_mov_b32 m0, 15
//       CHECK: cluster_load_b32
//       CHECK: s_wait_loadcnt 0
//       CHECK: global_store_b32
//       CHECK: s_wait_storecnt 0
//       CHECK: s_endpgm

amdgcn.module @cload_mod target = #amdgcn.target<gfx1250> {

  // scratch_buf (arg 0): 8 * 128 int32
  // output_buf  (arg 1): 8 * 128 int32
  amdgcn.kernel @cload arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {cluster_dims = array<i32: 4, 1, 1>,
                 grid_dims   = array<i32: 8, 1, 1>,
                 block_dims  = array<i32: 128, 1, 1>,
                 wavefront_size32,
                 shared_memory_size = 0 : i32} {
    %scratch_ptr = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %output_ptr  = amdgcn.load_arg 1 : !amdgcn.sgpr<[? + 2]>

    // cluster_id x (0 or 1); block_id x (0..7); thread_id x (0..127)
    %cid      = amdgcn.cluster_id x : !amdgcn.sgpr
    %wgid     = amdgcn.block_id   x : !amdgcn.sgpr
    %tid_vgpr = amdgcn.thread_id  x : !amdgcn.vgpr

    %c0 = arith.constant 0 : i32

    %cid_i32  = lsir.from_reg %cid      : !amdgcn.sgpr -> i32
    %wgid_i32 = lsir.from_reg %wgid     : !amdgcn.sgpr -> i32
    %tid_i32  = lsir.from_reg %tid_vgpr : !amdgcn.vgpr -> i32
    %cid_idx  = arith.index_cast %cid_i32  : i32 to index
    %wgid_idx = arith.index_cast %wgid_i32 : i32 to index
    %tid_idx  = arith.index_cast %tid_i32  : i32 to index

    // 1. store global id into scratch[wg_id*128 + tid]
    // 1.a. global id = wg_id*128 + tid (the value phase 1 stores).
    %gid_idx  = affine.apply affine_map<(d0, d1) -> (d0 * 128 + d1)>(%wgid_idx, %tid_idx)
    %gid_i32  = arith.index_cast %gid_idx : index to i32
    %gid_vgpr = lsir.to_reg %gid_i32 : i32 -> !amdgcn.vgpr
    // 1.b.wg byte offset = (wg_id*128 + tid) * 4  (each WG writes its own row)
    %wg_off_idx  = affine.apply affine_map<(d0, d1) -> ((d0 * 128 + d1) * 4)>(%wgid_idx, %tid_idx)
    %wg_off_i32  = arith.index_cast %wg_off_idx : index to i32
    %wg_off_vgpr = lsir.to_reg %wg_off_i32 : i32 -> !amdgcn.vgpr
    // 1.c. store global id into scratch[wg_id*128 + tid]
    %st_tok = amdgcn.global_store_b32 data %gid_vgpr addr %scratch_ptr
                offset d(%wg_off_vgpr) + c(%c0)
                : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32)
                -> !amdgcn.write_token<flat>
    %st_fence = amdgcn.wait_gfx1250 deps %st_tok
                  : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // 2, cluster barrier
    %bf = amdgcn.token_barrier scope(#amdgcn.barrier_scope<cluster>) deps %st_fence
            : !amdgcn.fence_token

    // 3. cluster_load_b32 multicast from the cluster-uniform address
    // 3.a. M0 = 0b1111 = 15: all 4 WGs in the cluster\
    %m0   = amdgcn.alloca : !amdgcn.m0<0>
    %mask = arith.constant 15 : i32
    amdgcn.s_mov_b32 outs(%m0) ins(%mask) : outs(!amdgcn.m0<0>) ins(i32)

    // 3.b. cluster-uniform byte offset = (cluster_id*4*128 + tid) * 4.
    %cu_off_idx  = affine.apply affine_map<(d0, d1) -> ((d0 * 512 + d1) * 4)>(%cid_idx, %tid_idx)
    %cu_off_i32  = arith.index_cast %cu_off_idx : index to i32
    %cu_off_vgpr = lsir.to_reg %cu_off_i32 : i32 -> !amdgcn.vgpr

    %dst_a = amdgcn.alloca : !amdgcn.vgpr
    %ld, %ld_tok = amdgcn.cluster_load_b32 dest %dst_a addr %scratch_ptr m0 %m0
                     offset d(%cu_off_vgpr) + c(%c0)
                     : outs(!amdgcn.vgpr) ins(!amdgcn.sgpr<[? + 2]>, !amdgcn.m0<0>, !amdgcn.vgpr)
                     mods(i32) -> !amdgcn.read_token<flat>
    %ld_fence = amdgcn.wait_gfx1250 deps %ld_tok
                  : !amdgcn.read_token<flat> -> !amdgcn.fence_token

    // 4. store output[wg_id * 128 + tid] = broadcast value
    %res_tok = amdgcn.global_store_b32 data %ld addr %output_ptr
                 offset d(%wg_off_vgpr) + c(%c0)
                 : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32)
                 -> !amdgcn.write_token<flat>
    %res_fence = amdgcn.wait_gfx1250 deps %res_tok
                   : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    amdgcn.end_kernel
  }
}
