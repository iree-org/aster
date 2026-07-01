// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm | FileCheck %s
//
// RUN: aster-opt %s %GFX1250_CLUSTER_ASM_PIPELINE% \
// RUN: | aster-translate --mlir-to-asm \
// RUN:   | llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 -mattr=+wavefrontsize32 -filetype=obj -o %t.o

// CHECK-LABEL: cluster_ids:
//  CHECK: s_bfe_u32 s0, ttmp9, 2097152
//  CHECK: s_bfe_u32 s1, ttmp6, 262144
//  CHECK: s_bfe_u32 s2, ttmp6, 262156
//  CHECK: s_endpgm
amdgcn.module @m target = #amdgcn.target<gfx1250> {
  amdgcn.kernel @cluster_ids {
    %cx = amdgcn.cluster_id x : !amdgcn.sgpr
    %wx = amdgcn.cluster_workgroup_id x : !amdgcn.sgpr
    %mx = amdgcn.cluster_workgroup_max_id x : !amdgcn.sgpr
    amdgcn.test_inst ins %cx, %wx, %mx : (!amdgcn.sgpr, !amdgcn.sgpr, !amdgcn.sgpr) -> ()
    amdgcn.end_kernel
  }

  // Launch: grid=(2,4,6), cluster=(1,2,3), block=(3,5,7).
  //
  // Each thread writes 12 4B values:
  //
  //   out[row] = [flat_block_id, block_id x/y/z,
  //               flat_cluster_id, cluster_id x/y/z,
  //               flat_thread_id, thread_id x/y/z]
  //
  // CHECK-LABEL: cluster_id_probe:
  //       CHECK: s_bfe_u32 {{.*}} ttmp9
  //       CHECK: s_bfe_u32 {{.*}} ttmp7
  //       CHECK: s_bfe_u32 {{.*}} ttmp6
  //       CHECK: global_store_b32
  //       CHECK: s_endpgm
  amdgcn.kernel @cluster_id_probe arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {cluster_dims = array<i32: 1, 2, 3>,
                 grid_dims   = array<i32: 2, 4, 6>,
                 block_dims  = array<i32: 3, 5, 7>,
                 wavefront_size32,
                 shared_memory_size = 0 : i32} {
    %out = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %c0 = arith.constant 0 : i32

    %bx = amdgcn.block_id   x : !amdgcn.sgpr
    %by = amdgcn.block_id   y : !amdgcn.sgpr
    %bz = amdgcn.block_id   z : !amdgcn.sgpr
    %cx = amdgcn.cluster_id x : !amdgcn.sgpr
    %cy = amdgcn.cluster_id y : !amdgcn.sgpr
    %cz = amdgcn.cluster_id z : !amdgcn.sgpr
    %tx = amdgcn.thread_id  x : !amdgcn.vgpr
    %ty = amdgcn.thread_id  y : !amdgcn.vgpr
    %tz = amdgcn.thread_id  z : !amdgcn.vgpr

    %bx_i = lsir.from_reg %bx : !amdgcn.sgpr -> i32
    %by_i = lsir.from_reg %by : !amdgcn.sgpr -> i32
    %bz_i = lsir.from_reg %bz : !amdgcn.sgpr -> i32
    %cx_i = lsir.from_reg %cx : !amdgcn.sgpr -> i32
    %cy_i = lsir.from_reg %cy : !amdgcn.sgpr -> i32
    %cz_i = lsir.from_reg %cz : !amdgcn.sgpr -> i32
    %tx_i = lsir.from_reg %tx : !amdgcn.vgpr -> i32
    %ty_i = lsir.from_reg %ty : !amdgcn.vgpr -> i32
    %tz_i = lsir.from_reg %tz : !amdgcn.vgpr -> i32

    // grid=(2,4,6): flat_block = bx + 2 * (by + 4 * bz)
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c4 = arith.constant 4 : i32
    %c5 = arith.constant 5 : i32
    %c105 = arith.constant 105 : i32
    %bz4 = arith.muli %bz_i, %c4 : i32
    %by_bz4 = arith.addi %by_i, %bz4 : i32
    %bid_inner = arith.muli %c2, %by_bz4 : i32
    %flat_bid_i = arith.addi %bx_i, %bid_inner : i32

    // n_clusters=(2,2,2): flat_cluster = cx + 2 * (cy + 2 * cz)
    %cz2 = arith.muli %cz_i, %c2 : i32
    %cy_cz2 = arith.addi %cy_i, %cz2 : i32
    %cid_inner = arith.muli %c2, %cy_cz2 : i32
    %flat_cid_i = arith.addi %cx_i, %cid_inner : i32

    // block=(3,5,7): flat_thread = tx + 3 * (ty + 5 * tz)
    %tz5 = arith.muli %tz_i, %c5 : i32
    %ty_tz5 = arith.addi %ty_i, %tz5 : i32
    %tid_inner = arith.muli %c3, %ty_tz5 : i32
    %flat_tid_i = arith.addi %tx_i, %tid_inner : i32

    %bid_row = arith.muli %flat_bid_i, %c105 : i32
    %row_i = arith.addi %bid_row, %flat_tid_i : i32
    %row_idx = arith.index_cast %row_i : i32 to index

    // col 0: flat_block_id
    %off0_idx = affine.apply affine_map<(d0) -> (d0 * 48)>(%row_idx)
    %off0_i = arith.index_cast %off0_idx : index to i32
    %off0_v = lsir.to_reg %off0_i : i32 -> !amdgcn.vgpr
    %flat_bid_v = lsir.to_reg %flat_bid_i : i32 -> !amdgcn.vgpr
    %t0 = amdgcn.global_store_b32 data %flat_bid_v addr %out offset d(%off0_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f0 = amdgcn.wait_gfx1250 deps %t0 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 1: block_id x
    %off1_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 4)>(%row_idx)
    %off1_i = arith.index_cast %off1_idx : index to i32
    %off1_v = lsir.to_reg %off1_i : i32 -> !amdgcn.vgpr
    %bx_v = lsir.to_reg %bx_i : i32 -> !amdgcn.vgpr
    %t1 = amdgcn.global_store_b32 data %bx_v addr %out offset d(%off1_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f1 = amdgcn.wait_gfx1250 deps %t1 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 2: block_id y
    %off2_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 8)>(%row_idx)
    %off2_i = arith.index_cast %off2_idx : index to i32
    %off2_v = lsir.to_reg %off2_i : i32 -> !amdgcn.vgpr
    %by_v = lsir.to_reg %by_i : i32 -> !amdgcn.vgpr
    %t2 = amdgcn.global_store_b32 data %by_v addr %out offset d(%off2_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f2 = amdgcn.wait_gfx1250 deps %t2 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 3: block_id z
    %off3_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 12)>(%row_idx)
    %off3_i = arith.index_cast %off3_idx : index to i32
    %off3_v = lsir.to_reg %off3_i : i32 -> !amdgcn.vgpr
    %bz_v = lsir.to_reg %bz_i : i32 -> !amdgcn.vgpr
    %t3 = amdgcn.global_store_b32 data %bz_v addr %out offset d(%off3_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f3 = amdgcn.wait_gfx1250 deps %t3 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 4: flat_cluster_id
    %off4_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 16)>(%row_idx)
    %off4_i = arith.index_cast %off4_idx : index to i32
    %off4_v = lsir.to_reg %off4_i : i32 -> !amdgcn.vgpr
    %flat_cid_v = lsir.to_reg %flat_cid_i : i32 -> !amdgcn.vgpr
    %t4 = amdgcn.global_store_b32 data %flat_cid_v addr %out offset d(%off4_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f4 = amdgcn.wait_gfx1250 deps %t4 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 5: cluster_id x
    %off5_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 20)>(%row_idx)
    %off5_i = arith.index_cast %off5_idx : index to i32
    %off5_v = lsir.to_reg %off5_i : i32 -> !amdgcn.vgpr
    %cx_v = lsir.to_reg %cx_i : i32 -> !amdgcn.vgpr
    %t5 = amdgcn.global_store_b32 data %cx_v addr %out offset d(%off5_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f5 = amdgcn.wait_gfx1250 deps %t5 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 6: cluster_id y
    %off6_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 24)>(%row_idx)
    %off6_i = arith.index_cast %off6_idx : index to i32
    %off6_v = lsir.to_reg %off6_i : i32 -> !amdgcn.vgpr
    %cy_v = lsir.to_reg %cy_i : i32 -> !amdgcn.vgpr
    %t6 = amdgcn.global_store_b32 data %cy_v addr %out offset d(%off6_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f6 = amdgcn.wait_gfx1250 deps %t6 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 7: cluster_id z
    %off7_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 28)>(%row_idx)
    %off7_i = arith.index_cast %off7_idx : index to i32
    %off7_v = lsir.to_reg %off7_i : i32 -> !amdgcn.vgpr
    %cz_v = lsir.to_reg %cz_i : i32 -> !amdgcn.vgpr
    %t7 = amdgcn.global_store_b32 data %cz_v addr %out offset d(%off7_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f7 = amdgcn.wait_gfx1250 deps %t7 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 8: flat_thread_id
    %off8_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 32)>(%row_idx)
    %off8_i = arith.index_cast %off8_idx : index to i32
    %off8_v = lsir.to_reg %off8_i : i32 -> !amdgcn.vgpr
    %flat_tid_v = lsir.to_reg %flat_tid_i : i32 -> !amdgcn.vgpr
    %t8 = amdgcn.global_store_b32 data %flat_tid_v addr %out offset d(%off8_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f8 = amdgcn.wait_gfx1250 deps %t8 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 9: thread_id x
    %off9_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 36)>(%row_idx)
    %off9_i = arith.index_cast %off9_idx : index to i32
    %off9_v = lsir.to_reg %off9_i : i32 -> !amdgcn.vgpr
    %tx_v = lsir.to_reg %tx_i : i32 -> !amdgcn.vgpr
    %t9 = amdgcn.global_store_b32 data %tx_v addr %out offset d(%off9_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f9 = amdgcn.wait_gfx1250 deps %t9 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 10: thread_id y
    %off10_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 40)>(%row_idx)
    %off10_i = arith.index_cast %off10_idx : index to i32
    %off10_v = lsir.to_reg %off10_i : i32 -> !amdgcn.vgpr
    %ty_v = lsir.to_reg %ty_i : i32 -> !amdgcn.vgpr
    %t10 = amdgcn.global_store_b32 data %ty_v addr %out offset d(%off10_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f10 = amdgcn.wait_gfx1250 deps %t10 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 11: thread_id z
    %off11_idx = affine.apply affine_map<(d0) -> (d0 * 48 + 44)>(%row_idx)
    %off11_i = arith.index_cast %off11_idx : index to i32
    %off11_v = lsir.to_reg %off11_i : i32 -> !amdgcn.vgpr
    %tz_v = lsir.to_reg %tz_i : i32 -> !amdgcn.vgpr
    %t11 = amdgcn.global_store_b32 data %tz_v addr %out offset d(%off11_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %f11 = amdgcn.wait_gfx1250 deps %t11 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    amdgcn.end_kernel
  }

  // Launch: grid=(2,3,4), block=(3,5,7).
  //
  // Each thread writes 8 4B values
  //
  //   out[row] = [flat_block_id, block_id x/y/z, flat_thread_id, thread_id x/y/z]
  //
  // where row = flat_block_id * 105 + flat_thread_id.
  //
  // CHECK-LABEL: block_id_probe:
  //       CHECK: s_bfe_u32 {{.*}} ttmp9
  //       CHECK: s_bfe_u32 {{.*}} ttmp7
  //   CHECK-NOT: ttmp6
  //       CHECK: global_store_b32
  //       CHECK: s_endpgm
  amdgcn.kernel @block_id_probe arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {grid_dims  = array<i32: 2, 3, 4>,
                 block_dims = array<i32: 3, 5, 7>,
                 wavefront_size32,
                 shared_memory_size = 0 : i32} {
    %out = amdgcn.load_arg 0 : !amdgcn.sgpr<[? + 2]>
    %c0 = arith.constant 0 : i32

    %bx = amdgcn.block_id  x : !amdgcn.sgpr
    %by = amdgcn.block_id  y : !amdgcn.sgpr
    %bz = amdgcn.block_id  z : !amdgcn.sgpr
    %tx = amdgcn.thread_id x : !amdgcn.vgpr
    %ty = amdgcn.thread_id y : !amdgcn.vgpr
    %tz = amdgcn.thread_id z : !amdgcn.vgpr

    %bx_i = lsir.from_reg %bx : !amdgcn.sgpr -> i32
    %by_i = lsir.from_reg %by : !amdgcn.sgpr -> i32
    %bz_i = lsir.from_reg %bz : !amdgcn.sgpr -> i32
    %tx_i = lsir.from_reg %tx : !amdgcn.vgpr -> i32
    %ty_i = lsir.from_reg %ty : !amdgcn.vgpr -> i32
    %tz_i = lsir.from_reg %tz : !amdgcn.vgpr -> i32

    // grid=(2,3,4): flat_block = bx + 2 * (by + 3 * bz)
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c5 = arith.constant 5 : i32
    %c105 = arith.constant 105 : i32
    %bz3 = arith.muli %bz_i, %c3 : i32
    %by_bz3 = arith.addi %by_i, %bz3 : i32
    %bid_inner = arith.muli %c2, %by_bz3 : i32
    %flat_bid_i = arith.addi %bx_i, %bid_inner : i32

    // block=(3,5,7): flat_thread = tx + 3 * (ty + 5 * tz)
    %tz5 = arith.muli %tz_i, %c5 : i32
    %ty_tz5 = arith.addi %ty_i, %tz5 : i32
    %tid_inner = arith.muli %c3, %ty_tz5 : i32
    %flat_tid_i = arith.addi %tx_i, %tid_inner : i32

    %bid_row = arith.muli %flat_bid_i, %c105 : i32
    %row_i = arith.addi %bid_row, %flat_tid_i : i32
    %row_idx = arith.index_cast %row_i : i32 to index

    // col 0: flat_block_id
    %boff0_idx = affine.apply affine_map<(d0) -> (d0 * 32)>(%row_idx)
    %boff0_i = arith.index_cast %boff0_idx : index to i32
    %boff0_v = lsir.to_reg %boff0_i : i32 -> !amdgcn.vgpr
    %bflat_bid_v = lsir.to_reg %flat_bid_i : i32 -> !amdgcn.vgpr
    %bt0 = amdgcn.global_store_b32 data %bflat_bid_v addr %out offset d(%boff0_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf0 = amdgcn.wait_gfx1250 deps %bt0 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 1: block_id x
    %boff1_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 4)>(%row_idx)
    %boff1_i = arith.index_cast %boff1_idx : index to i32
    %boff1_v = lsir.to_reg %boff1_i : i32 -> !amdgcn.vgpr
    %bbx_v = lsir.to_reg %bx_i : i32 -> !amdgcn.vgpr
    %bt1 = amdgcn.global_store_b32 data %bbx_v addr %out offset d(%boff1_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf1 = amdgcn.wait_gfx1250 deps %bt1 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 2: block_id y
    %boff2_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 8)>(%row_idx)
    %boff2_i = arith.index_cast %boff2_idx : index to i32
    %boff2_v = lsir.to_reg %boff2_i : i32 -> !amdgcn.vgpr
    %bby_v = lsir.to_reg %by_i : i32 -> !amdgcn.vgpr
    %bt2 = amdgcn.global_store_b32 data %bby_v addr %out offset d(%boff2_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf2 = amdgcn.wait_gfx1250 deps %bt2 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 3: block_id z
    %boff3_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 12)>(%row_idx)
    %boff3_i = arith.index_cast %boff3_idx : index to i32
    %boff3_v = lsir.to_reg %boff3_i : i32 -> !amdgcn.vgpr
    %bbz_v = lsir.to_reg %bz_i : i32 -> !amdgcn.vgpr
    %bt3 = amdgcn.global_store_b32 data %bbz_v addr %out offset d(%boff3_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf3 = amdgcn.wait_gfx1250 deps %bt3 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 4: flat_thread_id
    %boff4_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 16)>(%row_idx)
    %boff4_i = arith.index_cast %boff4_idx : index to i32
    %boff4_v = lsir.to_reg %boff4_i : i32 -> !amdgcn.vgpr
    %bflat_tid_v = lsir.to_reg %flat_tid_i : i32 -> !amdgcn.vgpr
    %bt4 = amdgcn.global_store_b32 data %bflat_tid_v addr %out offset d(%boff4_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf4 = amdgcn.wait_gfx1250 deps %bt4 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 5: thread_id x
    %boff5_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 20)>(%row_idx)
    %boff5_i = arith.index_cast %boff5_idx : index to i32
    %boff5_v = lsir.to_reg %boff5_i : i32 -> !amdgcn.vgpr
    %btx_v = lsir.to_reg %tx_i : i32 -> !amdgcn.vgpr
    %bt5 = amdgcn.global_store_b32 data %btx_v addr %out offset d(%boff5_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf5 = amdgcn.wait_gfx1250 deps %bt5 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 6: thread_id y
    %boff6_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 24)>(%row_idx)
    %boff6_i = arith.index_cast %boff6_idx : index to i32
    %boff6_v = lsir.to_reg %boff6_i : i32 -> !amdgcn.vgpr
    %bty_v = lsir.to_reg %ty_i : i32 -> !amdgcn.vgpr
    %bt6 = amdgcn.global_store_b32 data %bty_v addr %out offset d(%boff6_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf6 = amdgcn.wait_gfx1250 deps %bt6 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    // col 7: thread_id z
    %boff7_idx = affine.apply affine_map<(d0) -> (d0 * 32 + 28)>(%row_idx)
    %boff7_i = arith.index_cast %boff7_idx : index to i32
    %boff7_v = lsir.to_reg %boff7_i : i32 -> !amdgcn.vgpr
    %btz_v = lsir.to_reg %tz_i : i32 -> !amdgcn.vgpr
    %bt7 = amdgcn.global_store_b32 data %btz_v addr %out offset d(%boff7_v) + c(%c0)
            : ins(!amdgcn.vgpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
    %bf7 = amdgcn.wait_gfx1250 deps %bt7 : !amdgcn.write_token<flat> -> !amdgcn.fence_token

    amdgcn.end_kernel
  }
}
