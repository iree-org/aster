// RUN: aster-opt --pass-pipeline="builtin.module(amdgcn.module(amdgcn.kernel(aster-scf-pipeline)))" %s \
// RUN:   | FileCheck %s

// Note: repro obtained from contrib/kittens/test/test_102_gemm_python_multitile_cdna4.py

// CHECK-LABEL:   kernel @gemm_cdna4
// CHECK:           %[[C4:.*]] = arith.constant 4 : index
// CHECK:           %[[PTR_A:.*]] = load_arg 0
// CHECK:           %[[PTR_B:.*]] = load_arg 1
// CHECK:           %[[BUFA:.*]] = make_buffer_rsrc %[[PTR_A]]
// CHECK:           %[[BUFB:.*]] = make_buffer_rsrc %[[PTR_B]]
//
// 4-stage pipeline (A_LOAD=0, B_LOAD=1, LDS_READ=2, COMPUTE=3) uses
// 6 LDS buffers: 3 for A rotation + 3 for B rotation.
// CHECK-COUNT-6:   alloc_lds 1024
//
// Prologue warms up 3 iterations; steady-state loop runs exactly 1 iter.
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           scf.for {{.*}} = %[[C3]] to %[[C4]]
//
// Kernel loop body: A_LOAD, B_LOAD, MFMA.
// CHECK:             buffer_load_lds_dwordx4 addr %[[BUFA]]
// CHECK:             buffer_load_lds_dwordx4 addr %[[BUFB]]
// CHECK:             v_mfma_f32_16x16x32_f16
//
// The scf.yield rotates 6 LDS slot iter_args by position:
//   A slots: %arg23, %arg24, %arg22  (left-rotate by 1)
//   B slots: %arg26, %arg27, %arg25  (left-rotate by 1)
//
// The loop body's B G2S wrote to %arg25. After yield, %arg25 lands in
// result position #26. The drain reads #26 for its B_LOAD M0:
// CHECK:             scf.yield
//
// Issue:
// ======
// Drain B_LOAD M0 comes from scf.for result #26 = %arg25 after
// rotation = the SAME LDS slot the loop body just wrote to. The drain
// should advance to the next rotation slot (#24 or #25), not reuse #26.
//
// This overwrites B tile data that later drain MFMAs still need to read.
//
// CHECK:           lsir.to_reg %{{.*}}#26
// CHECK:           buffer_load_lds_dwordx4 addr %[[BUFB]]
// CHECK-NOT:       buffer_load_lds_dwordx4 addr %[[BUFA]]
// Drain flushes 3 in-flight MFMAs.
// CHECK:           v_mfma_f32_16x16x32_f16
// CHECK:           v_mfma_f32_16x16x32_f16
// CHECK:           v_mfma_f32_16x16x32_f16
// CHECK:           end_kernel

#map = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 64 + ((s3 + s1 * s4) * s0) * 64 + (((s2 + (s3 + s1 * s4) * s0) mod 64) floordiv 16) * 8 - ((s2 + (s3 + s1 * s4) * s0) floordiv 16) * 1024)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 64 + ((s3 + s1 * s4) * s0) * 64 + (((s2 + (s3 + s1 * s4) * s0) mod 64) floordiv 16) * 8 - ((s2 + (s3 + s1 * s4) * s0) floordiv 16) * 1024 + 32)>
#map3 = affine_map<()[s0, s1, s2, s3, s4] -> ((s2 + (s3 + s1 * s4) * s0) mod 64)>
#map4 = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 16 + ((s3 + s1 * s4) * s0) * 16 + (((s2 + (s3 + s1 * s4) * s0) mod 64) floordiv 4) * 64 - ((s2 + (s3 + s1 * s4) * s0) floordiv 4) * 64)>
#map5 = affine_map<()[s0, s1, s2] -> (s0 + s1 + s2)>
#map6 = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 4 + ((s3 + s1 * s4) * s0) * 4 + ((s2 + (s3 + s1 * s4) * s0) floordiv 16) * 192 - (((s2 + (s3 + s1 * s4) * s0) floordiv 16) floordiv 4) * 1024)>
#map7 = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 4 + ((s3 + s1 * s4) * s0) * 4 + ((s2 + (s3 + s1 * s4) * s0) floordiv 16) * 192 - (((s2 + (s3 + s1 * s4) * s0) floordiv 16) floordiv 4) * 1024 + 64)>
#map8 = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 4 + ((s3 + s1 * s4) * s0) * 4 + ((s2 + (s3 + s1 * s4) * s0) floordiv 16) * 192 - (((s2 + (s3 + s1 * s4) * s0) floordiv 16) floordiv 4) * 1024 + 128)>
#map9 = affine_map<()[s0, s1, s2, s3, s4] -> (s2 * 4 + ((s3 + s1 * s4) * s0) * 4 + ((s2 + (s3 + s1 * s4) * s0) floordiv 16) * 192 - (((s2 + (s3 + s1 * s4) * s0) floordiv 16) floordiv 4) * 1024 + 192)>
module {
  amdgcn.module @gemm_cdna4_mod target = <gfx950> {
    func.func private @_read_b(%arg0: index) -> (!aster_utils.any, !aster_utils.any, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>) {
      %c48_i32 = arith.constant 48 : i32
      %c4_i32 = arith.constant 4 : i32
      %c0_i32 = arith.constant 0 : i32
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %thread_id_z = gpu.thread_id z
      %block_dim_x = gpu.block_dim x
      %block_dim_y = gpu.block_dim y
      %0 = affine.apply #map()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %1 = arith.index_cast %0 : index to i32
      %2 = arith.shrui %1, %c4_i32 : i32
      %3 = arith.andi %2, %c48_i32 : i32
      %4 = arith.xori %1, %3 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.apply #map1()[%arg0, %5]
      %7 = amdgcn.alloca : !amdgcn.vgpr
      %8 = amdgcn.alloca : !amdgcn.vgpr
      %9 = amdgcn.make_register_range %7, %8 : !amdgcn.vgpr, !amdgcn.vgpr
      %10 = arith.index_cast %6 : index to i32
      %11 = lsir.to_reg %10 : i32 -> !amdgcn.vgpr
      %dest_res, %token = amdgcn.ds_read_b64 dest %9 addr %11 offset c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
      %12 = affine.apply #map2()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %13 = arith.index_cast %12 : index to i32
      %14 = arith.shrui %13, %c4_i32 : i32
      %15 = arith.andi %14, %c48_i32 : i32
      %16 = arith.xori %13, %15 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = affine.apply #map1()[%arg0, %17]
      %19 = amdgcn.alloca : !amdgcn.vgpr
      %20 = amdgcn.alloca : !amdgcn.vgpr
      %21 = amdgcn.make_register_range %19, %20 : !amdgcn.vgpr, !amdgcn.vgpr
      %22 = arith.index_cast %18 : index to i32
      %23 = lsir.to_reg %22 : i32 -> !amdgcn.vgpr
      %dest_res_0, %token_1 = amdgcn.ds_read_b64 dest %21 addr %23 offset c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
      %24 = aster_utils.to_any %dest_res : !amdgcn.vgpr<[? + 2]>
      %25 = aster_utils.to_any %dest_res_0 : !amdgcn.vgpr<[? + 2]>
      return %24, %25, %token, %token_1 : !aster_utils.any, !aster_utils.any, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>
    }
    func.func private @_read_a(%arg0: index) -> (!aster_utils.any, !aster_utils.any, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>) {
      %c48_i32 = arith.constant 48 : i32
      %c4_i32 = arith.constant 4 : i32
      %c0_i32 = arith.constant 0 : i32
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %thread_id_z = gpu.thread_id z
      %block_dim_x = gpu.block_dim x
      %block_dim_y = gpu.block_dim y
      %0 = affine.apply #map()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %1 = arith.index_cast %0 : index to i32
      %2 = arith.shrui %1, %c4_i32 : i32
      %3 = arith.andi %2, %c48_i32 : i32
      %4 = arith.xori %1, %3 : i32
      %5 = arith.index_cast %4 : i32 to index
      %6 = affine.apply #map1()[%arg0, %5]
      %7 = amdgcn.alloca : !amdgcn.vgpr
      %8 = amdgcn.alloca : !amdgcn.vgpr
      %9 = amdgcn.make_register_range %7, %8 : !amdgcn.vgpr, !amdgcn.vgpr
      %10 = arith.index_cast %6 : index to i32
      %11 = lsir.to_reg %10 : i32 -> !amdgcn.vgpr
      %dest_res, %token = amdgcn.ds_read_b64 dest %9 addr %11 offset c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
      %12 = affine.apply #map2()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %13 = arith.index_cast %12 : index to i32
      %14 = arith.shrui %13, %c4_i32 : i32
      %15 = arith.andi %14, %c48_i32 : i32
      %16 = arith.xori %13, %15 : i32
      %17 = arith.index_cast %16 : i32 to index
      %18 = affine.apply #map1()[%arg0, %17]
      %19 = amdgcn.alloca : !amdgcn.vgpr
      %20 = amdgcn.alloca : !amdgcn.vgpr
      %21 = amdgcn.make_register_range %19, %20 : !amdgcn.vgpr, !amdgcn.vgpr
      %22 = arith.index_cast %18 : index to i32
      %23 = lsir.to_reg %22 : i32 -> !amdgcn.vgpr
      %dest_res_0, %token_1 = amdgcn.ds_read_b64 dest %21 addr %23 offset c(%c0_i32) : outs(!amdgcn.vgpr<[? + 2]>) ins(!amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<shared>
      %24 = aster_utils.to_any %dest_res : !amdgcn.vgpr<[? + 2]>
      %25 = aster_utils.to_any %dest_res_0 : !amdgcn.vgpr<[? + 2]>
      return %24, %25, %token, %token_1 : !aster_utils.any, !aster_utils.any, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>
    }
    kernel @gemm_cdna4 arguments <[#amdgcn.buffer_arg<access = read_only>, #amdgcn.buffer_arg<access = read_only>, #amdgcn.buffer_arg<access = write_only>]> attributes {block_dims = array<i32: 64, 1, 1>, grid_dims = array<i32: 1, 1, 1>} {
      %c48_i32 = arith.constant 48 : i32
      %c4_i32 = arith.constant 4 : i32
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c0_i32 = arith.constant 0 : i32
      %c4096_i32 = arith.constant 4096 : i32
      %0 = load_arg 0 : !amdgcn.sgpr<[? + 2]>
      %1 = load_arg 1 : !amdgcn.sgpr<[? + 2]>
      %2 = load_arg 2 : !amdgcn.sgpr<[? + 2]>
      amdgcn.s_waitcnt lgkmcnt = 0
      %3 = alloca : !amdgcn.sgpr
      %4 = s_mov_b32 outs(%3) ins(%c4096_i32) : outs(!amdgcn.sgpr) ins(i32)
      %5 = alloca : !amdgcn.sgpr
      %6 = s_mov_b32 outs(%5) ins(%c4096_i32) : outs(!amdgcn.sgpr) ins(i32)
      %7 = make_buffer_rsrc %0, %4, %c0_i32, cache_swizzle = false, swizzle_enable = false, flags = 149796 : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> <[? + 4]>
      %8 = make_buffer_rsrc %1, %6, %c0_i32, cache_swizzle = false, swizzle_enable = false, flags = 149796 : (!amdgcn.sgpr<[? + 2]>, !amdgcn.sgpr, i32) -> <[? + 4]>
      %9 = alloca : !amdgcn.sgpr
      %10 = s_mov_b32 outs(%9) ins(%c0_i32) : outs(!amdgcn.sgpr) ins(i32)
      %11 = alloca : !amdgcn.m0<0>
      %thread_id_x = gpu.thread_id x
      %thread_id_y = gpu.thread_id y
      %thread_id_z = gpu.thread_id z
      %block_dim_x = gpu.block_dim x
      %block_dim_y = gpu.block_dim y
      %12 = affine.apply #map3()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %13 = affine.apply #map4()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %14 = arith.index_cast %13 : index to i32
      %15 = arith.shrui %14, %c4_i32 : i32
      %16 = arith.andi %15, %c48_i32 : i32
      %17 = arith.xori %14, %16 : i32
      %18 = arith.index_cast %17 : i32 to index
      %19:2 = affine.delinearize_index %12 into (16, 4) : index, index
      %20 = affine.linearize_index_by_strides[%19#0] by (192) : index
      %alloca = memref.alloca() : memref<1x!amdgcn.agpr<[? + 4]>>
      %21 = alloca : !amdgcn.agpr
      %22 = amdgcn.v_accvgpr_write outs(%21) ins(%c0_i32) : outs(!amdgcn.agpr) ins(i32)
      %23 = alloca : !amdgcn.agpr
      %24 = amdgcn.v_accvgpr_write outs(%23) ins(%c0_i32) : outs(!amdgcn.agpr) ins(i32)
      %25 = alloca : !amdgcn.agpr
      %26 = amdgcn.v_accvgpr_write outs(%25) ins(%c0_i32) : outs(!amdgcn.agpr) ins(i32)
      %27 = alloca : !amdgcn.agpr
      %28 = amdgcn.v_accvgpr_write outs(%27) ins(%c0_i32) : outs(!amdgcn.agpr) ins(i32)
      %29 = make_register_range %22, %24, %26, %28 : !amdgcn.agpr, !amdgcn.agpr, !amdgcn.agpr, !amdgcn.agpr
      memref.store %29, %alloca[%c0] : memref<1x!amdgcn.agpr<[? + 4]>>
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %48 = amdgcn.alloc_lds 1024 {sched.stage = 0 : i32}
        %49 = amdgcn.get_lds_offset %48 {sched.stage = 0 : i32} : i32
        %50 = arith.index_cast %49 {sched.stage = 0 : i32} : i32 to index
        %51 = amdgcn.alloc_lds 1024 {sched.stage = 0 : i32}
        %52 = amdgcn.get_lds_offset %51 {sched.stage = 0 : i32} : i32
        %53 = arith.index_cast %52 {sched.stage = 0 : i32} : i32 to index
        %54 = affine.linearize_index_by_strides[%arg0] by (64) : index
        %alloca_0 = memref.alloca() : memref<1x!amdgcn.read_token<flat>>
        %55 = affine.apply #map5()[%54, %18, %20]
        %56 = lsir.to_reg %49 {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
        %57 = amdgcn.alloca {sched.stage = 0 : i32} : !amdgcn.sgpr
        %58 = amdgcn.v_readfirstlane_b32 outs(%57) ins(%56) {sched.stage = 0 : i32} : outs(!amdgcn.sgpr) ins(!amdgcn.vgpr)
        amdgcn.s_mov_b32 outs(%11) ins(%58) {sched.stage = 0 : i32} : outs(!amdgcn.m0<0>) ins(!amdgcn.sgpr)
        %59 = arith.index_cast %55 {sched.stage = 0 : i32} : index to i32
        %60 = lsir.to_reg %59 {sched.stage = 0 : i32} : i32 -> !amdgcn.vgpr
        %61 = amdgcn.buffer_load_lds_dwordx4 addr %7 m0 %11 offset u(%10) + off_idx(%60) + c(%c0_i32) {offen, sched.stage = 0 : i32} : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.m0<0>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
        memref.store %61, %alloca_0[%c0] {sched.stage = 0 : i32} : memref<1x!amdgcn.read_token<flat>>
        %alloca_1 = memref.alloca() : memref<1x!amdgcn.read_token<flat>>
        %62 = lsir.to_reg %52 {sched.stage = 1 : i32} : i32 -> !amdgcn.vgpr
        %63 = amdgcn.alloca {sched.stage = 1 : i32} : !amdgcn.sgpr
        %64 = amdgcn.v_readfirstlane_b32 outs(%63) ins(%62) {sched.stage = 1 : i32} : outs(!amdgcn.sgpr) ins(!amdgcn.vgpr)
        amdgcn.s_mov_b32 outs(%11) ins(%64) {sched.stage = 1 : i32} : outs(!amdgcn.m0<0>) ins(!amdgcn.sgpr)
        %65 = arith.index_cast %55 {sched.stage = 1 : i32} : index to i32
        %66 = lsir.to_reg %65 {sched.stage = 1 : i32} : i32 -> !amdgcn.vgpr
        %67 = amdgcn.buffer_load_lds_dwordx4 addr %8 m0 %11 offset u(%10) + off_idx(%66) + c(%c0_i32) {offen, sched.stage = 1 : i32} : ins(!amdgcn.sgpr<[? + 4]>, !amdgcn.m0<0>, !amdgcn.sgpr, !amdgcn.vgpr) mods(i32) -> !amdgcn.read_token<flat>
        memref.store %67, %alloca_1[%c0] {sched.stage = 1 : i32} : memref<1x!amdgcn.read_token<flat>>
        %68 = memref.load %alloca_0[%c0] {sched.stage = 2 : i32} : memref<1x!amdgcn.read_token<flat>>
        amdgcn.wait deps %68 {sched.stage = 2 : i32} : !amdgcn.read_token<flat>
        %69 = memref.load %alloca_1[%c0] {sched.stage = 2 : i32} : memref<1x!amdgcn.read_token<flat>>
        amdgcn.wait deps %69 {sched.stage = 2 : i32} : !amdgcn.read_token<flat>
        amdgcn.s_barrier {sched.stage = 2 : i32}
        %alloca_2 = memref.alloca() : memref<2x!aster_utils.any>
        %alloca_3 = memref.alloca() : memref<2x!amdgcn.read_token<shared>>
        %70:4 = func.call @_read_a(%50) {sched.stage = 2 : i32} : (index) -> (!aster_utils.any, !aster_utils.any, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>)
        memref.store %70#0, %alloca_2[%c0] {sched.stage = 2 : i32} : memref<2x!aster_utils.any>
        memref.store %70#1, %alloca_2[%c1] {sched.stage = 2 : i32} : memref<2x!aster_utils.any>
        memref.store %70#2, %alloca_3[%c0] {sched.stage = 2 : i32} : memref<2x!amdgcn.read_token<shared>>
        memref.store %70#3, %alloca_3[%c1] {sched.stage = 2 : i32} : memref<2x!amdgcn.read_token<shared>>
        amdgcn.dealloc_lds %48 {sched.stage = 2 : i32}
        %alloca_4 = memref.alloca() : memref<2x!aster_utils.any>
        %alloca_5 = memref.alloca() : memref<2x!amdgcn.read_token<shared>>
        %71:4 = func.call @_read_b(%53) {sched.stage = 2 : i32} : (index) -> (!aster_utils.any, !aster_utils.any, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>)
        memref.store %71#0, %alloca_4[%c0] {sched.stage = 2 : i32} : memref<2x!aster_utils.any>
        memref.store %71#1, %alloca_4[%c1] {sched.stage = 2 : i32} : memref<2x!aster_utils.any>
        memref.store %71#2, %alloca_5[%c0] {sched.stage = 2 : i32} : memref<2x!amdgcn.read_token<shared>>
        memref.store %71#3, %alloca_5[%c1] {sched.stage = 2 : i32} : memref<2x!amdgcn.read_token<shared>>
        amdgcn.dealloc_lds %51 {sched.stage = 2 : i32}
        %72 = memref.load %alloca_3[%c0] {sched.stage = 3 : i32} : memref<2x!amdgcn.read_token<shared>>
        %73 = memref.load %alloca_3[%c1] {sched.stage = 3 : i32} : memref<2x!amdgcn.read_token<shared>>
        %74 = memref.load %alloca_5[%c0] {sched.stage = 3 : i32} : memref<2x!amdgcn.read_token<shared>>
        %75 = memref.load %alloca_5[%c1] {sched.stage = 3 : i32} : memref<2x!amdgcn.read_token<shared>>
        amdgcn.wait deps %72, %73, %74, %75 {sched.stage = 3 : i32} : !amdgcn.read_token<shared>, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>, !amdgcn.read_token<shared>
        %76 = memref.load %alloca[%c0] {sched.stage = 3 : i32} : memref<1x!amdgcn.agpr<[? + 4]>>
        %77 = memref.load %alloca_2[%c0] {sched.stage = 3 : i32} : memref<2x!aster_utils.any>
        %78 = aster_utils.from_any %77 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %79 = memref.load %alloca_2[%c1] {sched.stage = 3 : i32} : memref<2x!aster_utils.any>
        %80 = aster_utils.from_any %79 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %81:2 = amdgcn.split_register_range %78 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %82:2 = amdgcn.split_register_range %80 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %83 = amdgcn.make_register_range %81#0, %81#1, %82#0, %82#1 {sched.stage = 3 : i32} : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        %84 = memref.load %alloca_4[%c0] {sched.stage = 3 : i32} : memref<2x!aster_utils.any>
        %85 = aster_utils.from_any %84 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %86 = memref.load %alloca_4[%c1] {sched.stage = 3 : i32} : memref<2x!aster_utils.any>
        %87 = aster_utils.from_any %86 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %88:2 = amdgcn.split_register_range %85 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %89:2 = amdgcn.split_register_range %87 {sched.stage = 3 : i32} : !amdgcn.vgpr<[? + 2]>
        %90 = amdgcn.make_register_range %88#0, %88#1, %89#0, %89#1 {sched.stage = 3 : i32} : !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr, !amdgcn.vgpr
        %91 = amdgcn.v_mfma_f32_16x16x32_f16 outs(%76) ins(%83, %90, %76) {sched.stage = 3 : i32} : outs(!amdgcn.agpr<[? + 4]>) ins(!amdgcn.vgpr<[? + 4]>, !amdgcn.vgpr<[? + 4]>, !amdgcn.agpr<[? + 4]>)
        memref.store %91, %alloca[%c0] {sched.stage = 3 : i32} : memref<1x!amdgcn.agpr<[? + 4]>>
        amdgcn.s_barrier {sched.stage = 3 : i32}
      }
      %30 = memref.load %alloca[%c0] : memref<1x!amdgcn.agpr<[? + 4]>>
      %31:4 = split_register_range %30 : !amdgcn.agpr<[? + 4]>
      %32 = affine.apply #map6()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %33 = arith.index_cast %32 : index to i32
      %34 = lsir.to_reg %33 : i32 -> !amdgcn.vgpr
      %c0_i32_gs = arith.constant 0 : i32
      %35 = amdgcn.global_store_dword data %31#0 addr %2 offset d(%34) + c(%c0_i32_gs) : ins(!amdgcn.agpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
      %36 = affine.apply #map7()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %37 = arith.index_cast %36 : index to i32
      %38 = lsir.to_reg %37 : i32 -> !amdgcn.vgpr
      %39 = amdgcn.global_store_dword data %31#1 addr %2 offset d(%38) + c(%c0_i32_gs) : ins(!amdgcn.agpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
      %40 = affine.apply #map8()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %41 = arith.index_cast %40 : index to i32
      %42 = lsir.to_reg %41 : i32 -> !amdgcn.vgpr
      %43 = amdgcn.global_store_dword data %31#2 addr %2 offset d(%42) + c(%c0_i32_gs) : ins(!amdgcn.agpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
      %44 = affine.apply #map9()[%block_dim_x, %block_dim_y, %thread_id_x, %thread_id_y, %thread_id_z]
      %45 = arith.index_cast %44 : index to i32
      %46 = lsir.to_reg %45 : i32 -> !amdgcn.vgpr
      %47 = amdgcn.global_store_dword data %31#3 addr %2 offset d(%46) + c(%c0_i32_gs) : ins(!amdgcn.agpr, !amdgcn.sgpr<[? + 2]>, !amdgcn.vgpr) mods(i32) -> !amdgcn.write_token<flat>
      end_kernel
    }
  }
}
