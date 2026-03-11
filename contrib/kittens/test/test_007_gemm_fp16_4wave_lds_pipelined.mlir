// 4-wave GEMM kernel with pipelined LDS (16x32 tiles) + AGPR accumulators:
// C[32x32] = A[32xK] @ B[32xK]^T
//
// Uses lds_16x64_b.mlir: dwordx4 global loads, XOR-swizzled LDS writes/reads.
// Each 16x32 tile covers K=32, yielding 2 MFMA K-steps (K0 at byte-offset 0, K1 at 32).
//
// 2x2 wave grid (waves_m=2, waves_n=2).
//
// 4-stage pipeline:
//   STAGE_GLOBAL_LOAD: alloc + dwordx4 global loads
//   STAGE_DS_WRITE:    store to LDS (2 tokens per tile)
//   STAGE_DS_READ:     wait + barrier + read K0/K1 sub-tiles
//   STAGE_COMPUTE:     2 MFMAs (K0 + K1) + dealloc
//
// Template parameters:
//   {{K}}, {{K_TILES}}, {{STRIDE_AB}}
//   {{STAGE_GLOBAL_LOAD}}, {{STAGE_DS_WRITE}}, {{STAGE_DS_READ}}, {{STAGE_COMPUTE}}

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !ax4
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!index_pair = !aster_utils.struct<i: index, j: index>

amdgcn.module @kittens_gemm_4wave_lds_pipelined target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  func.func private @wave_id() -> index

  // From compute_16x16_f16.mlir (AGPR)
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_global_C_mfma_f32_16x16x16_f16(!rt_C_f32, !sx2, index, index, index)

  // From lds_16x64_b.mlir
  func.func private @load_global_tile_16x64_b(!sx2, index, index, index) -> !future_global_read
  func.func private @store_global_tile_to_lds_16x64_b(index, !future_global_read) -> (!lds_write_token, !lds_write_token)
  func.func private @load_lds_A_swizzled(index, index, index) -> !future_lds_read
  func.func private @load_lds_B_swizzled(index, index, index) -> !future_lds_read
  func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

  amdgcn.kernel @gemm_4wave_lds_pipelined arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = 0 : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index  // K1 byte offset within LDS row
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant 128 : index
    %K_tiles = arith.constant {{K_TILES}} : index

    // Wave position in 2x2 grid
    %wid = func.call @wave_id() : () -> index
    %wave_m, %wave_n = affine.delinearize_index %wid into (%c2, %c2) : index, index
    %m_offset = affine.apply affine_map<()[wm] -> (wm * 16)>()[%wave_m]
    %n_offset = affine.apply affine_map<()[wn] -> (wn * 16)>()[%wave_n]

    %C_init = func.call @zero_C() : () -> !rt_C_f32

    %C_final = scf.for %k = %c0 to %K_tiles step %c1 iter_args(%acc = %C_init) -> (!rt_C_f32) {
      %k_offset = affine.apply affine_map<(k) -> (k * 32)>(%k)

      // === Stage GLOBAL_LOAD ===
      %lds_a0_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_a1_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_b0_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
      %lds_b1_h = amdgcn.alloc_lds 1024 {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}

      %A_gfut = func.call @load_global_tile_16x64_b(%A_ptr, %m_offset, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read
      %B_gfut = func.call @load_global_tile_16x64_b(%B_ptr, %n_offset, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_GLOBAL_LOAD}} : i32}
          : (!sx2, index, index, index) -> !future_global_read

      // === Stage DS_WRITE ===
      // Select this wave's A/B LDS buffer via wave_m/wave_n
      %lds_A0 = amdgcn.get_lds_offset %lds_a0_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_A1 = amdgcn.get_lds_offset %lds_a1_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_B0 = amdgcn.get_lds_offset %lds_b0_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index
      %lds_B1 = amdgcn.get_lds_offset %lds_b1_h {sched.stage = {{STAGE_DS_WRITE}} : i32} : index

      %lds_A_stride = affine.apply affine_map<()[a0, a1] -> (a1 - a0)>()[%lds_A0, %lds_A1]
      %lds_A = affine.apply affine_map<()[base, wm, stride] -> (base + wm * stride)>
          ()[%lds_A0, %wave_m, %lds_A_stride]
      %lds_B_stride = affine.apply affine_map<()[b0, b1] -> (b1 - b0)>()[%lds_B0, %lds_B1]
      %lds_B = affine.apply affine_map<()[base, wn, stride] -> (base + wn * stride)>
          ()[%lds_B0, %wave_n, %lds_B_stride]

      %tok_A0, %tok_A1 = func.call @store_global_tile_to_lds_16x64_b(%lds_A, %A_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %tok_B0, %tok_B1 = func.call @store_global_tile_to_lds_16x64_b(%lds_B, %B_gfut)
          {sched.stage = {{STAGE_DS_WRITE}} : i32}
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)

      // === Stage DS_READ: Wait + barrier + read K0/K1 ===
      amdgcn.wait deps %tok_A0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_A1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B0 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.wait deps %tok_B1 {sched.stage = {{STAGE_DS_READ}} : i32} : !lds_write_token
      amdgcn.sopp.sopp #amdgcn.inst<s_barrier> {sched.stage = {{STAGE_DS_READ}} : i32}

      // K0 sub-tiles (byte offset 0 within LDS row)
      %A0_fut = func.call @load_lds_A_swizzled(%lds_A, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B0_fut = func.call @load_lds_B_swizzled(%lds_B, %c0, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read

      // K1 sub-tiles (byte offset 32 within LDS row)
      %A1_fut = func.call @load_lds_A_swizzled(%lds_A, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read
      %B1_fut = func.call @load_lds_B_swizzled(%lds_B, %c32, %c2)
          {sched.stage = {{STAGE_DS_READ}} : i32} : (index, index, index) -> !future_lds_read

      // === Stage COMPUTE: 2 MFMAs (K0 + K1) ===
      %A0 = func.call @get_lds_read_value_vx2(%A0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B0 = func.call @get_lds_read_value_vx2(%B0_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %acc_k0 = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %acc)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      %A1 = func.call @get_lds_read_value_vx2(%A1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_A_f16
      %B1 = func.call @get_lds_read_value_vx2(%B1_fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32} : (!future_lds_read) -> !rt_B_f16
      %acc_k1 = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %acc_k0)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32

      amdgcn.dealloc_lds %lds_a0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_a1_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b0_h {sched.stage = {{STAGE_COMPUTE}} : i32}
      amdgcn.dealloc_lds %lds_b1_h {sched.stage = {{STAGE_COMPUTE}} : i32}

      scf.yield %acc_k1 : !rt_C_f32
    }

    func.call @store_global_C_mfma_f32_16x16x16_f16(%C_final, %C_ptr, %m_offset, %n_offset, %stride_C)
        : (!rt_C_f32, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }
}
