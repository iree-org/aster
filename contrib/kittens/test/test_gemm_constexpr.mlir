// Constexpr multi-tile GEMM with LDS + pipelining:
// C[{{M_DIM}}x{{N_DIM}}] = A[{{M_DIM}}xK] @ B[{{N_DIM}}xK]^T
//
// Unified template for arbitrary M_T x N_T tile grids (1x1 through 4x4+).
// ALL loops are constexpr -- zero structural markers, only scalar substitutions.
//
// Key design:
//   - Accumulators: memref<MN x !rt_C_f32> alloca OUTSIDE K-loop (no iter_args)
//   - LDS handles: memref<M_T x !amdgcn.lds_buffer> in constexpr loops
//   - After pipeline: constexpr-expansion -> sroa -> mem2reg -> promote-loop-carried-memrefs
//     produces IR identical to hand-written reference kernels.
//
// Scalar substitutions:
//   M_T, N_T, MN  - tile dimensions and product
//   M_DIM, N_DIM  - output matrix dimensions (M_T*16, N_T*16)
//   K, K_TILES, STRIDE_AB, STRIDE_C, SHARED_MEM
//   STAGE_LOAD, STAGE_SYNC, STAGE_COMPUTE

// Type aliases
!sx2 = !amdgcn.sgpr<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!rt_A_f16 = !vx2
!rt_B_f16 = !vx2
!rt_C_f32 = !vx4
!write_token = !amdgcn.write_token<flat>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>

amdgcn.module @kittens_gemm_constexpr target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token

  // From kittens/lds_16x16_f16.mlir - XOR swizzle mode
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @load_lds_B_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // Single-wave multi-tile GEMM with pipelined LDS (64 threads = 1 wave)
  amdgcn.kernel @gemm_constexpr arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = write_only>
  ]> attributes {shared_memory_size = {{SHARED_MEM}} : i32} {
    %A_ptr = amdgcn.load_arg 0 : !sx2
    %B_ptr = amdgcn.load_arg 1 : !sx2
    %C_ptr = amdgcn.load_arg 2 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c_M_T = arith.constant {{M_T}} : index
    %c_N_T = arith.constant {{N_T}} : index
    %c_MN  = arith.constant {{MN}} : index
    %stride_AB = arith.constant {{STRIDE_AB}} : index
    %stride_C = arith.constant {{STRIDE_C}} : index
    %K_tiles = arith.constant {{K_TILES}} : index

    // === Initialize accumulators (constexpr over M_T*N_T) ===
    // Stored in memref -- promote-loop-carried-memrefs converts to iter_args.
    %C_buf = memref.alloca() : memref<{{MN}} x !rt_C_f32>
    scf.for %i = %c0 to %c_MN step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : memref<{{MN}} x !rt_C_f32>
    } {aster.constexpr}

    // === K-loop (no iter_args -- accumulators live in C_buf) ===
    scf.for %k = %c0 to %K_tiles step %c1 {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // === Stage LOAD: Allocate LDS handles (constexpr) ===
      %lds_a_buf = memref.alloca() : memref<{{M_T}} x !amdgcn.lds_buffer>
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
        memref.store %h, %lds_a_buf[%mt] : memref<{{M_T}} x !amdgcn.lds_buffer>
      } {aster.constexpr}
      %lds_b_buf = memref.alloca() : memref<{{N_T}} x !amdgcn.lds_buffer>
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
        memref.store %h, %lds_b_buf[%nt] : memref<{{N_T}} x !amdgcn.lds_buffer>
      } {aster.constexpr}

      // Get LDS offsets (constexpr)
      %a_offsets = memref.alloca() : memref<{{M_T}} x index>
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %h = memref.load %lds_a_buf[%mt] : memref<{{M_T}} x !amdgcn.lds_buffer>
        %off = amdgcn.get_lds_offset %h {sched.stage = {{STAGE_LOAD}} : i32} : index
        memref.store %off, %a_offsets[%mt] : memref<{{M_T}} x index>
      } {aster.constexpr}
      %b_offsets = memref.alloca() : memref<{{N_T}} x index>
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %h = memref.load %lds_b_buf[%nt] : memref<{{N_T}} x !amdgcn.lds_buffer>
        %off = amdgcn.get_lds_offset %h {sched.stage = {{STAGE_LOAD}} : i32} : index
        memref.store %off, %b_offsets[%nt] : memref<{{N_T}} x index>
      } {aster.constexpr}

      // Stage LOAD: Global -> LDS for A (constexpr over M_T)
      %tok_a_buf = memref.alloca() : memref<{{M_T}} x !lds_write_token>
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %off = memref.load %a_offsets[%mt] : memref<{{M_T}} x index>
        %m_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%mt)
        %tok = func.call @load_global_to_lds_xor_swizzle_f16(%off, %A_ptr, %m_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_LOAD}} : i32}
            : (index, !sx2, index, index, index) -> !lds_write_token
        memref.store %tok, %tok_a_buf[%mt] : memref<{{M_T}} x !lds_write_token>
      } {aster.constexpr}

      // Stage LOAD: Global -> LDS for B (constexpr over N_T)
      %tok_b_buf = memref.alloca() : memref<{{N_T}} x !lds_write_token>
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %off = memref.load %b_offsets[%nt] : memref<{{N_T}} x index>
        %n_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%nt)
        %tok = func.call @load_global_to_lds_xor_swizzle_f16(%off, %B_ptr, %n_off, %k_offset, %stride_AB)
            {sched.stage = {{STAGE_LOAD}} : i32}
            : (index, !sx2, index, index, index) -> !lds_write_token
        memref.store %tok, %tok_b_buf[%nt] : memref<{{N_T}} x !lds_write_token>
      } {aster.constexpr}

      // === Stage SYNC: Wait for LDS writes (constexpr) ===
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %tok = memref.load %tok_a_buf[%mt] : memref<{{M_T}} x !lds_write_token>
        amdgcn.wait deps %tok {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      } {aster.constexpr}
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %tok = memref.load %tok_b_buf[%nt] : memref<{{N_T}} x !lds_write_token>
        amdgcn.wait deps %tok {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
      } {aster.constexpr}

      // Stage SYNC: LDS -> Register loads (constexpr)
      %a_fut_buf = memref.alloca() : memref<{{M_T}} x !future_lds_read>
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %off = memref.load %a_offsets[%mt] : memref<{{M_T}} x index>
        %fut = func.call @load_lds_A_xor_swizzle_f16(%off)
            {sched.stage = {{STAGE_SYNC}} : i32}
            : (index) -> !future_lds_read
        memref.store %fut, %a_fut_buf[%mt] : memref<{{M_T}} x !future_lds_read>
      } {aster.constexpr}
      %b_fut_buf = memref.alloca() : memref<{{N_T}} x !future_lds_read>
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %off = memref.load %b_offsets[%nt] : memref<{{N_T}} x index>
        %fut = func.call @load_lds_B_xor_swizzle_f16(%off)
            {sched.stage = {{STAGE_SYNC}} : i32}
            : (index) -> !future_lds_read
        memref.store %fut, %b_fut_buf[%nt] : memref<{{N_T}} x !future_lds_read>
      } {aster.constexpr}

      // === Stage COMPUTE: Extract values (constexpr) ===
      %a_vals = memref.alloca() : memref<{{M_T}} x !rt_A_f16>
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %fut = memref.load %a_fut_buf[%mt] : memref<{{M_T}} x !future_lds_read>
        %a = func.call @get_lds_A_f16(%fut)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_lds_read) -> !rt_A_f16
        memref.store %a, %a_vals[%mt] : memref<{{M_T}} x !rt_A_f16>
      } {aster.constexpr}
      %b_vals = memref.alloca() : memref<{{N_T}} x !rt_B_f16>
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %fut = memref.load %b_fut_buf[%nt] : memref<{{N_T}} x !future_lds_read>
        %b = func.call @get_lds_B_f16(%fut)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!future_lds_read) -> !rt_B_f16
        memref.store %b, %b_vals[%nt] : memref<{{N_T}} x !rt_B_f16>
      } {aster.constexpr}

      // Stage COMPUTE: MFMAs (constexpr over M_T x N_T)
      // C_buf is the accumulator memref -- reads/writes form loop-carried pattern.
      scf.for %mt = %c0 to %c_M_T step %c1 {
        scf.for %nt = %c0 to %c_N_T step %c1 {
          %a = memref.load %a_vals[%mt] : memref<{{M_T}} x !rt_A_f16>
          %b = memref.load %b_vals[%nt] : memref<{{N_T}} x !rt_B_f16>
          %idx = affine.apply affine_map<(d0, d1) -> (d0 * {{N_T}} + d1)>(%mt, %nt)
          %c_old = memref.load %C_buf[%idx] : memref<{{MN}} x !rt_C_f32>
          %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
              {sched.stage = {{STAGE_COMPUTE}} : i32}
              : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
          memref.store %c_new, %C_buf[%idx] : memref<{{MN}} x !rt_C_f32>
        } {aster.constexpr}
      } {aster.constexpr}

      // Stage COMPUTE: Dealloc LDS (constexpr)
      scf.for %mt = %c0 to %c_M_T step %c1 {
        %h = memref.load %lds_a_buf[%mt] : memref<{{M_T}} x !amdgcn.lds_buffer>
        amdgcn.dealloc_lds %h {sched.stage = {{STAGE_COMPUTE}} : i32}
      } {aster.constexpr}
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %h = memref.load %lds_b_buf[%nt] : memref<{{N_T}} x !amdgcn.lds_buffer>
        amdgcn.dealloc_lds %h {sched.stage = {{STAGE_COMPUTE}} : i32}
      } {aster.constexpr}
    }

    // === Store results (constexpr over M_T x N_T) ===
    scf.for %mt = %c0 to %c_M_T step %c1 {
      scf.for %nt = %c0 to %c_N_T step %c1 {
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * {{N_T}} + d1)>(%mt, %nt)
        %c_tile = memref.load %C_buf[%idx] : memref<{{MN}} x !rt_C_f32>
        %m_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%mt)
        %n_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%nt)
        %tok = func.call @store_C_f32(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !sx2, index, index, index) -> !write_token
        amdgcn.wait deps %tok : !write_token
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.end_kernel
  }
}
