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
//   - K-loop body factored into helper functions; aster-selective-inlining
//     expands them before constexpr-expansion runs.
//
// Scalar substitutions:
//   M_T, N_T, MN  - tile dimensions and product
//   M_DIM, N_DIM  - output matrix dimensions (M_T*16, N_T*16)
//   K, K_TILES, STRIDE_AB, STRIDE_C, SHARED_MEM
//   STAGE_LOAD, STAGE_SYNC, STAGE_COMPUTE

// Register type aliases
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

// Memref buffer type aliases (used in helper function signatures)
!lds_a_buf = memref<{{M_T}} x !amdgcn.lds_buffer>
!lds_b_buf = memref<{{N_T}} x !amdgcn.lds_buffer>
!off_a_buf = memref<{{M_T}} x index>
!off_b_buf = memref<{{N_T}} x index>
!tok_a_buf = memref<{{M_T}} x !lds_write_token>
!tok_b_buf = memref<{{N_T}} x !lds_write_token>
!fut_a_buf = memref<{{M_T}} x !future_lds_read>
!fut_b_buf = memref<{{N_T}} x !future_lds_read>
!vals_a_buf = memref<{{M_T}} x !rt_A_f16>
!vals_b_buf = memref<{{N_T}} x !rt_B_f16>
!c_buf = memref<{{MN}} x !rt_C_f32>

amdgcn.module @kittens_gemm_constexpr target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library functions (external, provided by preload library)
  func.func private @zero_C() -> !rt_C_f32
  func.func private @mfma_f32_16x16x16_f16(!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
  func.func private @store_C_f32(!rt_C_f32, !sx2, index, index, index) -> !write_token
  func.func private @load_global_to_lds_xor_swizzle_f16(index, !sx2, index, index, index) -> !lds_write_token
  func.func private @load_lds_A_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @load_lds_B_xor_swizzle_f16(index) -> !future_lds_read
  func.func private @get_lds_A_f16(!future_lds_read) -> !rt_A_f16
  func.func private @get_lds_B_f16(!future_lds_read) -> !rt_B_f16

  // === K-loop helper functions (inlined before constexpr expansion) ===

  // Allocate LDS handles for A and B tiles.
  func.func private @k_alloc_lds(%m_t: index, %n_t: index) -> (!lds_a_buf, !lds_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = memref.alloca() : !lds_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      memref.store %h, %a[%i] : !lds_a_buf
    } {aster.constexpr}
    %b = memref.alloca() : !lds_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %h = amdgcn.alloc_lds 512 {sched.stage = {{STAGE_LOAD}} : i32}
      memref.store %h, %b[%i] : !lds_b_buf
    } {aster.constexpr}
    return %a, %b : !lds_a_buf, !lds_b_buf
  }

  // Extract LDS offsets from handles.
  func.func private @k_get_lds_offsets(%m_t: index, %n_t: index,
      %lds_a: !lds_a_buf, %lds_b: !lds_b_buf) -> (!off_a_buf, !off_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a_off = memref.alloca() : !off_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %h = memref.load %lds_a[%i] : !lds_a_buf
      %off = amdgcn.get_lds_offset %h {sched.stage = {{STAGE_LOAD}} : i32} : index
      memref.store %off, %a_off[%i] : !off_a_buf
    } {aster.constexpr}
    %b_off = memref.alloca() : !off_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %h = memref.load %lds_b[%i] : !lds_b_buf
      %off = amdgcn.get_lds_offset %h {sched.stage = {{STAGE_LOAD}} : i32} : index
      memref.store %off, %b_off[%i] : !off_b_buf
    } {aster.constexpr}
    return %a_off, %b_off : !off_a_buf, !off_b_buf
  }

  // Load A and B tiles from global memory into LDS.
  func.func private @k_load_tiles_to_lds(%m_t: index, %n_t: index,
      %a_off: !off_a_buf, %b_off: !off_b_buf,
      %A_ptr: !sx2, %B_ptr: !sx2, %k_offset: index, %stride_AB: index)
      -> (!tok_a_buf, !tok_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tok_a = memref.alloca() : !tok_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %off = memref.load %a_off[%i] : !off_a_buf
      %m_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
      %tok = func.call @load_global_to_lds_xor_swizzle_f16(%off, %A_ptr, %m_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      memref.store %tok, %tok_a[%i] : !tok_a_buf
    } {aster.constexpr}
    %tok_b = memref.alloca() : !tok_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %off = memref.load %b_off[%i] : !off_b_buf
      %n_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%i)
      %tok = func.call @load_global_to_lds_xor_swizzle_f16(%off, %B_ptr, %n_off, %k_offset, %stride_AB)
          {sched.stage = {{STAGE_LOAD}} : i32}
          : (index, !sx2, index, index, index) -> !lds_write_token
      memref.store %tok, %tok_b[%i] : !tok_b_buf
    } {aster.constexpr}
    return %tok_a, %tok_b : !tok_a_buf, !tok_b_buf
  }

  // Wait for LDS writes then load tiles from LDS into register futures.
  func.func private @k_sync_and_read_lds(%m_t: index, %n_t: index,
      %tok_a: !tok_a_buf, %tok_b: !tok_b_buf,
      %a_off: !off_a_buf, %b_off: !off_b_buf)
      -> (!fut_a_buf, !fut_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %m_t step %c1 {
      %tok = memref.load %tok_a[%i] : !tok_a_buf
      amdgcn.wait deps %tok {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
    } {aster.constexpr}
    scf.for %i = %c0 to %n_t step %c1 {
      %tok = memref.load %tok_b[%i] : !tok_b_buf
      amdgcn.wait deps %tok {sched.stage = {{STAGE_SYNC}} : i32} : !lds_write_token
    } {aster.constexpr}
    %a_fut = memref.alloca() : !fut_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %off = memref.load %a_off[%i] : !off_a_buf
      %fut = func.call @load_lds_A_xor_swizzle_f16(%off)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read
      memref.store %fut, %a_fut[%i] : !fut_a_buf
    } {aster.constexpr}
    %b_fut = memref.alloca() : !fut_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %off = memref.load %b_off[%i] : !off_b_buf
      %fut = func.call @load_lds_B_xor_swizzle_f16(%off)
          {sched.stage = {{STAGE_SYNC}} : i32}
          : (index) -> !future_lds_read
      memref.store %fut, %b_fut[%i] : !fut_b_buf
    } {aster.constexpr}
    return %a_fut, %b_fut : !fut_a_buf, !fut_b_buf
  }

  // Extract register values from LDS read futures.
  func.func private @k_extract_lds_values(%m_t: index, %n_t: index,
      %a_fut: !fut_a_buf, %b_fut: !fut_b_buf)
      -> (!vals_a_buf, !vals_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a_vals = memref.alloca() : !vals_a_buf
    scf.for %i = %c0 to %m_t step %c1 {
      %fut = memref.load %a_fut[%i] : !fut_a_buf
      %a = func.call @get_lds_A_f16(%fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_A_f16
      memref.store %a, %a_vals[%i] : !vals_a_buf
    } {aster.constexpr}
    %b_vals = memref.alloca() : !vals_b_buf
    scf.for %i = %c0 to %n_t step %c1 {
      %fut = memref.load %b_fut[%i] : !fut_b_buf
      %b = func.call @get_lds_B_f16(%fut)
          {sched.stage = {{STAGE_COMPUTE}} : i32}
          : (!future_lds_read) -> !rt_B_f16
      memref.store %b, %b_vals[%i] : !vals_b_buf
    } {aster.constexpr}
    return %a_vals, %b_vals : !vals_a_buf, !vals_b_buf
  }

  // Compute MFMAs: accumulate A*B into C tiles.
  func.func private @k_compute_mfmas(%m_t: index, %n_t: index,
      %a_vals: !vals_a_buf, %b_vals: !vals_b_buf, %c_buf: !c_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %a = memref.load %a_vals[%mt] : !vals_a_buf
        %b = memref.load %b_vals[%nt] : !vals_b_buf
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * {{N_T}} + d1)>(%mt, %nt)
        %c_old = memref.load %c_buf[%idx] : !c_buf
        %c_new = func.call @mfma_f32_16x16x16_f16(%a, %b, %c_old)
            {sched.stage = {{STAGE_COMPUTE}} : i32}
            : (!rt_A_f16, !rt_B_f16, !rt_C_f32) -> !rt_C_f32
        memref.store %c_new, %c_buf[%idx] : !c_buf
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

  // Deallocate LDS handles.
  func.func private @k_dealloc_lds(%m_t: index, %n_t: index,
      %lds_a: !lds_a_buf, %lds_b: !lds_b_buf) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %m_t step %c1 {
      %h = memref.load %lds_a[%i] : !lds_a_buf
      amdgcn.dealloc_lds %h {sched.stage = {{STAGE_COMPUTE}} : i32}
    } {aster.constexpr}
    scf.for %i = %c0 to %n_t step %c1 {
      %h = memref.load %lds_b[%i] : !lds_b_buf
      amdgcn.dealloc_lds %h {sched.stage = {{STAGE_COMPUTE}} : i32}
    } {aster.constexpr}
    return
  }

  // Store C accumulator tiles to global memory.
  func.func private @store_c_tiles(%m_t: index, %n_t: index,
      %c_buf: !c_buf, %C_ptr: !sx2, %stride_C: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %mt = %c0 to %m_t step %c1 {
      scf.for %nt = %c0 to %n_t step %c1 {
        %idx = affine.apply affine_map<(d0, d1) -> (d0 * {{N_T}} + d1)>(%mt, %nt)
        %c_tile = memref.load %c_buf[%idx] : !c_buf
        %m_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%mt)
        %n_off = affine.apply affine_map<(d0) -> (d0 * 16)>(%nt)
        %tok = func.call @store_C_f32(%c_tile, %C_ptr, %m_off, %n_off, %stride_C)
            : (!rt_C_f32, !sx2, index, index, index) -> !write_token
        amdgcn.wait deps %tok : !write_token
      } {aster.constexpr}
    } {aster.constexpr}
    return
  }

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
    %C_buf = memref.alloca() : !c_buf
    scf.for %i = %c0 to %c_MN step %c1 {
      %z = func.call @zero_C() : () -> !rt_C_f32
      memref.store %z, %C_buf[%i] : !c_buf
    } {aster.constexpr}

    // === K-loop (no iter_args -- accumulators live in C_buf) ===
    scf.for %k = %c0 to %K_tiles step %c1 {
      %k_offset = affine.apply affine_map<(k) -> (k * 16)>(%k)

      // Stage LOAD: allocate LDS, get offsets, load global -> LDS
      %lds_a, %lds_b = func.call @k_alloc_lds(%c_M_T, %c_N_T)
          : (index, index) -> (!lds_a_buf, !lds_b_buf)
      %a_off, %b_off = func.call @k_get_lds_offsets(%c_M_T, %c_N_T, %lds_a, %lds_b)
          : (index, index, !lds_a_buf, !lds_b_buf) -> (!off_a_buf, !off_b_buf)
      %tok_a, %tok_b = func.call @k_load_tiles_to_lds(
          %c_M_T, %c_N_T, %a_off, %b_off, %A_ptr, %B_ptr, %k_offset, %stride_AB)
          : (index, index, !off_a_buf, !off_b_buf, !sx2, !sx2, index, index)
          -> (!tok_a_buf, !tok_b_buf)

      // Stage SYNC: wait for LDS writes, load LDS -> register futures
      %a_fut, %b_fut = func.call @k_sync_and_read_lds(
          %c_M_T, %c_N_T, %tok_a, %tok_b, %a_off, %b_off)
          : (index, index, !tok_a_buf, !tok_b_buf, !off_a_buf, !off_b_buf)
          -> (!fut_a_buf, !fut_b_buf)

      // Stage COMPUTE: extract register values from futures
      %a_vals, %b_vals = func.call @k_extract_lds_values(
          %c_M_T, %c_N_T, %a_fut, %b_fut)
          : (index, index, !fut_a_buf, !fut_b_buf) -> (!vals_a_buf, !vals_b_buf)

      // Stage COMPUTE: MFMAs (constexpr over M_T x N_T)
      func.call @k_compute_mfmas(%c_M_T, %c_N_T, %a_vals, %b_vals, %C_buf)
          : (index, index, !vals_a_buf, !vals_b_buf, !c_buf) -> ()

      // Stage COMPUTE: deallocate LDS
      func.call @k_dealloc_lds(%c_M_T, %c_N_T, %lds_a, %lds_b)
          : (index, index, !lds_a_buf, !lds_b_buf) -> ()
    }

    // === Store results ===
    func.call @store_c_tiles(%c_M_T, %c_N_T, %C_buf, %C_ptr, %stride_C)
        : (index, index, !c_buf, !sx2, index) -> ()

    amdgcn.end_kernel
  }
}
