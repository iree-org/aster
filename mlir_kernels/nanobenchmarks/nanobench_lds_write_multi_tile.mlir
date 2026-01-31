// Nanobenchmark for @maybe_lds_write_wave_multi_tile_256xf16
// Uses garbage values in registers (no verification).

// From descriptors.mlir
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - k: outer loop index (for indexing load_memref -> mem2reg)
//   - cond_iter: condition index (execute only when cond_iter == 0)
//   - NT_I, NT_J: multi-tile factors (process NT_I x NT_J tiles at once)
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From copies.mlir
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    !lds_position_descriptor_2level_2d, index, index, memref<?x?x!vx2>)
  // From conditional-multi-tile-copies.mlir
  func.func private @maybe_lds_write_wave_multi_tile_256xf16(
    !conditional_execution_descriptor_2d,
    !lds_position_descriptor_2d,
    memref<?x?x!vx2>
  )

  amdgcn.kernel @nanobench_lds_write_multi_tile
  attributes {shared_memory_size = {{LDS_SIZE}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters matching typical GEMM usage
    %K = arith.constant 1 : index
    %II = arith.constant 4 : index
    %JJ = arith.constant 8 : index
    %NT_I = arith.constant 2 : index
    %NT_J = arith.constant 4 : index
    %SIZE_J = arith.constant 128 : index

    // Number of outer iterations
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    // Allocate memref for library functions (garbage values fine)
    %load_memref_static = memref.alloca() : memref<1x8x!vx2>
    %load_memref = memref.cast %load_memref_static : memref<1x8x!vx2> to memref<?x?x!vx2>

    // Outer timing loop - repeat the tile iterations NUM_ITERS times
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Inner loop over all tile indices
      scf.for %ii = %c0 to %II step %c1 {
        scf.for %jj = %c0 to %JJ step %c1 {
          // Call the LDS write function with garbage register values
          // Create conditional execution descriptor (k=0, cond_iter=0)
          %cond_desc = aster_utils.struct_create(%c0, %c0, %NT_I, %NT_J) : (index, index, index, index) -> !conditional_execution_descriptor_2d
          // LDS descriptor: lds_base=0, m_pos=ii, n_pos=jj (tile indices)
          %lds_stride_bytes = arith.constant 256 : index // SIZE_J * 2 bytes
          %elt_size_lds = arith.constant 2 : index
          %lds_desc = aster_utils.struct_create(%c0, %ii, %jj, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2d
          func.call @maybe_lds_write_wave_multi_tile_256xf16(
            %cond_desc,                   // conditional_execution_descriptor_2d
            %lds_desc,                    // lds_position_descriptor_2d
            %load_memref)                 // load_memref
            : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }
}
