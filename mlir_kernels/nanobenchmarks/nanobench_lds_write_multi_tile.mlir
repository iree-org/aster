// Nanobenchmark for @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait
// Uses garbage values in registers (no verification).

// From descriptors.mlir
!vx2 = !amdgcn.vgpr<[? + 2]>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!return_value_descriptor_1d_vx2 = !aster_utils.struct<memref: memref<?x!vx2>, offset: index>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From multi-tile-copies.mlir
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    !lds_position_descriptor_2level_2d, index, index, !return_value_descriptor_1d_vx2)

  amdgcn.kernel @nanobench_lds_write_multi_tile
  attributes {shared_memory_size = {{LDS_SIZE}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters matching typical GEMM usage
    %NT_I = arith.constant 2 : index
    %NT_J = arith.constant 4 : index

    // Number of outer iterations
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index

    // Number of tiles
    %num_tiles = arith.constant 8 : index  // NT_I * NT_J = 2 * 4 = 8

    // Allocate memref for library functions (garbage values fine, linearized)
    %load_memref_static = memref.alloca() : memref<8x!vx2>
    %load_memref = memref.cast %load_memref_static : memref<8x!vx2> to memref<?x!vx2>

    // LDS descriptor: lds_base=0, mm_pos=0, nn_pos=0 (base positions)
    %lds_stride_bytes = arith.constant 256 : index // SIZE_J * 2 bytes
    %elt_size_lds = arith.constant 2 : index
    %lds_desc = aster_utils.struct_create(%c0, %c0, %c0, %lds_stride_bytes, %elt_size_lds) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d

    // Create return value descriptor: memref + offset=0
    %result_desc = aster_utils.struct_create(%load_memref, %c0) : (memref<?x!vx2>, index) -> !return_value_descriptor_1d_vx2

    // Outer timing loop - repeat the tile iterations NUM_ITERS times
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      // Call the LDS write function with garbage register values
      func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
        %lds_desc, %NT_I, %NT_J, %result_desc)
        : (!lds_position_descriptor_2level_2d, index, index, !return_value_descriptor_1d_vx2) -> ()
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }
}
