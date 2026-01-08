// Nanobenchmark for @maybe_lds_write_multi_tile_coalesced
// Uses garbage values in registers (no verification).

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // Library declarations
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
    index, index, index, index, index, index, memref<?x!vx2>)

  func.func private @maybe_lds_write_multi_tile_coalesced(
    index, index, index, index,
    index, index, index,
    index, index,
    index, index,
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
          func.call @maybe_lds_write_multi_tile_coalesced(
            %c0, %ii, %jj, %c0,           // k, ii, jj, cond_iter
            %K, %II, %JJ,                 // K, II, JJ
            %NT_I, %NT_J,                 // NT_I, NT_J
            %c0, %SIZE_J,                 // lds_base_off, SIZE_J
            %load_memref)                 // load_memref
            : (index, index, index, index, index, index, index, index, index,
               index, index, memref<?x?x!vx2>) -> ()
        } {amdgcn.constexpr}
      } {amdgcn.constexpr}
    } {amdgcn.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    amdgcn.end_kernel
  }
}

