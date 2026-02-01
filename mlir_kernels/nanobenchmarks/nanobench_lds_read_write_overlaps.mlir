// Nanobenchmark for multi-tile token-aware operations.
// Tests the pattern where each ds_write waits on its specific global_load,
// and each ds_read waits on its specific ds_write.
//
// This tests whether the wait optimization pass can compute optimal wait counts
// for fine-grained dependencies (e.g., vmcnt=N-1 for first wait, vmcnt=N-2 for second, etc.)

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>

!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>
!future_global_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>
!future_lds_write = !amdgcn.write_token<shared>
!future_lds_read_any = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read_descriptor_1d = !aster_utils.struct<memref: memref<?x!future_global_read_any>, offset: index>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations
  //===--------------------------------------------------------------------===//
  // From multi-tile-copies.mlir
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
    !tensor_position_descriptor_2level_2d, index, index,
    !future_global_read_descriptor_1d)

  // From copies.mlir - single-tile variants
  func.func private @lds_write_wave_256xf16_via_dwordx2_future(
    !lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write
  func.func private @lds_read_A_wave_16x16xf16_fragment_future(
    !lds_position_descriptor_2d, i1) -> !future_lds_read_any

  // From futures.mlir - wait helpers
  func.func private @get_global_load_value_vx2(!future_global_read_any) -> !vx2
  func.func private @get_global_load_value_vx2_1d(memref<?x!future_global_read_any>, index) -> !vx2
  func.func private @get_lds_read_value_vx2(!future_lds_read_any) -> !vx2
  func.func private @wait_lds_write(!future_lds_write)

  //===--------------------------------------------------------------------===//
  // Kernel: DS Load/Store Overlaps Nanobench
  //===--------------------------------------------------------------------===//
  // Pattern:
  // 1. Issue all global_loads (N tiles) - stores values and tokens
  // 2. For each tile i: wait on global_load[i] token, issue ds_write[i]
  // 3. For each tile i: wait on ds_write[i] token, issue ds_read[i]
  // 4. Wait on all ds_read tokens and use values (test_inst)
  // 5. Control interleaving with schedule attributes.
  amdgcn.kernel @nanobench_lds_read_write_overlaps arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {shared_memory_size = {{LDS_SIZE}} : i32, block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    // Kernel arguments
    %ptr_s = amdgcn.load_arg 0 : !sx2
    %arg0_raw = lsir.assume_noalias %ptr_s : (!sx2) -> !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Parameters
    %NUM_TILES_I = arith.constant {{NUM_TILES_I}} : index
    %NUM_TILES_J = arith.constant {{NUM_TILES_J}} : index
    %NUM_ITERS = arith.constant {{NUM_ITERS}} : index
    %NUM_TILES = affine.apply affine_map<()[i, j] -> (i * j)>()[%NUM_TILES_I, %NUM_TILES_J]

    // Strides
    %global_stride_bytes = arith.constant {{GLOBAL_STRIDE_BYTES}} : index
    %lds_stride_bytes = arith.constant {{LDS_STRIDE_BYTES}} : index
    %elt_size = arith.constant 2 : index

    // Allocate memrefs for multi-tile operations
    %load_futures_1d = memref.alloca(%NUM_TILES) : memref<?x!future_global_read_any>

    // Create descriptor for future memref (offset=0)
    %load_futures_desc = aster_utils.struct_create(%load_futures_1d, %c0) : (memref<?x!future_global_read_any>, index) -> !future_global_read_descriptor_1d

    // Outer timing loop
    scf.for %iter = %c0 to %NUM_ITERS step %c1 {
      //-----------------------------------------------------------
      // Step 1: Issue ALL global loads (no waits between them)
      //-----------------------------------------------------------
      %tensor_desc = aster_utils.struct_create(%arg0_raw, %c0, %c0, %global_stride_bytes, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
      func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_future(
        %tensor_desc, %NUM_TILES_I, %NUM_TILES_J, %load_futures_desc)
        : (!tensor_position_descriptor_2level_2d, index, index, !future_global_read_descriptor_1d) -> ()

      //-----------------------------------------------------------
      // For each tile, interleave:
      //   - wait on load token, issue ds_write
      //   - wait on write token, issue ds_read
      //   - wait on read token, use value
      // Use sched attributes to control delays and interleaving forms.
      // Note: this is an expressiveness/control/API tradeoff: we could cram
      // everything in multi-tile-copies.mlir but that would limit the ability
      // to control interleaving.
      //-----------------------------------------------------------
      %num_rows = arith.constant 16 : index
      %transfer_size = arith.constant 8 : index
      %wave_size = arith.constant 64 : index
      scf.for %idx = %c0 to %NUM_TILES step %c1 {
        %ti, %tj = affine.delinearize_index %idx into (%NUM_TILES_I, %NUM_TILES_J) : index, index
        %m_pos = affine.apply affine_map<()[ti] -> (ti * 16)>()[%ti]
        %n_pos = affine.apply affine_map<()[tj] -> (tj * 16)>()[%tj]

        // Step 2: Issue ds_write for this tile (single-tile function)
        %lds_write_desc = aster_utils.struct_create(%c0, %m_pos, %n_pos, %lds_stride_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        %transfer_desc = aster_utils.struct_create(%num_rows, %transfer_size, %wave_size) : (index, index, index) -> !transfer_descriptor_2d
        %loaded = func.call @get_global_load_value_vx2_1d(%load_futures_1d, %idx)
          : (memref<?x!future_global_read_any>, index) -> !vx2
        %write_future = func.call @lds_write_wave_256xf16_via_dwordx2_future(
          %lds_write_desc, %transfer_desc, %loaded)
            {sched.delay = 0 : i64, sched.rate = 1 : i64}
          : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> !future_lds_write

        // Step 3: Issue ds_read for this tile (single-tile function)
        func.call @wait_lds_write(%write_future)
            {sched.delay = 8 : i64, sched.rate = 1 : i64}
          : (!future_lds_write) -> ()
        %lds_read_desc = aster_utils.struct_create(%c0, %m_pos, %n_pos, %lds_stride_bytes, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %transposed = arith.constant false
        %read_future = func.call @lds_read_A_wave_16x16xf16_fragment_future(%lds_read_desc, %transposed)
            {sched.delay = 8 : i64, sched.rate = 1 : i64}
          : (!lds_position_descriptor_2d, i1) -> !future_lds_read_any

        // Step 3: Wait on read and use value (delayed to very end of pipeline)
        %read_value = func.call @get_lds_read_value_vx2(%read_future)
            {sched.delay = 100 : i64, sched.rate = 1 : i64}
          : (!future_lds_read_any) -> !vx2
        amdgcn.test_inst ins %read_value : (!vx2) -> ()

      } {aster.constexpr, sched.dims = array<i64: {{NUM_TILES}}>}
    } {aster.constexpr}

    amdgcn.end_kernel
  }
}
