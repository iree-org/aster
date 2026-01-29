// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

// From descriptors.mlir
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>
!index_pair = !aster_utils.struct<i: index, j: index>
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - k: outer loop index (for indexing load_memref -> mem2reg)
//   - cond_iter: condition index (execute only when cond_iter == 0)
//   - NT_I, NT_J: multi-tile factors (process NT_I x NT_J tiles at once)
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // From register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4_reg(!v) -> !vx4
  // From indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  // From simple-copies.mlir
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(!tensor_position_descriptor_2d, !lds_position_descriptor_2d)
  func.func private @simple_global_store_wave_16x16xf16_wait(!vx2, !tensor_position_descriptor_2d)
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(!lds_position_descriptor_2d, !tensor_position_descriptor_2d)
  // From copies.mlir
  func.func private @global_load_to_lds_wave_16x16_f16_wait(!tensor_position_descriptor_2level_2d, index, index)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(!lds_position_descriptor_2d, i1) -> !vx2
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(!lds_position_descriptor_2d) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(!vx4, !tensor_position_descriptor_2level_2d, i1)
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>)
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>)
  // From simple-multi-tile-copies.mlir
  func.func private @simple_maybe_lds_write_multi_tile(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)
  // From multi-tile-copies.mlir
  func.func private @simple_maybe_global_load_multi_tile(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2d, memref<?x?x!vx2>)
  func.func private @maybe_global_load_multi_tile_coalesced(!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>)

  //===--------------------------------------------------------------------===//
  // Helper: store i32 to global at thread index
  //===--------------------------------------------------------------------===//
  func.func private @store_at_tid(%value: i32, %ptr: !sx2, %index_offset: index) {
    %tid = gpu.thread_id x
    %value_vgpr = lsir.to_reg %value : i32 -> !v

    %offset_index = affine.apply affine_map<()[tid, index_offset] -> (tid * 4 + index_offset)>()[%tid, %index_offset]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !v

    %c0 = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dword data %value_vgpr addr %ptr offset d(%offset_vgpr) + c(%c0) : ins(!v, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return
  }

  //===--------------------------------------------------------------------===//
  // Global <-> LDS
  //===--------------------------------------------------------------------===//

  // Test @global_load_wave_256xf16_via_dwordx2_wait + @lds_write_wave_256xf16_via_dwordx2_wait: decoupled global load and LDS write
  // Load from global to memref, then write from memref to LDS, then read back from LDS
  amdgcn.kernel @test_global_load_ds_write arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 512 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index // stride in bytes (16 elements * 2 bytes for f16)

    // Allocate memref for intermediate storage and cast to dynamic
    %memref_static = memref.alloca() : memref<1x1x!vx2>
    %memref = memref.cast %memref_static : memref<1x1x!vx2> to memref<?x?x!vx2>

    // Global load to memref, we know we are using 2B elements.
    %elt_size = arith.constant 2 : index
    %pos_desc = aster_utils.struct_create(%in_ptr, %c0, %c0, %c32, %c0, %c0, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
    %transfer_size_load = arith.constant 8 : index // dwordx2
    %wave_size_load = arith.constant 64 : index
    %transfer_desc_load = aster_utils.struct_create(%c16, %transfer_size_load, %wave_size_load) : (index, index, index) -> !transfer_descriptor_2d
    %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(%pos_desc, %transfer_desc_load) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> (!vx2)

    // DS write from memref to LDS
    %lds_pos_desc_write = aster_utils.struct_create(%c0, %c0, %c0, %c32, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
    %transfer_desc_write = aster_utils.struct_create(%c1, %transfer_size_load, %wave_size_load) : (index, index, index) -> !transfer_descriptor_2d
    func.call @lds_write_wave_256xf16_via_dwordx2_wait(%lds_pos_desc_write, %transfer_desc_write, %loaded)
      : (!lds_position_descriptor_2level_2d, !transfer_descriptor_2d, !vx2) -> ()

    // Read back from LDS and store to output
    %tid = gpu.thread_id x
    %lane = func.call @lane_id() : () -> index
    %dims = aster_utils.struct_create(%c16, %c4) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %iii, %jjj = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    %jjj_pos = affine.apply affine_map<()[jjj] -> (jjj * 4)>()[%jjj]

    %lds_off = affine.apply affine_map<()[iii, jjj_pos] -> ((iii * 16 + jjj_pos) * 2)>()[%iii, %jjj_pos]
    %lds_off_i32 = arith.index_cast %lds_off : index to i32
    %lds_off_vgpr = lsir.to_reg %lds_off_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %c0_i32 = arith.constant 0 : i32
    %from_lds, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %lds_off_vgpr offset c(%c0_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
    %out_off_i32 = arith.index_cast %out_off : index to i32
    %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
    %tok_store = amdgcn.store global_store_dwordx2 data %from_lds addr %out_ptr offset d(%out_off_vgpr) + c(%c0_i32) : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

  // Test @global_load_wave_multi_tile_256xf16_via_dwordx2_wait and @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait
  // Use a larger 64x128 input array and load 2x4 tiles at 4 different offsets (2x2 loop)
  // This tests that m_off and n_off are computed correctly for non-zero positions
  // LDS base offset is non-zero (256 bytes) to test offset handling
  amdgcn.kernel @test_global_load_multi_tile arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 4352 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index   // output stride = 16 * 2 bytes
    %c256 = arith.constant 256 : index // global/LDS stride = 128 * 2 bytes
    %elt_size = arith.constant 2 : index // f16 size in bytes

    // Allocate memref for multi-tile results and cast to dynamic
    %memref_static = memref.alloca() : memref<8x!vx2>
    %memref = memref.cast %memref_static : memref<8x!vx2> to memref<?x!vx2>

    // 2x2 loop over different starting positions in the 64x128 input
    // Position (0,0): m_pos=0, n_pos=0
    // Position (0,1): m_pos=0, n_pos=64
    // Position (1,0): m_pos=32, n_pos=0
    // Position (1,1): m_pos=32, n_pos=64
    scf.for %pm = %c0 to %c2 step %c1 {
      scf.for %pn = %c0 to %c2 step %c1 {
        // Calculate m_pos and n_pos for this iteration
        %m_pos = affine.apply affine_map<()[pm] -> (pm * 32)>()[%pm]
        %n_pos = affine.apply affine_map<()[pn] -> (pn * 64)>()[%pn]

        // Compute minor-tile base positions for this iteration
        // These offset the starting position within each major tile region
        %mm_pos_base = affine.apply affine_map<()[pm] -> (pm * 2)>()[%pm]
        %nn_pos_base = affine.apply affine_map<()[pn] -> (pn * 4)>()[%pn]

        // Multi-tile global load: 2 tiles in M, 4 tiles in N (32x64 region)
        // Create 2-level descriptor: m_pos/n_pos=major tile pos, mm_pos/nn_pos=minor tile base
        %global_load_desc = aster_utils.struct_create(%in_ptr, %m_pos, %n_pos, %c256, %mm_pos_base, %nn_pos_base, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
          %global_load_desc,          // tensor_position_descriptor_2level_2d
          %c2, %c4,                   // m_tiles=2, n_tiles=4
          %memref                     // result memref
        ) : (!tensor_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()

        // Write all tiles to LDS using multi-tile LDS write (with non-zero base offset)
        // Create 2-level LDS descriptor: lds_base=0, mm_pos/nn_pos=minor tile base
        %lds_write_desc = aster_utils.struct_create(%c0, %mm_pos_base, %nn_pos_base, %c256, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2level_2d
        func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
          %lds_write_desc,            // lds_position_descriptor_2level_2d
          %c2, %c4,                   // m_tiles=2, n_tiles=4
          %memref                     // values memref
        ) : (!lds_position_descriptor_2level_2d, index, index, memref<?x!vx2>) -> ()

        // Read back from LDS and store to output for each tile using lds_to_global
        scf.for %mt = %c0 to %c2 step %c1 {
          scf.for %nt = %c0 to %c4 step %c1 {
            // LDS tile position = mm_pos_base + mt*16, nn_pos_base + nt*16
            %lds_m = affine.apply affine_map<()[base, mt] -> (base + mt * 16)>()[%mm_pos_base, %mt]
            %lds_n = affine.apply affine_map<()[base, nt] -> (base + nt * 16)>()[%nn_pos_base, %nt]

            // Global tile position = m_pos + mt*16, n_pos + nt*16
            %global_m = affine.apply affine_map<()[base, mt] -> (base + mt * 16)>()[%m_pos, %mt]
            %global_n = affine.apply affine_map<()[base, nt] -> (base + nt * 16)>()[%n_pos, %nt]

            // Copy from LDS to global using simple function
            %lds_pos_out = aster_utils.struct_create(%c0, %lds_m, %lds_n, %c256, %elt_size) : (index, index, index, index, index) -> !lds_position_descriptor_2d
            %global_pos_out = aster_utils.struct_create(%out_ptr, %global_m, %global_n, %c256, %elt_size) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
            func.call @simple_lds_to_global_wave_16x16xf16_wait(%lds_pos_out, %global_pos_out)
              : (!lds_position_descriptor_2d, !tensor_position_descriptor_2d) -> ()
          } {aster.constexpr}
        } {aster.constexpr}
      } {aster.constexpr}
    } {aster.constexpr}

    // Ensure all stores complete before kernel ends
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

  //===--------------------------------------------------------------------===//
  // Test maybe_*_multi_tile_simple pattern from GEMM
  // This tests the library functions from multi_tile_copies.mlir
  //===--------------------------------------------------------------------===//
  // Pattern: Loop over (ii, jj) indices, execute multi-tile load/write when
  // ii % NT_I == 0 AND jj % NT_J == 0
  // Input: 64x64 array (4x4 tiles of 16x16)
  // Test with NT_I=2, NT_J=2 to exercise multiple batches
  amdgcn.kernel @test_maybe_multi_tile_simple arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 16384 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %SIZE_J = arith.constant 128 : index

    // Parameters: 4x8 tiles, load 2x4 tiles at a time
    %K = arith.constant 1 : index    // Outer loop size (single iteration for simplicity)
    %II = arith.constant 4 : index   // Total tiles in I dimension
    %JJ = arith.constant 8 : index   // Total tiles in J dimension
    %NT_I = arith.constant 2 : index // Multi-tile factor I
    %NT_J = arith.constant 4 : index // Multi-tile factor J

    // Allocate 2D memref for library functions: [K, NT_I*NT_J]
    %load_memref_static = memref.alloca() : memref<1x8x!vx2>
    %load_memref = memref.cast %load_memref_static : memref<1x8x!vx2> to memref<?x?x!vx2>

    // Loop over all tile indices like in GEMM (single k iteration)
    %global_stride_bytes = arith.constant 256 : index // SIZE_J * 2 = 128 * 2 bytes
    %elt_size_global = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        // Create conditional execution descriptor (k=0, cond_iter=0 to always execute when aligned)
        %cond_desc = aster_utils.struct_create(%c0, %c0, %NT_I, %NT_J) : (index, index, index, index) -> !conditional_execution_descriptor_2d

        // Call library function for global load
        // tensor_desc: m_pos=ii, n_pos=jj are tile indices
        %tensor_desc = aster_utils.struct_create(%in_ptr, %ii, %jj, %global_stride_bytes, %elt_size_global) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_maybe_global_load_multi_tile(
          %cond_desc,                   // conditional_execution_descriptor_2d
          %tensor_desc,                 // tensor_position_descriptor_2d
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !tensor_position_descriptor_2d, memref<?x?x!vx2>) -> ()

        // Call library function for LDS write
        %lds_stride_mt = arith.constant 256 : index // 128 * 2 bytes
        %elt_size_mt = arith.constant 2 : index
        %lds_pos_desc_mt = aster_utils.struct_create(%c0, %ii, %jj, %lds_stride_mt, %elt_size_mt) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        func.call @simple_maybe_lds_write_multi_tile(
          %cond_desc,                   // conditional_execution_descriptor_2d
          %lds_pos_desc_mt,             // lds_pos_desc_base (m_pos=ii, n_pos=jj as tile indices)
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read back all tiles from LDS and write to output
    %STRIDE_IN_BYTES = arith.constant 256 : index // 128 * 2 bytes
    %elt_size_simple = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        %lds_pos_read = aster_utils.struct_create(%c0, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_simple) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %global_pos_write = aster_utils.struct_create(%out_ptr, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_simple) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_lds_to_global_wave_16x16xf16_wait(%lds_pos_read, %global_pos_write)
          : (!lds_position_descriptor_2d, !tensor_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  //===--------------------------------------------------------------------===//
  // Test maybe_*_multi_tile_coalesced pattern from GEMM (bulk version)
  // This tests the bulk multi-tile library functions from multi_tile_copies.mlir
  //===--------------------------------------------------------------------===//
  // Pattern: Loop over (ii, jj) indices, execute multi-tile load/write when
  // ii % NT_I == 0 AND jj % NT_J == 0
  // Input: 32x64 array (2x4 tiles of 16x16)
  // Test with NT_I=2, NT_J=2 to exercise multiple batches
  amdgcn.kernel @test_maybe_multi_tile_coalesced arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 16384 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %SIZE_J = arith.constant 128 : index

    // Parameters: 4x8 tiles, load 2x4 tiles at a time
    %K = arith.constant 1 : index    // Outer loop size (single iteration for simplicity)
    %II = arith.constant 4 : index   // Total tiles in I dimension
    %JJ = arith.constant 8 : index   // Total tiles in J dimension
    %NT_I = arith.constant 2 : index // Multi-tile factor I
    %NT_J = arith.constant 4 : index // Multi-tile factor J

    // Allocate 2D memref for library functions: [K, NT_I*NT_J]
    %load_memref_static = memref.alloca() : memref<1x8x!vx2>
    %load_memref = memref.cast %load_memref_static : memref<1x8x!vx2> to memref<?x?x!vx2>

    // Loop over all tile indices like in GEMM (single k iteration)
    %global_stride_bytes_coal = arith.constant 256 : index // SIZE_J * 2 = 128 * 2 bytes
    %elt_size_global_coal = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        // Create conditional execution descriptor (k=0, cond_iter=0 to always execute when aligned)
        %cond_desc_coal = aster_utils.struct_create(%c0, %c0, %NT_I, %NT_J) : (index, index, index, index) -> !conditional_execution_descriptor_2d

        // Call library function for global load
        // 2-level descriptor: m_pos/n_pos=0 (base positions), mm_pos/nn_pos=ii/jj (tile indices)
        %tensor_desc_coal = aster_utils.struct_create(%in_ptr, %c0, %c0, %global_stride_bytes_coal, %ii, %jj, %elt_size_global_coal) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
        func.call @maybe_global_load_multi_tile_coalesced(
          %cond_desc_coal,              // conditional_execution_descriptor_2d
          %tensor_desc_coal,            // tensor_position_descriptor_2level_2d
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !tensor_position_descriptor_2level_2d, memref<?x?x!vx2>) -> ()

        // Call library function for LDS write
        // LDS descriptor: lds_base=0, m_pos=ii, n_pos=jj (tile indices)
        %lds_stride_bytes_coal = arith.constant 256 : index // SIZE_J * 2 = 128 * 2 bytes
        %elt_size_lds_coal = arith.constant 2 : index
        %lds_desc_coal = aster_utils.struct_create(%c0, %ii, %jj, %lds_stride_bytes_coal, %elt_size_lds_coal) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        func.call @maybe_lds_write_multi_tile_coalesced(
          %cond_desc_coal,              // conditional_execution_descriptor_2d
          %lds_desc_coal,               // lds_position_descriptor_2d
          %load_memref)                 // load_memref
          : (!conditional_execution_descriptor_2d, !lds_position_descriptor_2d, memref<?x?x!vx2>) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read back all tiles from LDS and write to output
    %STRIDE_IN_BYTES = arith.constant 256 : index // 128 * 2 bytes
    %elt_size_coal = arith.constant 2 : index // f16 size in bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        %lds_pos_coal = aster_utils.struct_create(%c0, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_coal) : (index, index, index, index, index) -> !lds_position_descriptor_2d
        %global_pos_coal = aster_utils.struct_create(%out_ptr, %m_pos, %n_pos, %STRIDE_IN_BYTES, %elt_size_coal) : (!sx2, index, index, index, index) -> !tensor_position_descriptor_2d
        func.call @simple_lds_to_global_wave_16x16xf16_wait(%lds_pos_coal, %global_pos_coal)
          : (!lds_position_descriptor_2d, !tensor_position_descriptor_2d) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

}
