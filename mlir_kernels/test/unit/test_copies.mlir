// Unit test kernels for copies.mlir library functions.
// Each kernel tests a single function by having all threads write their results
// to a global output buffer.

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

amdgcn.module @test_copies target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @alloc_vgprx4() -> !vx4
  func.func private @init_vgprx4(i32) -> !vx4
  func.func private @init_vgprx4_reg(!v) -> !vx4
  // indexing.mlir
  func.func private @lane_id() -> index
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @mfma_index_A_16x16xf16() -> (index, index)
  func.func private @mfma_index_C_16x16xf32() -> (index, index)
  // simple-copies.mlir
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(!sx2, index, index, index, index, index, index, index)
  func.func private @simple_global_store_wave_16x16xf16_wait(!vx2, !sx2, index, index, index)
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(index, index, index, index, !sx2, index, index, index)
  // copies.mlir
  func.func private @global_load_to_lds_wave_16x16_f16_wait(!sx2, index, index, index, index, index, index, index)
  func.func private @global_load_wave_256xf16_via_dwordx2_wait(!sx2, index, index, index, index, index, index) -> (!vx2)
  func.func private @lds_write_wave_256xf16_via_dwordx2_wait(index, index, index, index, index, !vx2) -> ()
  func.func private @store_to_global_dword_wait(!v, !sx2, index, index, index)
  func.func private @lds_read_A_wave_16x16xf16_fragment_wait(index, index, index, index) -> !vx2
  func.func private @lds_read_swizzled_wave_16x16xf16_fragment_wait(index, index, index, index) -> !vx2
  func.func private @global_store_wave_16x16xf32_C_fragment_wait(!vx4, !sx2, index, index, index, index, index)
  func.func private @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(!sx2, index, index, index, index, index, index, index, memref<?x!vx2>)
  func.func private @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(index, index, index, index, index, index, memref<?x!vx2>)
  func.func private @simple_global_load_wave_16x16xf16_wait(!sx2, index, index, index) -> !vx2
  func.func private @simple_lds_write_wave_16x16xf16_wait(!vx2, index, index, index, index)
  func.func private @simple_lds_read_wave_16x16xf16_wait(index, index, index, index) -> !vx2
  // simple-multi-tile-copies.mlir
  func.func private @simple_maybe_lds_write_multi_tile(index, index, index, index, index, index, index, index, index, index, index, memref<?x?x!vx2>)
  // multi-tile-copies.mlir
  func.func private @maybe_global_load_multi_tile_simple(index, index, index, index, index, index, index, index, index, !sx2, index, index, index, memref<?x?x!vx2>)
  func.func private @maybe_global_load_multi_tile_coalesced(index, index, index, index, index, index, index, index, index, !sx2, index, index, index, memref<?x?x!vx2>)
  func.func private @maybe_lds_write_multi_tile_coalesced(index, index, index, index, index, index, index, index, index, index, index, memref<?x?x!vx2>)

  //===--------------------------------------------------------------------===//
  // Helper: store i32 to global at thread index
  //===--------------------------------------------------------------------===//
  func.func private @store_at_tid(%value: i32, %ptr: !sx2, %index_offset: index) {
    %tid = gpu.thread_id x
    %value_vgpr = lsir.to_reg %value : i32 -> !v

    %offset_index = affine.apply affine_map<()[tid, index_offset] -> (tid * 4 + index_offset)>()[%tid, %index_offset]
    %offset = arith.index_cast %offset_index : index to i32
    %offset_vgpr = lsir.to_reg %offset : i32 -> !v

    amdgcn.flat.global_store #amdgcn.inst<global_store_dword> %value_vgpr, %ptr[%offset_vgpr]
      : !v, !sx2[!v]

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return
  }

  //===--------------------------------------------------------------------===//
  // Simple wave-level 16x16xf16 tile reads/writes
  //===--------------------------------------------------------------------===//
  // Test @simple_global_to_lds_wave_16x16xf16_wait: copy a single 16x16 tile from global to LDS
  // Input: 64x96 array, copy tile at position (3,5) = element (48, 80)
  // Verifies position handling by checking only the correct tile is copied
  amdgcn.kernel @test_global_to_lds_and_back_wave_16x16xf16_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 20000 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index   // m_pos = 1 * 16
    %c32 = arith.constant 32 : index   // n_pos = 2 * 16
    %c120 = arith.constant 120 : index // global stride = 60 * 2 bytes
    %c32_2 = arith.constant 32 : index   // LDS stride = 16 * 2 bytes

    // Copy tile at (48, 80) from global to LDS at base 0
    func.call @simple_global_to_lds_wave_16x16xf16_wait(
      %in_ptr, %c16, %c32, %c120,
      %c0, %c16, %c32, %c32_2)
      : (!sx2, index, index, index,
         index, index, index, index) -> ()

    // Copy from LDS to global at position (48, 80)
    func.call @simple_lds_to_global_wave_16x16xf16_wait(
      %c0, %c16, %c32, %c32_2,
      %out_ptr, %c16, %c32, %c120)
      : (index, index, index, index,
         !sx2, index, index, index) -> ()

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  //===--------------------------------------------------------------------===//
  // Global <-> LDS
  //===--------------------------------------------------------------------===//

  // Test @store_to_global_dword_wait: store a dword to global memory
  // Each thread stores (tid * 100) at position (tid/8, tid%8) in a 16-wide matrix
  amdgcn.kernel @test_store_to_global_dword_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %tid = gpu.thread_id x
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes)

    // Compute i, j from tid
    %i = affine.apply affine_map<()[tid] -> (tid floordiv 8)>()[%tid]
    %j = affine.apply affine_map<()[tid] -> (tid mod 8)>()[%tid]

    // Compute value to store: tid * 100
    %tid_i32 = arith.index_cast %tid : index to i32
    %c100 = arith.constant 100 : i32
    %value_i32 = arith.muli %tid_i32, %c100 : i32
    %value = lsir.to_reg %value_i32 : i32 -> !v

    // Store using the library function
    func.call @store_to_global_dword_wait(%value, %out_ptr, %i, %j, %c64)
      : (!v, !sx2, index, index, index) -> ()

    amdgcn.end_kernel
  }

  // Test @lds_read_A_wave_16x16xf16_fragment_wait: read MFMA A fragment from LDS
  // First populate LDS with known data, then read using the MFMA function
  amdgcn.kernel @test_load_and_lds_read_A_wave_16x16xf16_fragment_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 512 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index // stride in bytes (16 elements * 2 bytes for f16)

    // First load data to LDS using load_to_lds
    func.call @global_load_to_lds_wave_16x16_f16_wait(
      %in_ptr, %c0,  // ptr, lds_base_off
      %c0, %c0,      // i_pos, j_pos
      %c32,          // GLOBAL_STRIDE_IN_BYTES
      %c0, %c0,      // ii_pos, jj_pos
      %c32           // LDS_STRIDE_IN_BYTES
    ) : (!sx2, index, index, index, index, index, index, index) -> ()

    // Now read the A fragment using the MFMA read
    // i_pos=0, j_pos=0
    %fragment = func.call @lds_read_A_wave_16x16xf16_fragment_wait(%c0, %c0, %c0, %c32)
      : (index, index, index, index) -> !vx2

    // Store fragment to output (each thread writes 8 bytes)
    %tid = gpu.thread_id x
    %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
    %out_off_i32 = arith.index_cast %out_off : index to i32
    %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx2> %fragment, %out_ptr[%out_off_vgpr]
      : !vx2, !sx2[!v]
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    amdgcn.end_kernel
  }

  // Test @lds_read_swizzled_wave_16x16xf16_fragment_wait with XOR swizzling: read MFMA A fragment from LDS
  // Tests 2x3 tiles of 16x16, each tile contains iota 0-255
  // First populate LDS with known data, then read using the XOR-swizzled MFMA function
  amdgcn.kernel @test_lds_read_swizzled_A_wave_16x16xf16_fragment_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>,
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 4096 : i32} {
    %in_ptr = amdgcn.load_arg 0 : !sx2
    %out_ptr = amdgcn.load_arg 1 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index  // II = 2 tiles in M
    %c3 = arith.constant 3 : index  // JJ = 3 tiles in K

    // Global stride: JJ tiles * 16 elements * 2 bytes = JJ * 32 bytes
    %GLOBAL_STRIDE = affine.apply affine_map<()[j] -> (j * 32)>()[%c3]
    // LDS stride: same as global stride for this test
    %LDS_STRIDE = affine.apply affine_map<()[j] -> (j * 32)>()[%c3]

    // Load all 2x3 tiles from global to LDS
    scf.for %ii = %c0 to %c2 step %c1 {
      scf.for %jj = %c0 to %c3 step %c1 {
        %i_pos = affine.apply affine_map<()[i] -> (i * 16)>()[%ii]
        %j_pos = affine.apply affine_map<()[j] -> (j * 16)>()[%jj]
        func.call @global_load_to_lds_wave_16x16_f16_wait(
          %in_ptr, %c0,           // ptr, lds_base_off
          %i_pos, %j_pos,         // i_pos, j_pos (global position)
          %GLOBAL_STRIDE,         // GLOBAL_STRIDE_IN_BYTES
          %i_pos, %j_pos,         // ii_pos, jj_pos (LDS position)
          %LDS_STRIDE             // LDS_STRIDE_IN_BYTES
        ) : (!sx2, index, index, index, index, index, index, index) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read all tiles using XOR-swizzled read and store to output
    // Output layout: 6 tiles * 64 threads * 4 f16 = 1536 f16 values
    scf.for %ii = %c0 to %c2 step %c1 {
      scf.for %jj = %c0 to %c3 step %c1 {
        %m_pos = affine.apply affine_map<()[i] -> (i * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[j] -> (j * 16)>()[%jj]

        %fragment = func.call @lds_read_swizzled_wave_16x16xf16_fragment_wait(%c0, %m_pos, %n_pos, %LDS_STRIDE)
          : (index, index, index, index) -> !vx2

        // Store fragment to output using simple_global_store_wave_16x16xf16_wait
        // Output buffer is treated as II*16 rows x JJ*16 columns with stride JJ*32 bytes
        func.call @simple_global_store_wave_16x16xf16_wait(%fragment, %out_ptr, %m_pos, %n_pos, %GLOBAL_STRIDE)
          : (!vx2, !sx2, index, index, index) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

  // Test @global_store_wave_16x16xf32_C_fragment_wait: store C fragment to global
  // Initialize accumulators with known values, then store using MFMA function
  amdgcn.kernel @test_store_global_C_fragment_wait arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_write>
  ]> attributes {shared_memory_size = 0 : i32} {
    %out_ptr = amdgcn.load_arg 0 : !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index // stride in bytes (16 elements * 4 bytes for f32)

    // Initialize accumulator with lane_id (as float bits)
    %lane = func.call @lane_id() : () -> index
    %lane_i32 = arith.index_cast %lane : index to i32
    %lane_reg = lsir.to_reg %lane_i32 : i32 -> !v
    %acc = func.call @init_vgprx4_reg(%lane_reg) : (!v) -> !vx4

    // Store using the library function
    // i_pos=0, j_pos=0, ii_pos=0, jj_pos=0
    func.call @global_store_wave_16x16xf32_C_fragment_wait(
      %acc, %out_ptr, %c0, %c0, %c64, %c0, %c0
    ) : (!vx4, !sx2, index, index, index, index, index) -> ()

    amdgcn.end_kernel
  }

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
    %loaded = func.call @global_load_wave_256xf16_via_dwordx2_wait(
      %in_ptr,    // ptr
      %c0, %c0,   // m_pos, n_pos (major tile)
      %c32,       // GLOBAL_STRIDE_IN_BYTES
      %c0, %c0,   // mm_pos, nn_pos (minor tile)
      %c1         // num_rows
    ) : (!sx2, index, index, index, index, index, index) -> (!vx2)

    // DS write from memref to LDS
    func.call @lds_write_wave_256xf16_via_dwordx2_wait(
      %c0,        // lds_base_off
      %c0, %c0,   // mm_pos, nn_pos
      %c32,       // LDS_STRIDE_IN_BYTES
      %c1,        // num_rows
      %loaded     // value
    ) : (index, index, index, index, index, !vx2) -> ()

    // Read back from LDS and store to output
    %tid = gpu.thread_id x
    %lane = func.call @lane_id() : () -> index
    %iii, %jjj = func.call @lane_delinearize_2d(%c16, %c4) : (index, index) -> (index, index)
    %jjj_pos = affine.apply affine_map<()[jjj] -> (jjj * 4)>()[%jjj]

    %lds_off = affine.apply affine_map<()[iii, jjj_pos] -> ((iii * 16 + jjj_pos) * 2)>()[%iii, %jjj_pos]
    %lds_off_i32 = arith.index_cast %lds_off : index to i32
    %lds_off_vgpr = lsir.to_reg %lds_off_i32 : i32 -> !v

    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %c0_i32 = arith.constant 0 : i32
    %from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %dst, %lds_off_vgpr, offset = %c0_i32
      : !v, i32 -> !vx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %out_off = affine.apply affine_map<()[tid] -> (tid * 8)>()[%tid]
    %out_off_i32 = arith.index_cast %out_off : index to i32
    %out_off_vgpr = lsir.to_reg %out_off_i32 : i32 -> !v
    amdgcn.flat.global_store #amdgcn.inst<global_store_dwordx2> %from_lds, %out_ptr[%out_off_vgpr]
      : !vx2, !sx2[!v]
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
        func.call @global_load_wave_multi_tile_256xf16_via_dwordx2_wait(
          %in_ptr,                    // ptr
          %m_pos, %n_pos,             // m_pos, n_pos (major tile position)
          %c256,                      // GLOBAL_STRIDE_IN_BYTES (128 f16 elements * 2 bytes)
          %mm_pos_base, %nn_pos_base, // mm_pos_base, nn_pos_base (minor tile base)
          %c2, %c4,                   // m_tiles=2, n_tiles=4
          %memref                     // result memref
        ) : (!sx2, index, index, index, index, index, index, index, memref<?x!vx2>) -> ()

        // Write all tiles to LDS using multi-tile LDS write (with non-zero base offset)
        func.call @lds_write_wave_multi_tile_256xf16_via_dwordx2_wait(
          %c0,                        // lds_base_off (256 bytes offset)
          %mm_pos_base, %nn_pos_base, // mm_pos_base, nn_pos_base (minor tile base)
          %c256,                      // LDS_STRIDE_IN_BYTES
          %c2, %c4,                   // m_tiles=2, n_tiles=4
          %memref                     // values memref
        ) : (index, index, index, index, index, index, memref<?x!vx2>) -> ()

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
            func.call @simple_lds_to_global_wave_16x16xf16_wait(
              %c0, %lds_m, %lds_n, %c256,
              %out_ptr, %global_m, %global_n, %c256)
              : (index, index, index, index, !sx2, index, index, index) -> ()
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
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        // Call library function for global load (k=0, cond_iter=0 to always execute when aligned)
        func.call @maybe_global_load_multi_tile_simple(
          %c0, %ii, %jj, %c0,           // k, ii, jj, cond_iter
          %K, %II, %JJ,                 // K, II, JJ
          %NT_I, %NT_J,                 // NT_I, NT_J
          %in_ptr, %c0, %c0, %SIZE_J,   // ptr, i_pos_base, j_pos_base, SIZE_J
          %load_memref)                 // load_memref
          : (index, index, index, index, index, index, index, index, index,
             !sx2, index, index, index, memref<?x?x!vx2>) -> ()

        // Call library function for LDS write (k=0, cond_iter=0)
        func.call @simple_maybe_lds_write_multi_tile(
          %c0, %ii, %jj, %c0,           // k, ii, jj, cond_iter
          %K, %II, %JJ,                 // K, II, JJ
          %NT_I, %NT_J,                 // NT_I, NT_J
          %c0, %SIZE_J,                 // lds_base_off, SIZE_J
          %load_memref)                 // load_memref
          : (index, index, index, index, index, index, index, index, index,
             index, index, memref<?x?x!vx2>) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read back all tiles from LDS and write to output
    %STRIDE_IN_BYTES = arith.constant 256 : index // 128 * 2 bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        func.call @simple_lds_to_global_wave_16x16xf16_wait(
          %c0, %m_pos, %n_pos, %STRIDE_IN_BYTES,
          %out_ptr, %m_pos, %n_pos, %STRIDE_IN_BYTES)
          : (index, index, index, index, !sx2, index, index, index) -> ()
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
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        // Call library function for global load (k=0, cond_iter=0 to always execute when aligned)
        func.call @maybe_global_load_multi_tile_coalesced(
          %c0, %ii, %jj, %c0,           // k, ii, jj, cond_iter
          %K, %II, %JJ,                 // K, II, JJ
          %NT_I, %NT_J,                 // NT_I, NT_J
          %in_ptr, %c0, %c0, %SIZE_J,   // ptr, i_pos_base, j_pos_base, SIZE_J
          %load_memref)                 // load_memref
          : (index, index, index, index, index, index, index, index, index,
             !sx2, index, index, index, memref<?x?x!vx2>) -> ()

        // Call library function for LDS write (k=0, cond_iter=0)
        func.call @maybe_lds_write_multi_tile_coalesced(
          %c0, %ii, %jj, %c0,           // k, ii, jj, cond_iter
          %K, %II, %JJ,                 // K, II, JJ
          %NT_I, %NT_J,                 // NT_I, NT_J
          %c0, %SIZE_J,                 // lds_base_off, SIZE_J
          %load_memref)                 // load_memref
          : (index, index, index, index, index, index, index, index, index,
             index, index, memref<?x?x!vx2>) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    // Read back all tiles from LDS and write to output
    %STRIDE_IN_BYTES = arith.constant 256 : index // 128 * 2 bytes
    scf.for %ii = %c0 to %II step %c1 {
      scf.for %jj = %c0 to %JJ step %c1 {
        %m_pos = affine.apply affine_map<()[ii] -> (ii * 16)>()[%ii]
        %n_pos = affine.apply affine_map<()[jj] -> (jj * 16)>()[%jj]

        func.call @simple_lds_to_global_wave_16x16xf16_wait(
          %c0, %m_pos, %n_pos, %STRIDE_IN_BYTES,
          %out_ptr, %m_pos, %n_pos, %STRIDE_IN_BYTES)
          : (index, index, index, index, !sx2, index, index, index) -> ()
      } {aster.constexpr}
    } {aster.constexpr}

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    amdgcn.end_kernel
  }

}
