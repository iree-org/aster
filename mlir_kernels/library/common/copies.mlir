// Common copy functions for AMDGCN kernels.

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register_init.mlir,%p/library/common/indexing.mlir" \
// RUN: | FileCheck %s

!s   = !amdgcn.sgpr
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

!v   = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

!a   = !amdgcn.agpr
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax4 = !amdgcn.agpr_range<[? + 4]>

amdgcn.library @common_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register_init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4(i32) -> !vx4
  // indexing.mlir
  func.func private @lane_delinearize_2d(index, index) -> (index, index)
  func.func private @matrix_offset(index, index, index, index) -> !v
  func.func private @tiled_matrix_offset(index, index, index, index, index, index) -> !v
  func.func private @tiledx2_matrix_offset(index, index, index, index, index, index, index, index) -> !v
  func.func private @swizzle_A_16x16xf16() -> (index, index)
  func.func private @swizzle_C_16x16xf32() -> (index, index)

  //===--------------------------------------------------------------------===//
  // Global <-> LDS
  //===--------------------------------------------------------------------===//
  // Load a 16x16xdword2 tile from global memory to LDS within a single wave, in
  // a **synchronized fashion** (i.e. waitcnt 0 are inserted after global_load
  // and after lds_write).
  // The implementation assumes access to be performed in a tiled fashion, where
  // there is a major tile and a minor tile within it. The caller is
  // responsible for embedding distribution information into the indices.
  // The callee is responsible for computing the offsets within the tiles based on
  // the lane id.

  func.func private @load_to_lds_16x16_dwordx2_wait(
    %ptr: !sx2,           // The global base pointer
    %lds_base_off: index, // The local base offset in LDS
    %i_pos: index,        // The outer-most major-tile position
    %j_pos: index,        // The inner-most major-tile position
    %N_SIZE: index,       // The inner-most size
    %ii_pos: index,       // The outer-most minor-tile position
    %jj_pos: index,       // The inner-most minor-tile position
    %NN_SIZE: index       // The inner-most major-tile size
  ) {
    // Constants
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %elt_size = arith.constant 2 : index // f16 size in bytes

    // Get local positions within the minor tile
    %iii, %jjj = func.call @lane_delinearize_2d(%c16, %c4)
      : (index, index) -> (index, index)
    %jjj_pos = affine.apply affine_map<()[jjj] -> (jjj * 4)>()[%jjj]

    // Calculate global offset
    %off_reg = func.call @tiledx2_matrix_offset(
      %i_pos, %j_pos, %ii_pos, %jj_pos, %iii, %jjj_pos, %N_SIZE, %elt_size)
      : (index, index, index, index, index, index, index, index) -> !v

    // Perform the load
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %loaded = amdgcn.flat.global_load <global_load_dwordx2> %dst, %ptr[%off_reg]
      : !vx2, !sx2[!v] -> !vx2

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    // Calculate offset into LDS
    %off_lds_reg = func.call @tiled_matrix_offset(%ii_pos, %jj_pos, %iii, %jjj_pos, %NN_SIZE, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // DS write to LDS
    %l_off_i32 = arith.index_cast %lds_base_off : index to i32
    amdgcn.ds.write #amdgcn.inst<ds_write_b64> %loaded, %off_lds_reg, offset = %l_off_i32
      : !vx2, !v, i32

    // Wait for LDS write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return
  }

  // Loads from global memory to VGPRs, in a **synchronized fashion** (i.e.
  // waitcnt 0 are inserted after global_load).
  // This function cooperatively loads 256 f16 elements (64 dwordx2), depending on %NN:
  //   - when %NN =  1, a 16x16xf16 tile is loaded from global memory to VGPRs
  //   - when %NN =  2, a  8x32xf16 tile is loaded from global memory to VGPRs
  //   - when %NN =  4, a  4x64xf16 tile is loaded from global memory to VGPRs
  //   - when %NN =  8, a 2x128xf16 tile is loaded from global memory to VGPRs
  //   - when %NN = 16, a 1x256xf16 tile is loaded from global memory to VGPRs
  // This results in better global memory coalescing when %NN is not 1.
  func.func private @global_load_dwordx2_wait(
    %ptr: !sx2,           // The global base pointer
    %i_pos: index,        // The outer-most major-tile position
    %j_pos: index,        // The inner-most major-tile position
    %N_SIZE: index,       // The inner-most size
    %ii_pos: index,       // The outer-most minor-tile position
    %jj_pos: index,       // The inner-most minor-tile position
    %NN: index            // The number of 16 tiles in the inner-most major-tile
  ) -> !vx2 {
    // Constants
    %elt_size = arith.constant 2 : index // f16 size in bytes

    %SZ0 = affine.apply affine_map<()[NN] -> (16 ceildiv NN)>()[%NN]
    %SZ1 = affine.apply affine_map<()[NN] -> (4 * NN)>()[%NN]

    // Get local positions within the minor tile
    %iii, %jjj = func.call @lane_delinearize_2d(%SZ0, %SZ1)
      : (index, index) -> (index, index)
    %jjj_pos = affine.apply affine_map<()[jjj] -> (4 * jjj)>()[%jjj]

    // Calculate global offset
    %off_reg = func.call @tiledx2_matrix_offset(
      %i_pos, %j_pos, %ii_pos, %jj_pos, %iii, %jjj_pos, %N_SIZE, %elt_size)
      : (index, index, index, index, index, index, index, index) -> !v

    // Perform the load
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %loaded = amdgcn.flat.global_load <global_load_dwordx2> %dst, %ptr[%off_reg]
      : !vx2, !sx2[!v] -> !vx2

    // Wait for load completion
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0

    return %loaded : !vx2
  }

  // Write %value to LDS.
  // This function cooperatively writes 256 f16 elements (64 dwordx2), depending on %NN:
  //   - when %NN =  1, a 16x16xf16 tile is written from VGPRs to LDS
  //   - when %NN =  2, a  8x32xf16 tile is written from VGPRs to LDS
  //   - when %NN =  4, a  4x64xf16 tile is written from VGPRs to LDS
  //   - when %NN =  8, a 2x128xf16 tile is written from VGPRs to LDS
  //   - when %NN = 16, a 1x256xf16 tile is written from VGPRs to LDS
  // This results in better global memory coalescing when %NN is not 1.
  func.func private @lds_write_dwordx2_wait(
    %lds_base_off: index, // The local base offset in LDS
    %ii_pos: index,       // The outer-most minor-tile position
    %jj_pos: index,       // The inner-most minor-tile position
    %NN_SIZE: index,      // The inner-most major-tile size
    %NN: index,           // The number of 16 tiles in the inner-most major-tile
    %value: !vx2          // The value to write to LDS
  ) {
    // Constants
    %elt_size = arith.constant 2 : index // f16 size in bytes

    %SZ0 = affine.apply affine_map<()[NN] -> (16 ceildiv NN)>()[%NN]
    %SZ1 = affine.apply affine_map<()[NN] -> (4 * NN)>()[%NN]

    // Get local positions within the minor tile
    %iii, %jjj = func.call @lane_delinearize_2d(%SZ0, %SZ1)
      : (index, index) -> (index, index)
    %jjj_pos = affine.apply affine_map<()[jjj] -> (4 * jjj)>()[%jjj]

    // Calculate offset into LDS
    %off_lds_reg = func.call @tiled_matrix_offset(%ii_pos, %jj_pos, %iii, %jjj_pos, %NN_SIZE, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // DS write to LDS
    %l_off_i32 = arith.index_cast %lds_base_off : index to i32
    amdgcn.ds.write #amdgcn.inst<ds_write_b64> %value, %off_lds_reg, offset = %l_off_i32
      : !vx2, !v, i32

    // Wait for LDS write
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    return
  }

  // Store a dword to global memory, in a **synchronized fashion** (i.e.
  // waitcnt 0 are inserted after global_store).
  // The caller is responsible for embedding distribution information into the
  // positions %i and %j (and make them workgroup/wave/thread/lane-dependent).
  func.func private @store_to_global_dword_wait(
    %value: !v,     // Value to store
    %ptr: !sx2,     // The global base pointer
    %i: index,      // The outer-most position
    %j: index,      // The inner-most position
    %N_SIZE: index  // The inner-most size (stride)
  ) {
    %elt_size = arith.constant 4 : index // dword size in bytes
    %off_reg = func.call @matrix_offset(%i, %j, %N_SIZE, %elt_size)
      : (index, index, index, index) -> !v
    amdgcn.flat.global_store <global_store_dword> %value, %ptr[%off_reg] : !v, !sx2[!v]
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  //===--------------------------------------------------------------------===//
  // Swizzled fragment reads/writes
  //===--------------------------------------------------------------------===//
  // Read the `A` fragment (16x16xf16) from LDS to VGPRs, in a **synchronized
  // fashion** (i.e. waitcnt 0 is inserted after the ds_read).
  // The caller is responsible for embedding distribution information into the
  // positions %i_pos and %j_pos.
  func.func private @read_lds_A_16x16xf16_fragment_wait(
    %lds_base: index, // The local base offset in LDS
    %i_pos: index,    // The outer-most tile position
    %j_pos: index,    // The inner-most tile position
    %N_SIZE: index    // The inner-most size
  ) -> !vx2 {
    // Compute the swizzled offset
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %ii, %jj = func.call @swizzle_A_16x16xf16() : () -> (index, index)
    %off_lds_reg = func.call @tiled_matrix_offset(%i_pos, %j_pos, %ii, %jj, %N_SIZE, %elt_size)
      : (index, index, index, index, index, index) -> !v

    // Perform the DS read
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds = amdgcn.ds.read #amdgcn.inst<ds_read_b64> %dst, %off_lds_reg, offset = %lds_base_i32
      : !v, i32 -> !vx2

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %from_lds : !vx2
  }

  // Store the `C` fragment (16x16xf32) from VGPRs to global memory, in a
  // **synchronized fashion** (i.e. waitcnt 0 is inserted after each global_store).
  // The caller is responsible for embedding distribution information into the
  // positions. The callee computes and embeds the swizzled positions.
  // This function assumes a major/minor tile structure for the global positions.
  func.func private @store_global_16x16xf32_C_fragment_wait(
    %acc: !vx4,           // The accumulator fragment to store
    %ptr: !sx2,           // The global base pointer
    %i_pos: index,        // The outer-most major-tile position
    %j_pos: index,        // The inner-most major-tile position
    %N_SIZE: index,       // The inner-most size
    %ii_pos: index,       // The outer-most minor-tile position
    %jj_pos: index        // The inner-most minor-tile position
  ) {
    // Constants
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Split the fragment into 4 dword values
    %v0, %v1, %v2, %v3 = amdgcn.split_register_range %acc : !vx4
    %C_fragment =  memref.alloca() : memref<4x!v>
    memref.store %v0, %C_fragment[%c0] : memref<4x!v>
    memref.store %v1, %C_fragment[%c1] : memref<4x!v>
    memref.store %v2, %C_fragment[%c2] : memref<4x!v>
    memref.store %v3, %C_fragment[%c3] : memref<4x!v>

    // Compute the swizzled offset
    %iii, %jjj = func.call @swizzle_C_16x16xf32() : () -> (index, index)

    // Calculate global j position
    %j_global_pos = affine.apply
      affine_map<()[j_pos, jj_pos, jjj] -> (j_pos + jj_pos + jjj)>
      ()[%j_pos, %jj_pos, %jjj]

    // Store each fragment to global memory
    scf.for %iiii = %c0 to %c4 step %c1 {
      %fragment = memref.load %C_fragment[%iiii] : memref<4x!v>
      // Calculate global i position
      %i_global_pos = affine.apply
        affine_map<()[i_pos, ii_pos, iii, iiii] -> (i_pos + ii_pos + iii + iiii)>
        ()[%i_pos, %ii_pos, %iii, %iiii]

      // Store to global memory with wait
      func.call @store_to_global_dword_wait(%fragment, %ptr, %i_global_pos, %j_global_pos, %N_SIZE)
        : (!v, !sx2, index, index, index) -> ()
    } {amdgcn.constexpr}
    return
  }
}
