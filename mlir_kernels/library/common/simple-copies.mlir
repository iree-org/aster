// Common simple copy functions for AMDGCN kernels (mostly for testing purposes).

// Drive this through pytest, only check input IR validity here.
// RUN: cat %s \
// RUN: | aster-opt --amdgcn-preload-library="library-paths=%p/library/common/register-init.mlir,%p/library/common/indexing.mlir" \
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

!index_pair = !aster_utils.struct<i: index, j: index>
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>
!index_descriptor_3level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, iii: index, jjj: index, stride: index, elt_size_b: index>

amdgcn.library @common_copies isa = [#amdgcn.isa<cdna3>] {
  //===--------------------------------------------------------------------===//
  // Library function declarations (provided by amdgcn-preload-library pass)
  //===--------------------------------------------------------------------===//
  // register-init.mlir
  func.func private @alloc_vgprx2() -> !vx2
  func.func private @init_vgprx4(i32) -> !vx4
  // indexing.mlir
  func.func private @lane_delinearize_2d(!index_pair) -> !index_pair
  func.func private @matrix_offset(!index_descriptor_2d) -> !v
  func.func private @tiled_matrix_offset(!index_descriptor_2level_2d) -> !v
  func.func private @tiledx2_matrix_offset(!index_descriptor_3level_2d) -> !v
  func.func private @mfma_index_A_16x16xf16() -> !index_pair
  func.func private @mfma_index_C_16x16xf32() -> !index_pair


  //===--------------------------------------------------------------------===//
  // Simple wave-level 16x16xf16 tile reads/writes
  //===--------------------------------------------------------------------===//
  // Read a 16x16xf16 tile from global memory to VGPRs, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the global_load).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @simple_global_load_wave_16x16xf16_wait(
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most tile position
    %n_pos: index,                  // The inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index  // The inner-most stride **in bytes** in global memory
  ) -> !vx2 {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the global load
    %c0 = arith.constant 0 : i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_global, %tok_load = amdgcn.load global_load_dwordx2 dest %dst addr %ptr offset d(%off_reg) + c(%c0) : dps(!vx2) ins(!sx2, !v, i32) -> !amdgcn.read_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return %from_global : !vx2
  }

  // Write a 16x16xf16 tile from VGPRs to global memory, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the global_store).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @simple_global_store_wave_16x16xf16_wait(
    %value: !vx2,                   // The value to write to global memory
    %ptr: !sx2,                     // The global base pointer
    %m_pos: index,                  // The outer-most tile position
    %n_pos: index,                  // The inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index  // The inner-most stride **in bytes** in global memory
  ) {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %GLOBAL_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the global store
    %c0 = arith.constant 0 : i32
    %tok_store = amdgcn.store global_store_dwordx2 data %value addr %ptr offset d(%off_reg) + c(%c0) : ins(!vx2, !sx2, !v, i32) -> !amdgcn.write_token<flat>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
    return
  }

  // Read a 16x16xf16 tile from LDS to VGPRs, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the ds_read).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @simple_lds_read_wave_16x16xf16_wait(
    %lds_base: index,           // The local base offset in LDS
    %m_pos: index,              // The outer-most tile position
    %n_pos: index,              // The inner-most tile position
    %LDS_STRIDE_IN_BYTES: index // The inner-most stride **in bytes** in LDS
  ) -> !vx2 {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the DS read
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %dst = func.call @alloc_vgprx2() : () -> (!vx2)
    %from_lds, %tok_read = amdgcn.load ds_read_b64 dest %dst addr %off_lds_reg offset c(%lds_base_i32) : dps(!vx2) ins(!v, i32) -> !amdgcn.read_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return %from_lds : !vx2
  }

  // Write a 16x16xf16 tile from VGPRs to LDS, in a **synchronized fashion**
  // (i.e. waitcnt 0 is inserted after the ds_write).
  // The caller is responsible for embedding distribution information into the
  // positions %m_pos and %n_pos.
  func.func private @simple_lds_write_wave_16x16xf16_wait(
    %value: !vx2,               // The value to write to LDS
    %lds_base: index,           // The local base offset in LDS
    %m_pos: index,              // The outer-most tile position
    %n_pos: index,              // The inner-most tile position
    %LDS_STRIDE_IN_BYTES: index // The inner-most stride **in bytes** in LDS
  ) {
    %num_rows = arith.constant 16 : index
    %num_cols = arith.constant 4 : index
    %dims = aster_utils.struct_create(%num_rows, %num_cols) : (index, index) -> !index_pair
    %result = func.call @lane_delinearize_2d(%dims) : (!index_pair) -> !index_pair
    %mm_pos, %nn = aster_utils.struct_extract %result ["i", "j"] : !index_pair -> index, index
    // Scale nn by 4 since each thread handles 4 elements (dwordx2 = 8 bytes / 2 bytes per f16)
    %nn_pos = affine.apply affine_map<()[nn] -> (nn * 4)>()[%nn]
    %elt_size = arith.constant 2 : index // f16 size in bytes
    %desc = aster_utils.struct_create(%m_pos, %n_pos, %mm_pos, %nn_pos, %LDS_STRIDE_IN_BYTES, %elt_size) : (index, index, index, index, index, index) -> !index_descriptor_2level_2d
    %off_lds_reg = func.call @tiled_matrix_offset(%desc) : (!index_descriptor_2level_2d) -> !v

    // Perform the DS write
    %lds_base_i32 = arith.index_cast %lds_base : index to i32
    %tok_write = amdgcn.store ds_write_b64 data %value addr %off_lds_reg offset c(%lds_base_i32) : ins(!vx2, !v, i32) -> !amdgcn.write_token<shared>

    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0
    return
  }

  // Simple variant: load a 16x16xf16 tile from global to LDS.
  func.func private @simple_global_to_lds_wave_16x16xf16_wait(
    %ptr: !sx2,                     // The global base pointer
    %m_pos_global: index,           // The global outer-most tile position
    %n_pos_global: index,           // The global inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index, // Global stride in bytes
    %lds_base: index,               // The local base offset in LDS
    %m_pos_lds: index,              // The LDS outer-most tile position
    %n_pos_lds: index,              // The LDS inner-most tile position
    %LDS_STRIDE_IN_BYTES: index     // LDS stride in bytes
  ) {
    %loaded = func.call @simple_global_load_wave_16x16xf16_wait(
        %ptr, %m_pos_global, %n_pos_global, %GLOBAL_STRIDE_IN_BYTES)
      : (!sx2, index, index, index) -> !vx2
    func.call @simple_lds_write_wave_16x16xf16_wait(
        %loaded, %lds_base, %m_pos_lds, %n_pos_lds, %LDS_STRIDE_IN_BYTES)
      : (!vx2, index, index, index, index) -> ()
    return
  }

  // Simple variant: load a 16x16xf16 tile from LDS to global.
  func.func private @simple_lds_to_global_wave_16x16xf16_wait(
    %lds_base: index,               // The local base offset in LDS
    %m_pos_lds: index,              // The LDS outer-most tile position
    %n_pos_lds: index,              // The LDS inner-most tile position
    %LDS_STRIDE_IN_BYTES: index,    // LDS stride in bytes
    %ptr: !sx2,                     // The global base pointer
    %m_pos_global: index,           // The global outer-most tile position
    %n_pos_global: index,           // The global inner-most tile position
    %GLOBAL_STRIDE_IN_BYTES: index  // Global stride in bytes
  ) {
    %loaded = func.call @simple_lds_read_wave_16x16xf16_wait(
        %lds_base, %m_pos_lds, %n_pos_lds, %LDS_STRIDE_IN_BYTES)
      : (index, index, index, index) -> !vx2
    func.call @simple_global_store_wave_16x16xf16_wait(
        %loaded, %ptr, %m_pos_global, %n_pos_global, %GLOBAL_STRIDE_IN_BYTES)
      : (!vx2, !sx2, index, index, index) -> ()
    return
  }

}
