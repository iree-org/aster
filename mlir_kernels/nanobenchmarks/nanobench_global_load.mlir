// Nanobenchmark for @global_load_wave_xxx_nowait
// Measures global load performance with data fitting in L1 cache.

// From descriptors.mlir
!sx2 = !amdgcn.sgpr<[? + 2]>
!vx1 = !amdgcn.vgpr
!vx2 = !amdgcn.vgpr<[? + 2]>
!vx3 = !amdgcn.vgpr<[? + 3]>
!vx4 = !amdgcn.vgpr<[? + 4]>
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

amdgcn.module @nanobench_module target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
  // From indexing.mlir
  func.func private @wave_id() -> index
  func.func private @wave_count() -> index
  // From copies.mlir
  func.func private @global_load_wave_128xf16_via_dword_nowait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx1
  func.func private @global_load_wave_256xf16_via_dwordx2_nowait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx2
  func.func private @global_load_wave_384xf16_via_dwordx3_nowait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx3
  func.func private @global_load_wave_512xf16_via_dwordx4_nowait(
    !tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx4

  amdgcn.kernel @nanobench_global_load arguments <[
    #amdgcn.buffer_arg<address_space = generic, access = read_only>
  ]> attributes {block_dims = array<i32: {{NUM_THREADS}}, 1, 1>, grid_dims = array<i32: {{NUM_BLOCKS}}, 1, 1>} {
    %ptr_s = amdgcn.load_arg 0 : !sx2
    %ptr = lsir.assume_noalias %ptr_s : (!sx2) -> !sx2
    amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> lgkmcnt = 0

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    // Parameters for {{NUM_TILES}} tile configuration
    %NT_J = arith.constant {{NUM_TILES}} : index

    // Tile size (in units of f16 elements)
    %TILE_SIZE = affine.apply affine_map<()[] -> ({{TILE_SIZE_BYTES}} floordiv 2)>()[]

    // Tile reuse factor
    %TILE_REUSE_FACTOR = arith.constant {{TILE_REUSE_FACTOR}} : index

    // DWORDX bitfield: bit0=dword, bit1=dwordx2, bit2=dwordx3, bit3=dwordx4
    %DWORDXBITS = arith.constant {{DWORDXBITS}} : index
    %c8 = arith.constant 8 : index
    %elt_size = arith.constant 2 : index

    %wave_id = func.call @wave_id() : () -> index
    %WAVE_COUNT = func.call @wave_count() : () -> index
    %block_id_x = gpu.block_id x
    %n_pos = affine.apply affine_map<()[block_id_x, wave_id, WAVE_COUNT, NT_J, TILE_SIZE]
      -> ((block_id_x * WAVE_COUNT + wave_id) * NT_J * TILE_SIZE)>()
          [%block_id_x, %wave_id, %WAVE_COUNT, %NT_J, %TILE_SIZE]

    // Test bit 0 for dword
    %bit0 = arith.andi %DWORDXBITS, %c1 : index
    %run_dword = arith.cmpi ne, %bit0, %c0 : index
    scf.if %run_dword {
      //===--------------------------------------------------------------------===//
      // dword
      //===--------------------------------------------------------------------===//
      scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
        scf.for %nt = %c0 to %NT_J step %c1 {
          %nn_pos = affine.apply affine_map<()[nt, TILE_SIZE] -> (nt * TILE_SIZE)>()[%nt, %TILE_SIZE]
          %transfer_size_vx1 = arith.constant 4 : index // dword
          %wave_size_vx1 = arith.constant 64 : index
          %pos_desc_vx1 = aster_utils.struct_create(%ptr, %c0, %n_pos, %c0, %c0, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
          %transfer_desc_vx1 = aster_utils.struct_create(%c1, %transfer_size_vx1, %wave_size_vx1) : (index, index, index) -> !transfer_descriptor_2d
          %result_vx1 = func.call @global_load_wave_128xf16_via_dword_nowait(%pos_desc_vx1, %transfer_desc_vx1) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx1
          amdgcn.test_inst ins %result_vx1 : (!vx1) -> ()
        } {aster.constexpr}
        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>
      } {aster.constexpr}
    }

    // Test bit 1 for dwordx2
    %bit1 = arith.andi %DWORDXBITS, %c2 : index
    %run_dwordx2 = arith.cmpi ne, %bit1, %c0 : index
    scf.if %run_dwordx2 {
      //===--------------------------------------------------------------------===//
      // dwordx2
      //===--------------------------------------------------------------------===//
      scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
        scf.for %nt = %c0 to %NT_J step %c2 {
          %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
          %transfer_size_vx2 = arith.constant 8 : index // dwordx2
          %wave_size_vx2 = arith.constant 64 : index
          %pos_desc_vx2 = aster_utils.struct_create(%ptr, %c0, %n_pos, %c0, %c0, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
          %transfer_desc_vx2 = aster_utils.struct_create(%c1, %transfer_size_vx2, %wave_size_vx2) : (index, index, index) -> !transfer_descriptor_2d
          %result_vx2 = func.call @global_load_wave_256xf16_via_dwordx2_nowait(%pos_desc_vx2, %transfer_desc_vx2) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx2
          amdgcn.test_inst ins %result_vx2 : (!vx2) -> ()
        } {aster.constexpr}
        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>
      } {aster.constexpr}
    }

    // Test bit 2 for dwordx3
    %bit2 = arith.andi %DWORDXBITS, %c4 : index
    %run_dwordx3 = arith.cmpi ne, %bit2, %c0 : index
    scf.if %run_dwordx3 {
      //===--------------------------------------------------------------------===//
      // dwordx3
      //===--------------------------------------------------------------------===//
      scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
        scf.for %nt = %c0 to %NT_J step %c3 {
          %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
          %transfer_size_vx3 = arith.constant 12 : index // dwordx3
          %wave_size_vx3 = arith.constant 64 : index
          %pos_desc_vx3 = aster_utils.struct_create(%ptr, %c0, %n_pos, %c0, %c0, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
          %transfer_desc_vx3 = aster_utils.struct_create(%c1, %transfer_size_vx3, %wave_size_vx3) : (index, index, index) -> !transfer_descriptor_2d
          %result_vx3 = func.call @global_load_wave_384xf16_via_dwordx3_nowait(%pos_desc_vx3, %transfer_desc_vx3) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx3
          amdgcn.test_inst ins %result_vx3 : (!vx3) -> ()
        } {aster.constexpr}
        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>
      } {aster.constexpr}
    }

    // Test bit 3 for dwordx4
    %bit3 = arith.andi %DWORDXBITS, %c8 : index
    %run_dwordx4 = arith.cmpi ne, %bit3, %c0 : index
    scf.if %run_dwordx4 {
      //===--------------------------------------------------------------------===//
      // dwordx4
      //===--------------------------------------------------------------------===//
      scf.for %reuse = %c0 to %TILE_REUSE_FACTOR step %c1 {
        scf.for %nt = %c0 to %NT_J step %c4 {
          %nn_pos = affine.apply affine_map<()[nn, TILE_SIZE] -> (nn * TILE_SIZE)>()[%nt, %TILE_SIZE]
          %transfer_size_vx4 = arith.constant 16 : index // dwordx4
          %wave_size_vx4 = arith.constant 64 : index
          %pos_desc_vx4 = aster_utils.struct_create(%ptr, %c0, %n_pos, %c0, %c0, %nn_pos, %elt_size) : (!sx2, index, index, index, index, index, index) -> !tensor_position_descriptor_2level_2d
          %transfer_desc_vx4 = aster_utils.struct_create(%c1, %transfer_size_vx4, %wave_size_vx4) : (index, index, index) -> !transfer_descriptor_2d
          %result_vx4 = func.call @global_load_wave_512xf16_via_dwordx4_nowait(%pos_desc_vx4, %transfer_desc_vx4) : (!tensor_position_descriptor_2level_2d, !transfer_descriptor_2d) -> !vx4
          amdgcn.test_inst ins %result_vx4 : (!vx4) -> ()
        } {aster.constexpr}
        amdgcn.sopp.s_waitcnt #amdgcn.inst<s_waitcnt> vmcnt = 0
        amdgcn.sopp.sopp <s_barrier>
      } {aster.constexpr}
    }

    amdgcn.end_kernel
  }
}
