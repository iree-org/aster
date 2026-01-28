// Source of truth for all descriptors reused across mlir_kernels.
// New descriptors should be added here once they have proven generally useful.

//===----------------------------------------------------------------------===//
// Type aliases (required for descriptors)
//===----------------------------------------------------------------------===//

// Scalar General Purpose Registers (SGPR)
!s   = !amdgcn.sgpr
!sx1 = !amdgcn.sgpr_range<[? + 1]>
!sx2 = !amdgcn.sgpr_range<[? + 2]>
!sx3 = !amdgcn.sgpr_range<[? + 3]>
!sx4 = !amdgcn.sgpr_range<[? + 4]>

// Vector General Purpose Registers (VGPR)
!v   = !amdgcn.vgpr
!vx1 = !amdgcn.vgpr_range<[? + 1]>
!vx2 = !amdgcn.vgpr_range<[? + 2]>
!vx3 = !amdgcn.vgpr_range<[? + 3]>
!vx4 = !amdgcn.vgpr_range<[? + 4]>

// Accumulator General Purpose Registers (AGPR)
!a   = !amdgcn.agpr
!ax1 = !amdgcn.agpr_range<[? + 1]>
!ax2 = !amdgcn.agpr_range<[? + 2]>
!ax3 = !amdgcn.agpr_range<[? + 3]>
!ax4 = !amdgcn.agpr_range<[? + 4]>

//===----------------------------------------------------------------------===//
// Index descriptors
//===----------------------------------------------------------------------===//

// A 2D index pair containing row (i) and column (j) indices.
!index_pair = !aster_utils.struct<i: index, j: index>

// An 8-tuple of index values (used for LDS bank indices)
// Note: MLIR doesn't support tuple type aliases directly, so we use a struct
!index_tuple_8 = !aster_utils.struct<b0: index, b1: index, b2: index, b3: index, b4: index, b5: index, b6: index, b7: index>

// A 2D index descriptor containing:
//   - i, j: row and column indices
//   - stride: stride in elements (not bytes)
//   - elt_size_b: element size in bytes
!index_descriptor_2d = !aster_utils.struct<i: index, j: index, stride: index, elt_size_b: index>

// A 2-level 2D index descriptor containing:
//   - i, j: row and column indices of the outer tile
//   - ii, jj: row and column indices of the inner tile
//   - stride: stride in elements (not bytes)
//   - elt_size_b: element size in bytes
!index_descriptor_2level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, stride: index, elt_size_b: index>

// A 3-level 2D index descriptor containing:
//   - i, j: The outer-most major tile position (e.g. the start row of a tile)
//   - ii, jj: The outer-most minor tile position (e.g. the start row of the sub-tile)
//   - iii, jjj: The outer-most position (e.g. a row relative to the sub-tile)
//   - stride: The stride (e.g. inner 2-D tile size **in bytes**)
//   - elt_size_b: The element size **in bytes**
!index_descriptor_3level_2d = !aster_utils.struct<i: index, j: index, ii: index, jj: index, iii: index, jjj: index, stride: index, elt_size_b: index>

//===----------------------------------------------------------------------===//
// Position descriptors
//===----------------------------------------------------------------------===//

// A 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!tensor_position_descriptor_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, elt_size: index>

// A 2-level 2D tensor position descriptor containing:
//   - ptr: global base pointer
//   - m_pos, n_pos: row and column positions of the outer tile (in elements)
//   - global_stride_in_bytes: stride in bytes
//   - mm_pos, nn_pos: row and column positions of the inner tile (in elements)
//   - elt_size: element size in bytes
!tensor_position_descriptor_2level_2d = !aster_utils.struct<ptr: !sx2, m_pos: index, n_pos: index, global_stride_in_bytes: index, mm_pos: index, nn_pos: index, elt_size: index>

// A 2D LDS position descriptor containing:
//   - lds_base: local base offset in LDS
//   - m_pos, n_pos: row and column positions (in elements)
//   - lds_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!lds_position_descriptor_2d = !aster_utils.struct<lds_base: index, m_pos: index, n_pos: index, lds_stride_in_bytes: index, elt_size: index>

// A 2-level 2D LDS position descriptor containing:
//   - lds_base: local base offset in LDS
//   - mm_pos, nn_pos: row and column positions of the minor tile (in elements)
//   - lds_stride_in_bytes: stride in bytes
//   - elt_size: element size in bytes
!lds_position_descriptor_2level_2d = !aster_utils.struct<lds_base: index, mm_pos: index, nn_pos: index, lds_stride_in_bytes: index, elt_size: index>

//===----------------------------------------------------------------------===//
// Transfer descriptors
//===----------------------------------------------------------------------===//

// A 2D transfer descriptor containing:
//   - num_rows: number of rows for the transfer (must divide wave_size evenly)
//   - transfer_size: size of each transfer in bytes
//   - wave_size: number of threads per wave
!transfer_descriptor_2d = !aster_utils.struct<num_rows: index, transfer_size: index, wave_size: index>

//===----------------------------------------------------------------------===//
// Execution and scheduling descriptors
//===----------------------------------------------------------------------===//

// A 2D conditional execution descriptor for multi-tile operations containing:
//   - k: outer loop index (for indexing load_memref -> mem2reg)
//   - cond_iter: condition index (execute only when cond_iter == 0)
//   - NT_I, NT_J: multi-tile factors (process NT_I x NT_J tiles at once)
!conditional_execution_descriptor_2d = !aster_utils.struct<k: index, cond_iter: index, NT_I: index, NT_J: index>
