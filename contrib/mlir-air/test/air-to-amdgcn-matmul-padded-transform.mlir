// Transform sequence for padded matmul with device-side padding.
// A/B are 40x64 (actual), NOT padded. C is 48x48 (padded for safe stores).
//
// The matmul is 40x40x64. tile_sizes [16, 16, 0] creates a 3x3 grid:
//   - Interior tiles (2x2): full 16x16 output, full 16x64 A/B loads
//   - M-boundary (row 2): 8xN output, 8x64 A loads
//   - N-boundary (col 2): Mx8 output, 8x64 B loads
//   - Corner (2,2): 8x8 output, 8x64 A and B loads
//
// After air-split-launch-for-padding, boundary DMAs get pad_after attribute.
// ConvertToAMDGCNLibraryCalls emits fill_f16_16x64 + copy_f16_16x64_padded
// for boundary tiles.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Tile 40x40 matmul into 3x3 grid. Boundary tiles have 8 rows/cols.
    %outer_tiled, %outer_forall =
      transform.structured.tile_using_forall %matmul
        tile_sizes [16, 16, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad A,B for LDS promotion (16x64 tiles; K untiled).
    // nofold ensures ALL tiles (including full interior tiles) go through
    // the pad→alloc→copy path, so all get air.dma_memcpy_nd ops.
    %padded, %pad, %copy_back = transform.structured.pad %outer_tiled {
      padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f32],
      padding_dimensions = [0, 1, 2],
      pack_paddings = [1, 1, 0],
      nofold_flags = [1, 1, 0]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                 !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad
        : (!transform.any_op) -> !transform.any_op

    // Promote padded A,B to LDS (memory_space = 2 = shared/LDS).
    %padded_lhs = transform.get_producer_of_operand %padded[0]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_a, %new_a = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 2, bufferize_destination_only}
        : !transform.any_op
    %padded_rhs = transform.get_producer_of_operand %padded[1]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_b, %new_b = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 2, bufferize_destination_only}
        : !transform.any_op

    // Canonicalize.
    %func_0 = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_0 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_0 : !transform.any_op

    // Bufferize.
    %func_1 = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %func_buf = transform.bufferization.one_shot_bufferize %func_1 {
      allow_return_allocs_from_loops = true
    } : (!transform.any_op) -> !transform.any_op

    // Cleanup.
    %func_2 = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_2 : !transform.any_op
    %func_3 = transform.air.remove_uninitialized_copy %func_2
        : (!transform.any_op) -> !transform.any_op

    // forall → parallel → launch (M and N both in launch).
    %forall_2 = transform.structured.match ops{["scf.forall"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_2
        : (!transform.any_op) -> !transform.any_op
    %launch = transform.air.par_to_launch %parallel
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
