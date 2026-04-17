// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Transform sequence for 64x64 matmul: tile, pad, bufferize, map to AIR herd.
// Adapted from xrt/12 (tile-using-pad, no packing).

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Outer tiling: 2x1 forall on M (becomes 2-wavefront air.herd).
    // 64/32 = 2 iterations — non-trivial, survives canonicalization.
    %outer_tiled, %outer_forall =
      transform.structured.tile_using_forall %matmul
        tile_sizes [32, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Compute tiling inside forall: 16x16 M×N tiles, no K tiling.
    // mfma_matmul_f16_16x32 calls zero_C internally so K must not be tiled
    // across loop iterations (each call would reset the accumulator).
    // The library internally handles K by reading two 16x16 panels from LDS.
    %tiled, %lm, %ln = transform.structured.tile_using_for %outer_tiled
        tile_sizes [16, 16, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                   !transform.any_op)

    // Pad A and B only (not C — the matmul stores C directly to global).
    %padded, %pad, %copy_back = transform.structured.pad %tiled {
      padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f32],
      padding_dimensions = [0, 1, 2],
      pack_paddings = [1, 1, 0],
      nofold_flags = [1, 1, 0],
      copy_back_op = "linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                 !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad
        : (!transform.any_op) -> !transform.any_op

    // Promote A,B pads to L1 (memory_space=2) via bufferize_to_allocation.
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

    // One-shot bufferize.
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

    // Convert outer forall → parallel → air.herd (now on memrefs).
    %forall_2 = transform.structured.match ops{["scf.forall"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_2
        : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
