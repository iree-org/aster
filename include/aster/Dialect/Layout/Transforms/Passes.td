//===- Passes.td - Layout passes ---------------------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_TRANSFORMS_PASSES_TD
#define ASTER_DIALECT_LAYOUT_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LowerLayoutToAffine : Pass<"lower-layout-to-affine"> {
  let summary = "Lower layout ops to affine.delinearize_index + affine.linearize_index_by_strides";
  let description = [{
    Lowers `layout.linearize` and `layout.swizzle` to `affine` + `arith`.

    For multi-rank layouts, the lowering is:
      layout.linearize %c, <[s0, s1] : [d0, d1]>
      =>
      %c0, %c1 = affine.delinearize_index %c into (s0, s1)
      %off = affine.linearize_index_by_strides[%c0, %c1] by (d0, d1) : index

    For rank-1 layouts:
      layout.linearize %c, <[s0] : [d0]>
      =>
      %off = affine.linearize_index_by_strides[%c] by (d0) : index

    For nested layouts, the lowering flattens the shape and stride trees to
    their leaves before emitting delinearize + linearize_index_by_strides.

    For `layout.swizzle`, the lowering emits arith bit operations (index_cast,
    shrui, andi, xori) implementing the XOR swizzle formula.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // ASTER_DIALECT_LAYOUT_TRANSFORMS_PASSES_TD
