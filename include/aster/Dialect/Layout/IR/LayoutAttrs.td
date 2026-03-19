//===- LayoutAttrs.td - Layout attributes ----------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_IR_LAYOUTATTRS_TD
#define ASTER_DIALECT_LAYOUT_IR_LAYOUTATTRS_TD

include "aster/Dialect/Layout/IR/LayoutDialect.td"
include "mlir/IR/CommonAttrConstraints.td"

//===----------------------------------------------------------------------===//
// LayoutAttr: a strided layout with optional nesting = shape : stride
//===----------------------------------------------------------------------===//

def LayoutAttr : Layout_Attr<"Layout", "strided_layout"> {
  let summary = "A strided layout (shape : stride) with optional nesting";
  let description = [{
    A layout attribute representing a mapping from logical coordinates to
    physical offsets. Defined by a shape (sizes of each mode) and stride
    (offset multiplier for each mode).

    Uses first-mode-slowest convention (C row-major), matching upstream MLIR.
    The first shape element is the outermost (slowest-varying) dimension.

    The layout function evaluates as:
      L(c) = sum(delinearize(c, shape)[i] * stride[i])

    That is: delinearize the coordinate by the shape (first-mode-slowest),
    then dot-product with the strides. For nested modes, leaves are flattened
    before delinearization.

    Syntax (flat):
      #layout.strided_layout<[64] : [16]>       -- rank-1: offset = c * 16
      #layout.strided_layout<[4, 8] : [8, 1]>   -- rank-2: row-major 4x8

    Syntax (nested):
      #layout.strided_layout<[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>
        -- rank-2 with nested sub-modes: each top-level mode is subdivided

    Shape and stride must have the same nesting/tree structure.
    All leaf values must be non-negative integers.
    Stride 0 indicates broadcast (contrast this to upstream MLIR strided layout
    in which internal aliasing is specifically disallowed).
  }];

  // Both shape and stride are stored as ArrayAttr, which can contain
  // IntegerAttr (leaf) or nested ArrayAttr (sub-mode).
  let parameters = (ins
    "::mlir::ArrayAttr":$shape,
    "::mlir::ArrayAttr":$stride
  );

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Total number of logical elements (product of all leaf shape values).
    int64_t getSize() const;

    /// Span of physical addresses: 1 + max offset.
    int64_t getExtent() const;

    /// Number of top-level modes.
    int64_t getRank() const { return getShape().size(); }

    /// Evaluate the layout at an integral coordinate.
    int64_t evaluate(int64_t coord) const;

    /// Convenience builder from flat int arrays (no nesting).
    static LayoutAttr getFlat(::mlir::MLIRContext *ctx,
                              ::llvm::ArrayRef<int64_t> shape,
                              ::llvm::ArrayRef<int64_t> stride);
  }];
}

#endif // ASTER_DIALECT_LAYOUT_IR_LAYOUTATTRS_TD
