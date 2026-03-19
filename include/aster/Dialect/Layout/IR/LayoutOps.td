//===- LayoutOps.td - Layout operations -----------------*- tablegen -*-===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LAYOUT_IR_LAYOUTOPS_TD
#define ASTER_DIALECT_LAYOUT_IR_LAYOUTOPS_TD

include "aster/Dialect/Layout/IR/LayoutDialect.td"
include "aster/Dialect/Layout/IR/LayoutAttrs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// layout.linearize -- evaluate a layout at a runtime coordinate
//===----------------------------------------------------------------------===//

def Layout_LinearizeOp : Layout_Op<"linearize", [Pure]> {
  let summary = "Evaluate a layout at a coordinate to produce an offset";
  let description = [{
    Maps a 1D logical coordinate to a 1D physical offset using a layout
    attribute. This is a "layout-aware linearize" that handles the
    delinearization internally.

    The lowering decomposes into two steps:
      1. `affine.delinearize_index` -- extract coordinates from the shape
      2. `affine.linearize_index_by_strides` -- dot product with explicit strides

    For a rank-2 layout `[4, 8] : [1, 4]`:
    ```mlir
    %off = layout.linearize %c, #layout.strided_layout<[4, 8] : [1, 4]>
    ```
    lowers to:
    ```mlir
    %c0, %c1 = affine.delinearize_index %c into (4, 8) : index, index
    %off = affine.linearize_index_by_strides [%c0, %c1] by (1, 4) : index
    ```

    For a rank-1 layout `[64] : [16]`:
    ```mlir
    %off = layout.linearize %c, #layout.strided_layout<[64] : [16]>
    ```
    lowers to:
    ```mlir
    %off = affine.linearize_index_by_strides [%c] by (16) : index
    ```
  }];

  let arguments = (ins
    Index:$coord,
    LayoutAttr:$layout
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $coord `,` $layout attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// layout.swizzle -- XOR-based address swizzle
//===----------------------------------------------------------------------===//

def Layout_SwizzleOp : Layout_Op<"swizzle", [Pure]> {
  let summary = "Apply an XOR-based swizzle to an offset";
  let description = [{
    Applies a bit-manipulation swizzle to a physical offset:
      result = offset ^ ((offset >> shift) & (((1 << bits) - 1) << base))

    This is used for LDS bank conflict avoidance. The swizzle XORs a group
    of `bits` bits starting at position `base` with the same number of bits
    shifted by `shift` positions.

    Example (bits=2, base=3, shift=4):
      Bits [3:5] are XORed with bits [7:9] of the offset.

    ```mlir
    %swizzled = layout.swizzle %offset, bits = 2, base = 3, shift = 4
    ```
    lowers to:
    ```mlir
    %mask = arith.constant 24 : index          // ((1 << 2) - 1) << 3 = 0b11000
    %shift = arith.constant 4 : index
    %shifted = arith.shrui %offset, %shift : index
    %masked = arith.andi %shifted, %mask : index
    %result = arith.xori %offset, %masked : index
    ```
  }];

  let arguments = (ins
    Index:$offset,
    I64Attr:$bits,
    I64Attr:$base,
    I64Attr:$shift
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $offset `,` `bits` `=` $bits `,` `base` `=` $base `,` `shift` `=` $shift attr-dict
  }];
}

#endif // ASTER_DIALECT_LAYOUT_IR_LAYOUTOPS_TD
