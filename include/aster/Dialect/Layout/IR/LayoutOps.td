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
// layout.apply
//===----------------------------------------------------------------------===//

def Layout_ApplyOp : Layout_Op<"apply", [Pure]> {
  let summary = "Evaluate a layout at a coordinate to produce an offset";
  let description = [{
    Evaluates a layout function at a coordinate to produce a 1D offset. This
    parallels `affine.apply`, which evaluates an affine map at operands.

    A layout L = (shape : strides) defines a mapping from a coordinate domain
    to a physical offset. `layout.apply` takes coordinates in one of two forms:

    1. Linear form: one coord. Recursive delinearize against the layout's
       shape tree, one `affine.delinearize_index` per nested mode against
       that mode's sub-shape product.
       The convention is first-mode-slowest (C row-major).

       Flat layout:
       ```mlir
       %off = layout.apply[%c], #layout.strided_layout<[4, 8] : [8, 1]>
       // lowers to:
       %c0, %c1 = affine.delinearize_index %c into (4, 8) : index, index
       %off = affine.linearize_index_by_strides[%c0, %c1] by (8, 1) : index
       ```

       Nested layout: one delinearize per nesting level.
       ```mlir
       // Layout <[(2,2), (2,4)] : [(1,4), (2,8)]> has rank 2 and flat_rank 4.
       %off = layout.apply[%c],
         #layout.strided_layout<[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>

       // lowers to:
       %a, %b = affine.delinearize_index %c into (4, 8) : index, index
       %a0, %a1 = affine.delinearize_index %a into (2, 2) : index, index
       %b0, %b1 = affine.delinearize_index %b into (2, 4) : index, index
       %off = affine.linearize_index_by_strides[%a0, %a1, %b0, %b1]
                                                by (1, 4, 2, 8) : index
       ```

    2. Decomposed form: as many coords as the flat rank of the layout. Dot
       coords with flat strides directly, no delinearization. For a nested
       layout this means one coord per *leaf*, not one per top-level mode.

       Flat layout:
       ```mlir
       %off = layout.apply[%c0, %c1], #layout.strided_layout<[4, 8] : [8, 1]>
       // lowers to:
       %off = affine.linearize_index_by_strides[%c0, %c1] by (8, 1) : index
       ```

       Nested layout: same example as form 1, flat_rank coords supplied
       directly so the delinearize levels disappear.
       ```mlir
       // Layout <[(2,2), (2,4)] : [(1,4), (2,8)]> has rank 2 and flat_rank 4.
       %off = layout.apply[%a0, %a1, %b0, %b1],
         #layout.strided_layout<[(2, 2), (2, 4)] : [(1, 4), (2, 8)]>
       // lowers to:
       %off = affine.linearize_index_by_strides[%a0, %a1, %b0, %b1]
                                                by (1, 4, 2, 8) : index
       ```

    The two forms coincide when flat_rank = 1.

    Verifier: the number of coords must be 1 or flat_rank.
  }];

  let arguments = (ins
    Variadic<Index>:$coords,
    LayoutAttr:$layout
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    `[` $coords `]` `,` $layout attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// layout.thread_value_offsets
//===----------------------------------------------------------------------===//

def Layout_ThreadValueOffsetsOp : Layout_Op<"thread_value_offsets", [Pure]> {
  let summary = "Return offsets of a thread-value layout sliced at thread tid";
  let description = [{
    Returns the list of offsets with length fixed by `value_layout.size()`:

        results[v] = thread_layout(tid) + value_layout(v)
                   for v in [0, value_layout.size())

    `thread_layout` and `value_layout` are independent strided layouts; each may
    be multi-dimensional or nested. Both must use the same stride units, either
    elements or bytes. The resulting offsets are in the same unit.

    The verifier enforces injectivity, so no two `(tid, v)` pairs alias the same
    offset.

    Degenerate `value_layout = <[1] : [0]>` is the single-result case
    (one transfer per thread).

    Example. 16 threads on a (4, 4) grid, 2 transfers per thread at byte stride
    2 between transfers:

    ```mlir
    %offs:2 = layout.thread_value_offsets[%tid]
      thread_layout = #layout.strided_layout<[4, 4] : [16, 4]>
      value_layout = #layout.strided_layout<[2] : [2]> -> index, index
    ```

    lowers to base offset + per-value `affine.apply`:

    ```mlir
    %d:2 = affine.delinearize_index %tid into (4, 4) : index, index
    %a   = affine.linearize_index_by_strides[%d#0, %d#1] by (16, 4) : index
    %r1  = affine.apply affine_map<(d0) -> (d0 + 2)>(%a)
    // offs:2 = %a, %r1
    ```

    Rank-N `thread_layout` produces a rank-N `affine.delinearize_index`. A nested
    `thread_layout` produces one delinearize per nesting level.
    Multi-dimensional `value_layout` contributes one offset per `v`.
  }];

  let arguments = (ins
    Index:$tid,
    LayoutAttr:$thread_layout,
    LayoutAttr:$value_layout
  );
  let results = (outs Variadic<Index>:$results);

  let assemblyFormat = [{
    `[` $tid `]`
    `thread_layout` `=` $thread_layout
    `value_layout` `=` $value_layout attr-dict `->` type($results)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// layout.swizzle
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
