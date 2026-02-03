//===- Passes.td - AsterUtils passes -----------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD
#define ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// AsterUtils passes
//===----------------------------------------------------------------------===//

def AsterSelectiveInlining : Pass<"aster-selective-inlining", "mlir::ModuleOp"> {
  let summary = "Selective inlining pass with ASTER-specific behavior";
  let description = [{
    This pass performs inlining similar to the standard inline pass, but with
    ASTER-specific behavior. By default, calls with sched.* attributes are not
    inlined, but this can be controlled via the allow-scheduled-calls option.
  }];
  let options = [
    Option<"allowScheduledCalls",
           "allow-scheduled-calls",
           "bool",
           "false",
           "If true, allow inlining calls that have sched.* attributes">,
  ];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect"
  ];
}

def InlineExecuteRegion : Pass<"aster-inline-execute-region"> {
  let summary = "Inline aster_utils.execute_region operations";
  let description = [{
    This pass inlines `aster_utils.execute_region` operations by replacing
    them with the contents of their regions. The yield operation results
    are replaced with the yielded values.
  }];
}

def WrapCallsWithExecuteRegion : Pass<"aster-wrap-calls-with-execute-region"> {
  let summary = "Wrap function calls with aster_utils.execute_region";
  let description = [{
    This pass wraps function calls with `aster_utils.execute_region` operations.
  }];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect"
  ];
}

def FromAnyToPoison : Pass<"aster-from-any-to-poison"> {
  let summary = "Transform FromAnyOp to ub.poison";
  let description = [{
    This pass transforms `aster_utils.from_any` operations into `ub.poison`
    operations. This is useful for lowering type-erased values when the
    original type cannot be recovered.
  }];
  let dependentDialects = [
    "mlir::ub::UBDialect"
  ];
}

def OptimizePtrAdd : Pass<"aster-optimize-ptr-add"> {
  let summary = "Optimize ptr.ptr_add operations by splitting offsets";
  let description = [{
    This pass optimizes `ptr.ptr_add` operations by analyzing the offset
    expression tree and splitting it into three components:
    - `const_offset`: A constant offset known at compile time
    - `uniform_offset`: An offset that is uniform across all threads
    - `offset`: The remaining non-uniform offset

    The pass walks the definition of the offset operand while it's comprised
    of supported arithmetic operations (muli, shli, shrui, shrsi, divui,
    divsi, addi), and extracts constant and uniform components from the
    expression tree.

    The transformation requires that all three resulting offsets are
    non-negative to ensure correctness.

    Example transformation:
    ```mlir
    // Before: x1 = (a + b + y) * t0 + u
    // where b and t0 are uniform, y and u are constants
    %ptr = ptr.ptr_add %p, %x1 : !ptr.ptr, i64

    // After:
    %ptr = aster_utils.ptr_add %p, %offset, %uniform const_offset = u
        : !ptr.ptr, i64
    // where uniform = t0 * (y + b), offset = t0 * a
    ```
  }];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD
