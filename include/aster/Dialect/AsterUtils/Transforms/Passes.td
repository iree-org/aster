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

def DestructureStructIterArgs : Pass<"aster-destructure-struct-iter-args"> {
  let summary = "Destructure struct-typed scf.for iter_args into individual fields";
  let description = [{
    This pass replaces struct-typed iter_args in scf.for loops with individual
    field-typed iter_args. For each struct iter_arg:
    - Init values: struct_extract decomposes the init value
    - Loop body entry: struct_create reconstructs from individual block args
    - Yield: struct_extract decomposes back into individual fields
    - Loop results: struct_create reconstructs for downstream users

    After this pass, canonicalization can fold the struct_create/struct_extract
    pairs that become adjacent within each scope.

    This is needed after SCF pipelining, which may create struct-typed iter_args
    when cross-stage values are bundled in structs (e.g., futures).
  }];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect"
  ];
}

def ResolveAnyIterArgs : Pass<"aster-resolve-any-iter-args"> {
  let summary = "Resolve any-typed iter_args and block args to concrete types";
  let description = [{
    This pass specializes `!aster_utils.any`-typed values to their concrete
    types by analyzing the to_any/from_any chain.

    Handles two patterns:

    **scf.for iter_args:** Init must be `to_any %x : T`, yield must be
    `to_any %y : T`, and all block arg / result uses must be `from_any : T`.
    Supports passthrough rotation patterns from the SCF pipeliner.

    **CF block arguments:** All predecessor branches (cf.br / cf.cond_br)
    must provide `to_any %x : T` for the same type T, and all uses within
    the block must be `from_any : T`. Passthrough chains across blocks are
    handled via fixpoint iteration.
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

def RemoveAssumeOps : Pass<"aster-remove-assume-ops", "mlir::ModuleOp"> {
  let summary = "Remove assume_uniform, assume_range, and passthrough operations";
  let description = [{
    This pass removes aster_utils assume and passthrough operations by replacing
    them with their input values. Options control which op kinds are removed:

    - remove-uniform: remove all aster_utils.assume_uniform ops
    - remove-range: remove all aster_utils.assume_range ops
    - remove-passthrough: remove all aster_utils.passthrough ops
    - tag-regex: remove passthrough ops whose tag attribute matches the regex
      (LLVM regex syntax). Only applies to passthrough ops that have a tag.
  }];
  let options = [
    Option<"removeUniform",
           "remove-uniform",
           "bool",
           "false",
           "Remove all aster_utils.assume_uniform operations">,
    Option<"removeRange",
           "remove-range",
           "bool",
           "false",
           "Remove all aster_utils.assume_range operations">,
    Option<"removePassthrough",
           "remove-passthrough",
           "bool",
           "false",
           "Remove all aster_utils.passthrough operations">,
    Option<"tagRegex",
           "tag-regex",
           "std::string",
           "\"\"",
           "Remove passthrough ops with tags matching this regex (LLVM regex)">,
  ];
}


def RaiseToAffine : Pass<"aster-raise-to-affine"> {
  let summary = "Raise aster_utils.addi and muli to affine.apply";
  let description = [{
    This pass converts `aster_utils.addi` and `aster_utils.muli` operations
    to `affine.apply` operations, and applies affine canonicalization patterns
    greedily to simplify the resulting affine expressions.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect"
  ];
}

def DecomposeByLoopInvariant : Pass<"aster-decompose-by-loop-invariant"> {
  let summary = "Split loop-invariant operands of addi/muli ops into inner ops";
  let description = [{
    This pass splits the loop-invariant operands of `aster_utils.addi` and
    `aster_utils.muli` operations into a separate op of the same kind, inserting
    an `aster_utils.passthrough` barrier between the two ops to prevent the
    optimizer from re-merging them.

    For example, if `%a` and `%b` are defined outside the enclosing loop:

    ```mlir
    // Before:
    %r = aster_utils.addi %a, %b, %c : index

    // After:
    %t0 = aster_utils.addi %a, %b : index
    %t1 = aster_utils.passthrough %t0 : index
    %r  = aster_utils.addi %t1, %c : index
    ```

    After this transformation, standard LICM can hoist `%t0` and `%t1` out of
    the loop, leaving only `%r` inside.
  }];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect"
  ];
}


def ExpandAffineApply : Pass<"aster-expand-affine-apply"> {
  let summary = "Expand affine.apply into aster_utils n-ary addi/muli operations";
  let description = [{
    This pass converts `affine.apply` operations into trees of
    `aster_utils.addi` and `aster_utils.muli` by traversing the affine
    expression tree. Consecutive additions and multiplications are collected
    into flat n-ary operand lists instead of binary chains.

    Constant sub-expressions are materialized as `arith.constant` index ops.
    Add and mul sub-expressions are expanded; mod, floordiv, and ceildiv remain
    as `affine.apply`.

    Example:
    ```mlir
    // Before:
    %r = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()
                      [%a, %b, %c, %d]

    // After:
    %r = aster_utils.addi %a, %b, %c, %d : index
    ```
  }];
  let dependentDialects = [
    "mlir::aster::aster_utils::AsterUtilsDialect",
    "mlir::arith::ArithDialect"
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
