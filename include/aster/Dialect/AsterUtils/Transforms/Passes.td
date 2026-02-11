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

def FromAnyToPoison : Pass<"aster-from-any-to-poison"> {
  let summary = "Transform FromAnyOp to ub.poison";
  let description = [{
    This pass transforms `aster_utils.from_any` operations into `ub.poison`
    operations. This is useful for lowering type-erased values to undefined
    behavior when the original type cannot be recovered.
  }];
  let dependentDialects = [
    "mlir::ub::UBDialect"
  ];
}

#endif // ASTER_DIALECT_ASTERUTILS_TRANSFORMS_PASSES_TD
