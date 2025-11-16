//===- AMDGCNPasses.td - AMDGCN Pass Definitions ----------------*- tblgen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AMDGCN pass definitions using tablegen.
//
//===----------------------------------------------------------------------===//

#ifndef AMDGCN_PASSES
#define AMDGCN_PASSES

include "mlir/Pass/PassBase.td"

def LowerToAMDGCNPass : Pass<"lower-to-amdgcn", "mlir::ModuleOp"> {
  let summary = "Lower a simple copy kernel to AMDGCN ops with fixed registers";
  let description = [{
    A very simple, prototype lowering that recognizes enough IR for a simple
    "copy" kernel.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def RegisterAlloc : Pass<"amdgcn-register-allocation"> {
  let summary = "Perform register allocation for AMDGCN kernels";
}

def RegisterDealloc : Pass<"amdgcn-register-dealloc", ""> {
  let summary = "Convert a register allocated kernel to an unallocated one";
}

def InstructionScheduling : Pass<"amdgcn-instruction-scheduling"> {
  let summary = "Schedule instructions based on delay and rate attributes";
  let description = [{
    This pass implements instruction scheduling by reading delay and rate
    attributes from operations and emitting them at the appropriate times.
    It proceeds by taking charge of the unrolling of particular scf.for loops
    marked with sched.apply and sched.dims attributes. It supports unrolling
    across multiple dimensions.

    Operations within the loop body can have the following attributes:
    - "sched.delay" : i32 - Initial delay before first emission
    - "sched.rate" : i32 - How often to emit (every N iterations)
    - "sched.permutation" : array<i32> - Optional permutation for dimension
      ordering. The convention is that the highest dimension is the fastest
      varying dimension. More specifically, assuming an (i, j, k) iteration
      space, of ranges [0, 1), [0, 2), [0, 3) and a permutation
      array<i32: 2, 0, 1>, the operations will be emitted in the order:
        - j fastest varying dimension
        - i second fastest varying dimension
        - k slowest varying dimension
      In other words, the operations will be emitted in the order:
        (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (0, 0, 2), (0, 1, 2)
  }];
  let options = [
    Option<"numIterations", "num-iterations", "unsigned", "1",
           "Number of global iterations to schedule">,
    ListOption<"dimensions", "dims", "int64_t",
               "Dimensions for multi-dimensional unrolling (e.g., --dims=4,4,4 for m,n,k)">,
    Option<"testOnly", "test-only", "bool", "false",
           "If true, add unroll attributes for testing purposes">,
  ];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect", "mlir::scf::SCFDialect"];
}

def InstructionSchedulingAutoschedule : Pass<"amdgcn-instruction-scheduling-autoschedule"> {
  let summary = "Apply autoschedule rules to operations without explicit schedules";
  let description = [{
    This pass applies autoschedule rules to operations without explicit schedule
    attributes in scf.for loops marked with sched.dims. It should run before the
    amdgcn-instruction-scheduling pass, if one want schedule automation.

    Operations without explicit schedules will inherit schedules based on:
    1. Their single consumer's schedule (if they have exactly one user)
    2. The smallest schedule from any consumer (if they have multiple users)
    3. Default schedule (delay=0, rate=1) otherwise

    This simple heuristic will at least keep SSA def-use chains valid for the
    autoscheduled operations but has no performance guarantees.

    In the future, better heuristics can be added.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

def Mem2Reg : Pass<"amdgcn-mem2reg"> {
  let summary = "Promote memory operations to register operations";
  let description = [{
    This pass promotes memory allocations to register usage where possible,
    converting load and store operations to direct register operations.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def ConstexprExpansion : Pass<"amdgcn-constexpr-expansion"> {
  let summary = "Expand operations tagged with amdgcn.constexpr attribute";
  let description = [{
    This pass unrolls scf.for loops that have the "amdgcn.constexpr" attribute.
    The attribute indicates that the loop bounds are compile-time constants
    and the loop should be fully unrolled for AMDGCN code generation.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

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
}

def ToAMDGCN : Pass<"aster-to-amdgcn"> {
  let summary = "Convert operations to AMDGCN dialect";
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

#endif // AMDGCN_PASSES
