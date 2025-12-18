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

def RegisterAlloc : Pass<"amdgcn-register-allocation"> {
  let summary = "Perform register allocation for AMDGCN kernels";
}

def RegisterDealloc : Pass<"amdgcn-register-dealloc"> {
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

def SetABI : Pass<"aster-amdgcn-set-abi"> {
  let summary = "Set AMDGCN ABI information";
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
}

def ToAMDGCN : InterfacePass<"aster-to-amdgcn", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert operations to AMDGCN dialect";
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::pdl::PDLDialect",
    "mlir::pdl_interp::PDLInterpDialect",
    "mlir::arith::ArithDialect"
  ];
  let options = [
    Option<"inputFile",
           "input-file",
           "std::string",
           "\"\"",
           "Input file to parse for ISel">,
  ];
}

def AMDGCNOptimizeStraightLineWaits : Pass<"amdgcn-optimize-straight-line-waits"> {
  let summary = "Optimize straight-line waits between first load and last store";
  let description = [{
    This pass identifies top-level blocks (not in loops) and finds the first load
    operation and last store operation in each block. All operations within the
    first-load, last-write range must be operations without regions and only
    AMDGCN operations.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    Option<"flushAllMemoryOnExit",
           "flush-all-memory-on-exit",
           "bool",
           "true",
           "If true, flush all pending memory operations before end_kernel">
  ];
}

def AMDGCNNopInsertion : Pass<"amdgcn-nop-insertion"> {
  let summary = "Insert NOP instructions in AMDGCN kernels";
  let description = [{
    This pass inserts NOP instructions in AMDGCN kernels.

    Section "4.5. Manually Inserted Wait States (NOPs)" of the "AMD Instinct MI300"
    Instruction Set Architecture manual lists the following dependencies that must
    be resolved by inserting NOPs or independent instructions.

    This pass inserts such dependencies.
  }];
  let dependentDialects = ["mlir::aster::amdgcn::AMDGCNDialect"];
  let options = [
    Option<"conservativeExtraDelays",
           "conservative-extra-delays",
           "unsigned",
           "0",
           "If non-zero, insert this many NOPs after each MFMA, global_load, ds_read, global_store, ds_write operation">
  ];
}

def HoistOps : InterfacePass<"aster-hoist-ops", "::mlir::FunctionOpInterface"> {
  let summary = "Hoist special AMDGCN operations to the entry block";
}

def ExpandMetadataOps : Pass<"aster-amdgcn-expand-md-ops", "::mlir::aster::amdgcn::KernelOp"> {
  let summary = "Expand special AMDGCN metadata operations";
  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];
}

def SelectRegClasses : InterfacePass<"aster-amdgcn-select-reg-classes", "mlir::aster::ModuleOpInterface"> {
  let summary = "Select register classes for AMDGCN instructions";
}

def PreloadLibrary : Pass<"amdgcn-preload-library", "mlir::ModuleOp"> {
  let summary = "Preload AMDGCN library functions into modules";
  let description = [{
    This pass preloads functions from amdgcn.library operations into amdgcn.module
    operations. For each amdgcn.module in the input, it:

    1. Parses library files from the specified paths
    2. For each func.call in the module that references an undefined function,
       looks up the function in the loaded libraries
    3. Clones the library function into the module

    This enables modular development of reusable kernel components.
  }];
  let dependentDialects = [
    "mlir::aster::amdgcn::AMDGCNDialect",
    "mlir::func::FuncDialect"
  ];
  let options = [
    ListOption<"libraryPaths", "library-paths", "std::string",
               "Paths to MLIR files containing amdgcn.library operations">
  ];
}

#endif // AMDGCN_PASSES
