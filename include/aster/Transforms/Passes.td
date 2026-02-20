//===- Passes.td - Aster passes ----------------------------*- tablegen -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TRANSFORMS_PASSES_TD
#define ASTER_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ConstexprExpansion
//===----------------------------------------------------------------------===//

def ConstexprExpansion : Pass<"aster-constexpr-expansion"> {
  let summary = "Expand operations tagged with aster.constexpr attribute";
  let description = [{
    This pass unrolls scf.for loops that have the "aster.constexpr" attribute.
    The attribute indicates that the loop bounds are compile-time constants
    and the loop should be fully unrolled.
  }];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// OpScheduling
//===----------------------------------------------------------------------===//

def OpScheduling : Pass<"aster-op-scheduling"> {
  let summary = "Schedule operations based on delay and rate attributes";
  let description = [{
    This pass implements operation scheduling by reading delay and rate
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
               "Dimensions for multi-dimensional unrolling (e.g., --dims=4,4,4 "
               "for m,n,k)">,
    Option<"testOnly", "test-only", "bool", "false",
           "If true, add unroll attributes for testing purposes">,
  ];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// ReplaceConstantGPUDims
//===----------------------------------------------------------------------===//

def ReplaceConstantGPUDims : InterfacePass<"aster-replace-constant-gpu-dims",
                                           "mlir::aster::ModuleOpInterface"> {
  let summary = "Replace gpu.block_dim and gpu.grid_dim with constants";
  let description = [{
    This pass identifies functions implementing GPUFuncInterface and replaces
    gpu.block_dim and gpu.grid_dim operations with constant values based on
    the block and grid dimension annotations on the function.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::gpu::GPUDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ToIntArith
//===----------------------------------------------------------------------===//

def ToIntArith
    : InterfacePass<"aster-to-int-arith", "mlir::aster::ModuleOpInterface"> {
  let summary = "Convert index operations to integer operations";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
  ];
  let options = [
    Option<"bitwidth", "bitwidth", "unsigned", "32",
           "Index bitwidth to convert to">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::aster::aster_utils::AsterUtilsDialect",
  ];
}

//===----------------------------------------------------------------------===//
// OptimizeArith
//===----------------------------------------------------------------------===//

def OptimizeArith
    : InterfacePass<"aster-optimize-arith", "mlir::aster::ModuleOpInterface"> {
  let summary = "Optimize arithmetic operations";
}

//===----------------------------------------------------------------------===//
// SCFPipelineAsterSched
//===----------------------------------------------------------------------===//

def SCFPipelineAsterSched : Pass<"aster-scf-pipeline"> {
  let summary = "Pipeline SCF loops using stage-based scheduling";
  let description = [{
    This pass pipelines scf.for loops whose operations are annotated with
    sched.stage attributes. It generates prologue (ramp-up), kernel
    (steady-state), and epilogue (drain) sections.

    Operations without a sched.stage attribute default to stage 0.

    Example (2-stage, load at stage 0, compute at stage 1):
      Prologue: load(0)
      Kernel:   for i in [1, N): load(i), compute(i-1)
      Epilogue: compute(N-1)
  }];
  let options = [
    Option<"gcdUnroll", "gcd-unroll", "bool", /*default=*/"false",
           "Unroll the kernel loop by the GCD of the pipeline stage values">,
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::aster::amdgcn::AMDGCNDialect",
  ];
}

#endif // ASTER_TRANSFORMS_PASSES_TD
