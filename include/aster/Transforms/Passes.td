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
// CanonicalizePtrOps
//===----------------------------------------------------------------------===//

def CanonicalizePtrOps : Pass<"aster-canonicalize-ptr"> {
  let summary = "Canonicalize pointer computation";
  let description = [{
    This pass canonicalizes pointer computation operations.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::affine::AffineDialect"
  ];
}

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
// DecomposeMemrefIterArgs
//===----------------------------------------------------------------------===//

def DecomposeMemrefIterArgs
    : Pass<"aster-decompose-memref-iter-args"> {
  let summary = "Decompose memref iter_args into individual scalar iter_args";
  let description = [{
    Decomposes scf.for iter_args of type memref<?xT> (or memref<NxT>) into N
    individual scalar iter_args of type T. This handles rotating-buffer patterns
    post inlining and canonicalization simplications:

    - The init value is a memref.cast from a static memref.alloca with
      constant-index stores.
    - The yield value is a memref.cast from a static memref.alloca with
      constant-index stores.
    - In-loop and post-loop uses are all memref.load with constant indices.

    After decomposition, the memref allocas, casts, stores, and loads are
    replaced by direct scalar values flowing through iter_args. This enables
    subsequent SROA/mem2reg/canonicalize to further simplify the IR.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
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
  let description = [{
    This pass lowers affine dialect operations to standard arith operations
    and converts index types to integer types of the specified bitwidth.

    Post-condition: #amdgcn.no_affine_ops (set on amdgcn.module)
  }];
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
    Option<"lcmUnroll", "lcm-unroll", "bool", /*default=*/"false",
           "Unroll the kernel loop by the LCM of the pipeline stage values">,
    Option<"unrollFactorMultiplier", "unroll-factor-multiplier", "int64_t",
           /*default=*/"1",
           "Multiplier for the LCM unroll factor (effective = lcm * multiplier)">,
    Option<"epiloguePeeling", "epilogue-peeling", "bool", /*default=*/"false",
           "Fully unroll the cleanup epilogue loop after LCM unrolling">,
    Option<"enableRotateStage", "enable-rotate-stage", "bool",
           /*default=*/"false",
           "Enable rotating pipeline stages so the selected stage becomes stage 0">,
    Option<"rotateStage", "rotate-stage", "int64_t", /*default=*/"0",
           "Stage to rotate to 0 when enable-rotate-stage is set">,
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::aster::amdgcn::AMDGCNDialect",
  ];
}

//===----------------------------------------------------------------------===//
// SimplifyAllocaIterArgs
//===----------------------------------------------------------------------===//

def SimplifyAllocaIterArgs : Pass<"aster-simplify-alloca-iter-args"> {
  let summary = "Fold memref.cast on scf.for iter_args and deduplicate";
  let description = [{
    Folds memref.cast on scf.for iter_args when the source type is strictly
    more static than the dest type (e.g., memref<4xf32> -> memref<?xf32>).
    Combined with upstream folds (foldMemRefCast on loads/stores, cast-chain
    folding) and RemoveDuplicateSuccessorInputUses, this deduplicates paired
    alloca+cast iter_args created by the SCF pipeliner into a single
    static-typed iter_arg.

    This is the memref analogue of upstream's ForOpTensorCastFolder.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
  ];
}

//===----------------------------------------------------------------------===//
// AffineOptimizePtrAdd
//===----------------------------------------------------------------------===//

def AffineOptimizePtrAdd : Pass<"aster-affine-optimize-ptr-add"> {
  let summary = "Optimize ptr.ptr_add with affine_apply offsets using ValueBounds";
  let description = [{
    This pass optimizes `ptr.ptr_add` operations when the offset is produced by
    `affine.apply`. It decomposes the affine expression into const, uniform, and
    dynamic components using ValueBounds analysis (instead of IntRange) for
    constant detection, and ThreadUniformAnalysis for uniformity.
  }];
  let options = [
    Option<"assumePositive", "assume-positive", "bool", "false",
           "If true, assume all offset components are non-negative">,
  ];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::affine::AffineDialect"
  ];
}

//===----------------------------------------------------------------------===//
// ApplySched
//===----------------------------------------------------------------------===//

def ApplySched : Pass<"aster-apply-sched"> {
  let summary = "Apply scheduling attributes to blocks";
  let description = [{
    This pass collects SchedAttrInterface attributes from the discardable
    attribute dictionaries of operations (walking in preorder with regions),
    then applies each requested schedule in order.
    Note, that schedules are applied only to the immediately nested blocks.
    The discardable attribute is removed from the operation after the schedule
    is applied.

    If no schedules are requested, the pass uses "aster.sched" as the default
    schedule name.

    If a requested schedule is not found and silent-mode is true, the pass
    continues without erroring out. Otherwise, the pass errors out.
  }];
  let options = [
    ListOption<"scheds", "scheds", "std::string",
               "List of schedule attribute names to apply (in order)">,
    Option<"silentMode", "silent-mode", "bool", "true",
           "If true, missing schedules are silently ignored instead of "
           "erroring out">
  ];
}

//===----------------------------------------------------------------------===//
// LoopUnroll
//===----------------------------------------------------------------------===//

def LoopUnroll : Pass<"aster-loop-unroll"> {
  let summary = "Unroll scf.for loops by the largest applicable unroll factor";
  let description = [{
    This pass unrolls scf.for loops using the largest unroll factor that
    evenly divides the loop's trip count.

    For each loop, unrolling factors are determined as follows:
    1. If the loop has an "unroll_factor" attribute, it is tried first. The
       attribute may be either:
       - A single integer (e.g. unroll_factor = 4 : i64)
       - A dense int array (e.g. unroll_factor = array<i64: 2, 4, 8>)
       For an array, factors are tried in descending order; the first factor
       that divides the trip count is used.
    2. Otherwise, the pass uses the factors from the --unroll-factors option,
       tried in descending order.

    If loopUnrollByFactor fails, the pass fails.
  }];
  let options = [
    ListOption<"unrollFactors", "unroll-factors", "int64_t",
               "Unrolling factors to try (largest matching factor is used)">,
  ];
  let dependentDialects = ["mlir::scf::SCFDialect"];
}

//===----------------------------------------------------------------------===//
// FactorizeAffineExpr
//===----------------------------------------------------------------------===//

def FactorizeAffineExpr : Pass<"aster-factorize-affine-expr"> {
  let summary = "Factorize affine expressions to minimize multiplications";
  let description = [{
    Rewrites the affine expression inside each affine.apply op to minimize the
    number of multiplications via aggressive polynomial factorization.

    The algorithm expands the affine expression into a sum-of-monomials
    polynomial, then greedily extracts the most-frequent variable as a common
    factor, recursing on the quotient and remainder.  A post-processing pass
    then extracts any remaining common multiplicative factors shared across all
    addends of each addition node, and the two phases repeat until fixpoint.

    Examples:
      x * y + z * y  →  (x + z) * y
      x*y + z*y + x*w + z*w  →  (x + z) * (y + w)
      a*x*y + b*x*y + c*x + c*y  →  ((a + b) * y + c) * x + c * y
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}

//===----------------------------------------------------------------------===//
// CFGSimplification
//===----------------------------------------------------------------------===//

def CFGSimplification : Pass<"aster-cfg-simplification"> {
  let summary = "Simplify the control flow graph";
  let description = [{
    This pass performs two CFG simplification transformations:

    1. Block merging: if a block BBSucc has BBPred as its only predecessor and
       BBPred has BBSucc as its only successor, BBSucc is merged into BBPred.

    2. Empty block elimination: if a block contains only a single-successor
       terminator, all predecessors are redirected to branch directly to the
       successor and the now-unreachable block is removed.

    Both transformations are applied to fixpoint within each region.
  }];
  let dependentDialects = [];
}

#endif // ASTER_TRANSFORMS_PASSES_TD
