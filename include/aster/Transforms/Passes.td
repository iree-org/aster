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
// DecomposeAffineApply
//===----------------------------------------------------------------------===//

def DecomposeAffineApply : Pass<"aster-decompose-affine-apply"> {
  let summary = "Decompose large affine.apply ops into smaller reusable pieces";
  let description = [{
    Splits large affine.apply operations into chains of smaller affine.apply
    ops, reordered from least to most hoistable operands. This exposes more
    opportunities for LICM, CSE, and int-range-based optimizations.

    Wraps upstream affine::decompose() and reorderOperandsByHoistability().

    Example:
      %r = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1 * 4 + 42)>()[%a, %b]
    becomes:
      %c42 = affine.apply affine_map<() -> (42)>()
      %t0 = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%b]
      %t1 = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%c42, %t0]
      %r = affine.apply affine_map<()[s0, s1] -> (s0 * 8 + s1)>()[%a, %t1]

    Note: canonicalization tries to maximally compose chains of affine.apply,
    undoing this decomposition. Run this pass after canonicalization and before
    lowering affine to arith.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
  ];
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
// PromoteLoopCarriedMemrefs
//===----------------------------------------------------------------------===//

def PromoteLoopCarriedMemrefs
    : Pass<"aster-promote-loop-carried-memrefs"> {
  let summary = "Promote scalar memref allocas with loop-carried patterns to "
                "scf.for iter_args";
  let description = [{
    Upstream mem2reg promotes memref allocas to SSA values within
    straight-line code using dominance. However, it cannot promote across
    loop boundaries: inside an scf.for body the store from iteration N-1
    does not dominate the load in iteration N (the body is a nested
    region). Introducing iter_args + yield requires scf-specific rewriting
    outside the scope of upstream mem2reg.

    The motivating case is constexpr-expanded GEMM templates. After constexpr
    expansion + SROA + mem2reg, accumulators that cross the non-constexpr K-loop.

    This pass handles scalar memref allocas (0-d or 1-element) with exactly one
    pre-loop store, one in-loop load-before-store pair, and one post-loop load:

    ```
    memref.store %init, %alloca[] : memref<T>
    scf.for %k = ... {
      %old = memref.load %alloca[] : memref<T>
      %new = compute(%old)
      memref.store %new, %alloca[] : memref<T>
    }
    %final = memref.load %alloca[] : memref<T>
    ```
    becomes:
    ```
    %final = scf.for %k = ... iter_args(%acc = %init) -> (T) {
      %new = compute(%acc)
      scf.yield %new : T
    }
    ```
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

    If no schedules are requested, the pass errors out.
  }];
  let options = [
    ListOption<"scheds", "scheds", "std::string",
               "List of schedule attribute names to apply (in order)">
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
// ForwardStoreToLoad
//===----------------------------------------------------------------------===//

def ForwardStoreToLoad : Pass<"aster-forward-store-to-load"> {
  let summary = "Forward memref stores to subsequent loads in the same block";
  let description = [{
    Per-block reaching-def analysis on memref values (allocas and block args).
    Handles both scalar memrefs (rank-0 or [1]) and indexed memrefs (rank-1
    with static shape and constant indices), tracking reaching definitions per
    (memref, index) slot.
    When a store is followed by a load from the same slot with no interleaved
    aliasing operation, the load is replaced by the stored value. Dead stores
    (overwritten before any read) are also eliminated, along with dead allocas
    and dead stores to memref values with no load users.
    This pass is meant to be used with and to augment SROA / mem2reg and
    PromoteLoopCarriedMemrefs to support more cases.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
  ];
}

#endif // ASTER_TRANSFORMS_PASSES_TD
