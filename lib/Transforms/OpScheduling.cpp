//===- InstructionScheduling.cpp - Generator-based instruction scheduling -===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements generator-based instruction scheduling using delay and rate
// attributes on operations.
//
//===----------------------------------------------------------------------===//

#include "aster/Transforms/Passes.h"
#include "aster/Transforms/SchedUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::aster {
#define GEN_PASS_DEF_OPSCHEDULING
#include "aster/Transforms/Passes.h.inc"
} // namespace mlir::aster

#define DEBUG_TYPE "aster-op-scheduling"

namespace mlir::aster {
namespace {

// Attribute name constants
constexpr const char *kUnrollGlobalIdxAttr = "unroll.global_idx";
constexpr const char *kUnrollDimsAttr = "unroll.dims";

/// Structure to hold operation firing expression information.
struct OpFiringExpr {
  /// Construct OpFiringExpr from operation, delay, rate, and dimension
  /// expression.
  OpFiringExpr(Operation *op, AffineExpr d0, int delay, int rate)
      : op(op), delay(delay), rate(rate),
        firingExpr((d0 - delay) % static_cast<uint64_t>(rate)) {}

  /// Check if an operation should fire at the given global index.
  bool shouldFire(OpBuilder &b, int globalIdx) const;

  Operation *op;
  int delay;
  int rate;
  // globalIdx -> bool if this operation should fire at this globalIdx
  //   (d0 - delay) % rate == 0
  AffineExpr firingExpr;
};

/// Check if an operation should fire at the given global index.
bool OpFiringExpr::shouldFire(OpBuilder &b, int globalIdx) const {
  // Operation can't fire before its delay
  if (globalIdx < delay)
    return false;
  AffineExpr globalIdxExpr = b.getAffineConstantExpr(globalIdx);
  AffineExpr evaluatedExpr = firingExpr.replaceDims({globalIdxExpr});
  return evaluatedExpr == 0;
}

/// Data structure holding firing expressions and scheduling bounds.
struct FiringExpressionsAndBounds {
  int minDelay;
  int maxDelay;
  int maxRate;
  int maxGlobalIdx;
  llvm::DenseMap<Operation *, OpFiringExpr> opToExprMap;

  /// Get the firing expression for an operation, or nullptr if not found.
  const OpFiringExpr *getExpr(Operation *op) const {
    auto it = opToExprMap.find(op);
    return it != opToExprMap.end() ? &it->second : nullptr;
  }
};

/// Key to identify a unique operation instance (operation + iteration indices).
/// Used to map original operations to their cloned instances.
using OperationInstance = std::pair<Operation *, SmallVector<int64_t>>;
using MappingInstance = SmallVector<int64_t>;

/// Struct to hold scheduled operation information with pre-computed iteration
/// indices, following the permuted order of dimensions, if required.
struct ScheduledOp {
  int globalIdx;
  Operation *op;
  SmallVector<int64_t> iterationIndices;

  /// Get the key for looking up this operation instance in maps.
  /// Returns (op, iterationIndices) pair, ignoring globalIdx.
  OperationInstance getKey() const { return {op, iterationIndices}; }
};

} // namespace

/// Build firing expressions for all operations and calculate scheduling bounds.
static FiringExpressionsAndBounds
getFiringExpressionsAndBounds(ArrayRef<Operation *> opsToSchedule,
                              int totalNumIterations) {
  auto d0 = getAffineDimExpr(0, opsToSchedule.front()->getContext());
  llvm::DenseMap<Operation *, OpFiringExpr> opToExprMap;
  int minDelay = std::numeric_limits<int>::max(), maxDelay = 0, maxRate = 1;

  for (Operation *op : opsToSchedule) {
    int delay = op->getAttrOfType<IntegerAttr>(kSchedDelayAttr)
                    ? op->getAttrOfType<IntegerAttr>(kSchedDelayAttr).getInt()
                    : 0;
    int rate = op->getAttrOfType<IntegerAttr>(kSchedRateAttr)
                   ? op->getAttrOfType<IntegerAttr>(kSchedRateAttr).getInt()
                   : 1;
    assert(rate > 0 && "Rate must be positive");
    opToExprMap.try_emplace(op, op, d0, delay, rate);
    minDelay = std::min(minDelay, delay);
    maxDelay = std::max(maxDelay, delay);
    maxRate = std::max(maxRate, rate);
  }

  if (minDelay == std::numeric_limits<int>::max())
    minDelay = 0;
  return {minDelay, maxDelay, maxRate,
          totalNumIterations * maxRate + (maxDelay - minDelay),
          std::move(opToExprMap)};
}

/// Validates that the scheduled operations have unique iteration indices
/// and that all indices are within bounds.
static void validateSchedule(ArrayRef<ScheduledOp> scheduledOps,
                             ArrayRef<int64_t> shape, int totalNumIterations) {
  llvm::DenseMap<Operation *, llvm::DenseSet<SmallVector<int64_t>>>
      seenIterations;

  for (const ScheduledOp &schedOp : scheduledOps) {
    Operation *op = schedOp.op;
    const SmallVector<int64_t> &indices = schedOp.iterationIndices;

    // Check that indices are within bounds
    for (size_t i = 0; i < indices.size() && i < shape.size(); ++i) {
      assert(indices[i] >= 0 && indices[i] < shape[i] &&
             "Iteration index out of bounds");
    }

    // Check for duplicate iterations for the same operation
    auto &seenForOp = seenIterations[op];
    bool wasInserted = seenForOp.insert(indices).second;
    assert(wasInserted && "Duplicate iteration indices for the same operation");
  }

  // Verify that each operation has exactly totalNumIterations iterations
  for (auto &[op, iterations] : seenIterations) {
    assert(static_cast<int>(iterations.size()) == totalNumIterations &&
           "Operation does not have the expected number of iterations");
  }
}

/// Compute iteration indices for an operation instance, applying permutation if
/// present.
static SmallVector<int64_t> computeIterationIndices(Operation *op,
                                                    int instanceNumber,
                                                    ArrayRef<int64_t> shape) {
  int totalNumIterations = computeProduct(shape);
  if (totalNumIterations == 1)
    return {0};

  SmallVector<int64_t> strides = computeStrides(shape);

  auto permutationAttr =
      op->getAttrOfType<DenseI32ArrayAttr>(kSchedPermutationAttr);
  if (!permutationAttr ||
      static_cast<size_t>(permutationAttr.size()) != shape.size())
    return delinearize(instanceNumber, strides);

  // Apply permutation: delinearize in permuted space, then inverse permute
  SmallVector<int64_t> perm(permutationAttr.asArrayRef().begin(),
                            permutationAttr.asArrayRef().end());
  SmallVector<int64_t> permutedStrides =
      computeStrides(applyPermutation(shape, perm));
  return applyPermutation(delinearize(instanceNumber, permutedStrides),
                          invertPermutationVector(perm));
}

/// Build the list of scheduled operations by iterating over global indices
/// and checking which operations should fire at each index.
static SmallVector<ScheduledOp>
createOpSchedule(OpBuilder &b, const FiringExpressionsAndBounds &firingData,
                 ArrayRef<Operation *> opsToSchedule, ArrayRef<int64_t> shape) {
  int totalNumIterations = computeProduct(shape);
  llvm::DenseMap<Operation *, int> emissionCounts;
  for (Operation *op : opsToSchedule)
    emissionCounts[op] = 0;

  LDBG() << "Creating op schedule with max global index: "
         << firingData.maxGlobalIdx << "";
  SmallVector<ScheduledOp> scheduledOps;
  for (int globalIdx = 0; globalIdx < firingData.maxGlobalIdx + 1;
       ++globalIdx) {
    for (Operation *op : opsToSchedule) {
      if (emissionCounts[op] >= totalNumIterations)
        continue;

      const OpFiringExpr *opExpr = firingData.getExpr(op);
      if (!opExpr || !opExpr->shouldFire(b, globalIdx))
        continue;

      SmallVector<int64_t> iterationIndices =
          computeIterationIndices(op, emissionCounts[op], shape);

      LDBG_OS([&](raw_ostream &os) {
        os << "Scheduling operation '" << op->getName() << "' at global index "
           << globalIdx << " with iteration indices [";
        llvm::interleaveComma(iterationIndices, os);
        os << "]";
      });
      scheduledOps.push_back({globalIdx, op, std::move(iterationIndices)});
      emissionCounts[op]++;
    }
  }

  validateSchedule(scheduledOps, shape, totalNumIterations);
  return scheduledOps;
}

/// Collect all operations that should be scheduled from a loop body.
static SmallVector<Operation *> collectOpsToSchedule(scf::ForOp forOp) {
  SmallVector<Operation *> opsToSchedule;
  Block *loopBody = forOp.getBody();
  for (Operation &op : *loopBody) {
    // Skip the terminator
    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;
    opsToSchedule.push_back(&op);
  }
  return opsToSchedule;
}

/// Validates that all operands of an operation can be properly mapped
/// before cloning. Emits warnings for violations but doesn't fail.
static LogicalResult validateOperandMappings(const ScheduledOp &schedOp,
                                             const IRMapping &mapping) {
  for (OpOperand &operand : schedOp.op->getOpOperands()) {
    Value originalOperand = operand.get();

    // Check if it's in the IRMapping
    if (!mapping.contains(originalOperand)) {
      Operation *definingOp = originalOperand.getDefiningOp();
      if (definingOp) {
        schedOp.op->emitWarning()
            << "op scheduling: '" << schedOp.op->getName() << "' depends on '"
            << originalOperand << "' produced by '" << definingOp->getName()
            << "' which hasn't been cloned yet for this iteration instance. "
               "This indicates a scheduling violation - operations must be "
               "scheduled in dependency order. Check delay/rate attributes. "
               "Pass fails gracefully.";
      } else {
        schedOp.op->emitWarning()
            << "op scheduling: '" << schedOp.op->getName() << "' depends on '"
            << originalOperand
            << "' which isn't in the mapping. This indicates a scheduling "
               "violation. Check delay/rate attributes. Pass fails gracefully.";
      }
      // Pass fails gracefully.
      return failure();
    }
  }
  return success();
}

/// Clone and insert operations in scheduled order, setting unroll attributes.
/// Maintains SSA use-def chains by remapping operands to cloned producers.
/// Returns the list of cloned operations on success, or an empty vector on
/// failure.
static FailureOr<SmallVector<Operation *>>
materializeOpSchedule(OpBuilder &b, ArrayRef<ScheduledOp> scheduledOps,
                      llvm::DenseMap<MappingInstance, IRMapping> &irMappings,
                      bool testOnly) {
  assert(!scheduledOps.empty() && "expected ops to schedule");
  Block *block = scheduledOps.front().op->getBlock();
  assert(block && "expected block");

  Region *parentRegion = block->getParent();
  Block *tempBlock = new Block();
  parentRegion->push_back(tempBlock);

  Operation *originalTerminator = block->getTerminator();
  b.setInsertionPointToEnd(tempBlock);
  b.clone(*originalTerminator);
  b.setInsertionPoint(tempBlock->getTerminator());

  SmallVector<Operation *> clonedOps;

  // Clone operations in scheduled order
  for (const ScheduledOp &schedOp : scheduledOps) {
    IRMapping &mapping = irMappings[schedOp.iterationIndices];

    // Validate that all operands can be mapped before cloning
    if (failed(validateOperandMappings(schedOp, mapping))) {
      tempBlock->erase();
      return failure();
    }

    Operation *clonedOp = b.clone(*schedOp.op, mapping);
    clonedOp->removeAttr(kSchedDelayAttr);
    clonedOp->removeAttr(kSchedRateAttr);
    clonedOp->removeAttr(kSchedPermutationAttr);
    clonedOps.push_back(clonedOp);

    if (testOnly) {
      clonedOp->setAttr(kUnrollGlobalIdxAttr,
                        b.getI32IntegerAttr(schedOp.globalIdx));
      clonedOp->setAttr(kUnrollDimsAttr,
                        b.getDenseI64ArrayAttr(schedOp.iterationIndices));
    }
  }

  for (Operation *clonedOp : clonedOps)
    clonedOp->moveBefore(originalTerminator);

  tempBlock->erase();
  return clonedOps;
}

class OpSchedulingPass : public impl::OpSchedulingBase<OpSchedulingPass> {
public:
  using OpSchedulingBase::OpSchedulingBase;

  void runOnOperation() override {
    // Walk through all scf.for loops in the module
    getOperation()->walk([&](scf::ForOp forOp) { processLoop(forOp); });
  }

private:
  void processLoop(scf::ForOp forOp) {
    // Get dimensions from the loop's sched.dims attribute
    auto dimsAttr = forOp->getAttrOfType<DenseI64ArrayAttr>(kSchedDimsAttr);
    // No dimsAttr, no scheduling, no need for a warning.
    if (!dimsAttr)
      return;

    if (forOp.getNumResults() > 0) {
      forOp.emitWarning()
          << "op scheduling: skipping loop - loop yields values. Operation "
             "scheduling only supports loops that don't yield values atm.";
      return;
    }

    auto maybeConstantTripCount = forOp.getStaticTripCount();
    if (!maybeConstantTripCount.has_value()) {
      forOp.emitWarning()
          << "op scheduling: skipping loop - dynamic bounds detected. "
             "Operation scheduling requires constant trip count atm.";
      return;
    }

    SmallVector<int64_t> shape;
    shape.assign(dimsAttr.asArrayRef().begin(), dimsAttr.asArrayRef().end());
    int totalNumIterations = maybeConstantTripCount->getSExtValue();
    LDBG_OS([&](raw_ostream &os) {
      os << "Loop trip count: " << totalNumIterations << ", shape: [";
      llvm::interleaveComma(shape, os);
      os << "], product: " << computeProduct(shape);
    });

    if (totalNumIterations != computeProduct(shape)) {
      forOp.emitWarning()
          << "op scheduling: skipping loop - trip count (" << totalNumIterations
          << ") does not match product of dimensions (" << computeProduct(shape)
          << "). Ensure 'sched.dims' matches the actual loop iteration space.";
      return;
    }

    // Collect operations to schedule within the loop body
    SmallVector<Operation *> opsToSchedule = collectOpsToSchedule(forOp);
    if (opsToSchedule.empty()) {
      forOp.emitWarning()
          << "op scheduling: skipping loop - no operations to schedule in "
             "loop body (only terminator found).";
      return;
    }

    LDBG_OS([&](raw_ostream &os) {
      os << "Found " << opsToSchedule.size() << " ops, schedule with dims: [";
      llvm::interleaveComma(shape, os);
      os << "]";
    });

    // Preconditions done, start scheduling.
    OpBuilder b(forOp.getContext());
    FiringExpressionsAndBounds firingData =
        getFiringExpressionsAndBounds(opsToSchedule, totalNumIterations);

    SmallVector<ScheduledOp> scheduledOps =
        createOpSchedule(b, firingData, opsToSchedule, shape);
    assert(!scheduledOps.empty() && "expected ops to schedule");

    // Unroll and materialize the schedule
    b.setInsertionPoint(forOp);
    Value loopInductionVar = forOp.getInductionVar();
    Type inductionVarType = loopInductionVar.getType();
    SetVector<Value> valuesDefinedAbove;
    getUsedValuesDefinedAbove(forOp.getRegion(), valuesDefinedAbove);
    llvm::DenseMap<MappingInstance, IRMapping> irMappings;
    for (int i = 0; i < totalNumIterations; ++i) {
      MappingInstance mappingInstance =
          computeIterationIndices(forOp, i, shape);
      IRMapping mapping;
      Value constantValue;
      if (isa<IndexType>(inductionVarType)) {
        constantValue = b.create<arith::ConstantIndexOp>(forOp.getLoc(), i);
      } else if (auto intType = dyn_cast<IntegerType>(inductionVarType)) {
        TypedAttr attr = b.getIntegerAttr(intType, i);
        constantValue = b.create<arith::ConstantOp>(forOp.getLoc(), attr);
      } else {
        llvm_unreachable("Unsupported loop induction variable type");
      }
      mapping.map(loopInductionVar, constantValue);
      mapping.map(valuesDefinedAbove, valuesDefinedAbove);
      irMappings[mappingInstance] = mapping;
    }

    FailureOr<SmallVector<Operation *>> clonedOps =
        materializeOpSchedule(b, scheduledOps, irMappings, testOnly);
    if (failed(clonedOps)) {
      forOp.emitWarning()
          << "op scheduling: failed to materialize operation schedule. "
             "Check previous warnings for scheduling violations. Loop will "
             "not be unrolled.";
      return;
    }

    // Move all the cloned operations before the loop
    for (Operation *op : clonedOps.value())
      op->moveBefore(forOp);

    // Erase the original loop
    forOp.erase();
  }
};
} // namespace mlir::aster
