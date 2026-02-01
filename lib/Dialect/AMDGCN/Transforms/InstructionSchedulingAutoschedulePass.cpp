//===- InstructionSchedulingAutoschedulePass.cpp - Autoschedule pass ------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Transforms/SchedUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"

#include <cassert>

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_INSTRUCTIONSCHEDULINGAUTOSCHEDULE
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

#define DEBUG_TYPE "amdgcn-instruction-scheduling-autoschedule"

namespace mlir::aster {
namespace amdgcn {
namespace {

/// Check if an operation has a schedule (both delay and rate attributes).
static bool hasSchedule(Operation *op) {
  return op->hasAttr(kSchedDelayAttr) && op->hasAttr(kSchedRateAttr);
}

/// Copy schedule attributes from source to target operation.
static void copyScheduleAttributes(Operation *source, Operation *target) {
  if (auto delayAttr = source->getAttrOfType<IntegerAttr>(kSchedDelayAttr))
    target->setAttr(kSchedDelayAttr, delayAttr);
  if (auto rateAttr = source->getAttrOfType<IntegerAttr>(kSchedRateAttr))
    target->setAttr(kSchedRateAttr, rateAttr);
  if (auto permAttr =
          source->getAttrOfType<DenseI32ArrayAttr>(kSchedPermutationAttr))
    target->setAttr(kSchedPermutationAttr, permAttr);
}

/// Gather all consumers of the operation's results that have schedules.
static SmallVector<Operation *>
gatherConsumersWithSchedule(Operation *op,
                            ArrayRef<Operation *> opsToSchedule) {
  SmallVector<Operation *> consumersWithSchedule;
  for (Value result : op->getResults()) {
    for (Operation *consumer : result.getUsers()) {
      if (llvm::is_contained(opsToSchedule, consumer)) {
        // By our reverse traversal order, any consumer in opsToSchedule must
        // have already been processed and thus have a schedule.
        assert(hasSchedule(consumer) &&
               "consumer in opsToSchedule must have a schedule");
        consumersWithSchedule.push_back(consumer);
      }
    }
  }
  return consumersWithSchedule;
}

/// Find the consumer with the earliest schedule (delay first, then rate).
/// Returns nullptr if no consumer with a schedule is found.
static Operation *
findConsumerWithEarliestSchedule(ArrayRef<Operation *> consumersWithSchedule) {
  if (consumersWithSchedule.empty())
    return nullptr;

  Operation *earliestConsumer = nullptr;
  int earliestDelay = std::numeric_limits<int>::max();
  int earliestRate = std::numeric_limits<int>::max();

  for (Operation *consumer : consumersWithSchedule) {
    int delay =
        consumer->getAttrOfType<IntegerAttr>(kSchedDelayAttr)
            ? consumer->getAttrOfType<IntegerAttr>(kSchedDelayAttr).getInt()
            : 0;
    int rate =
        consumer->getAttrOfType<IntegerAttr>(kSchedRateAttr)
            ? consumer->getAttrOfType<IntegerAttr>(kSchedRateAttr).getInt()
            : 1;

    if (delay < earliestDelay ||
        (delay == earliestDelay && rate < earliestRate)) {
      earliestDelay = delay;
      earliestRate = rate;
      earliestConsumer = consumer;
    }
  }

  return earliestConsumer;
}

/// Compute the maximum delay among operand producers that have schedules.
/// Returns 0 if no operand producer has a schedule.
static int computeMaxOperandDelay(Operation *op,
                                  ArrayRef<Operation *> opsToSchedule) {
  int maxDelay = 0;
  for (Value operand : op->getOperands()) {
    Operation *producer = operand.getDefiningOp();
    if (!producer || !llvm::is_contained(opsToSchedule, producer))
      continue;
    if (auto delayAttr = producer->getAttrOfType<IntegerAttr>(kSchedDelayAttr))
      maxDelay = std::max(maxDelay, static_cast<int>(delayAttr.getInt()));
  }
  return maxDelay;
}

/// Propagate schedule from a parent operation to all nested operations.
static void propagateScheduleToNestedOps(Operation *parent) {
  for (Region &region : parent->getRegions()) {
    region.walk([&](Operation *nestedOp) {
      if (!hasSchedule(nestedOp))
        copyScheduleAttributes(parent, nestedOp);
    });
  }
}

/// Apply autoschedules to operations that don't have explicit ones.
/// Returns failure if conflicting constraints are detected.
static LogicalResult
applyAutoschedules(SmallVector<Operation *> &opsToSchedule) {
  // Phase 1: Autoschedule all top-level ops (in reverse order)
  // Does not recurse into nested regions.
  for (size_t i = opsToSchedule.size(); i > 0; --i) {
    Operation *op = opsToSchedule[i - 1];
    if (hasSchedule(op))
      continue;

    // Compute the minimum delay required by operand constraints.
    // The operation must be scheduled at a delay >= max of its operand delays.
    int minDelayFromOperands = computeMaxOperandDelay(op, opsToSchedule);

    // Rule 1: Gather all consumers with schedules and select the earliest one
    // (first by delay, then by rate)
    SmallVector<Operation *> consumersWithSchedule =
        gatherConsumersWithSchedule(op, opsToSchedule);
    Operation *earliestConsumer =
        findConsumerWithEarliestSchedule(consumersWithSchedule);
    if (earliestConsumer) {
      int consumerDelay =
          earliestConsumer->getAttrOfType<IntegerAttr>(kSchedDelayAttr)
              ? earliestConsumer->getAttrOfType<IntegerAttr>(kSchedDelayAttr)
                    .getInt()
              : 0;
      int rate =
          earliestConsumer->getAttrOfType<IntegerAttr>(kSchedRateAttr)
              ? earliestConsumer->getAttrOfType<IntegerAttr>(kSchedRateAttr)
                    .getInt()
              : 1;

      // Check for conflicting constraints: consumer wants earlier than operands
      // allow
      if (consumerDelay < minDelayFromOperands) {
        op->emitWarning()
            << "autoschedule conflict: consumer '"
            << earliestConsumer->getName()
            << "' requires delay=" << consumerDelay
            << ", but operand constraints require delay>="
            << minDelayFromOperands
            << ". Please add explicit sched.delay/sched.rate attributes to "
               "resolve this conflict.";
        return failure();
      }

      // Apply Rule 1: inherit schedule from consumer with earliest schedule
      copyScheduleAttributes(earliestConsumer, op);
      LDBG() << "Operation '" << op->getName()
             << "' inherits schedule from consumer '"
             << earliestConsumer->getName() << "' (delay=" << consumerDelay
             << ", rate=" << rate << ")\n";
      continue;
    }

    // Rule 2: Apply schedule with delay=max(0, operand delays), rate=1
    OpBuilder b(op->getContext());
    op->setAttr(kSchedDelayAttr, b.getI32IntegerAttr(minDelayFromOperands));
    op->setAttr(kSchedRateAttr, b.getI32IntegerAttr(1));
    LDBG() << "Operation '" << op->getName()
           << "' gets schedule (delay=" << minDelayFromOperands
           << ", rate=1) based on operand constraints\n";
  }

  // Phase 2: Propagate schedule to nested ops for ops with regions
  for (Operation *op : opsToSchedule) {
    if (op->getNumRegions() > 0) {
      propagateScheduleToNestedOps(op);
      LDBG() << "Propagated schedule to nested ops of '" << op->getName()
             << "'\n";
    }
  }

  return success();
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

class InstructionSchedulingAutoschedulePass
    : public impl::InstructionSchedulingAutoscheduleBase<
          InstructionSchedulingAutoschedulePass> {
public:
  using InstructionSchedulingAutoscheduleBase::
      InstructionSchedulingAutoscheduleBase;

  void runOnOperation() override {
    // Walk through all scf.for loops in the module
    getOperation()->walk([&](scf::ForOp forOp) { processLoop(forOp); });
  }

private:
  void processLoop(scf::ForOp forOp) {
    // Get dimensions from the loop's sched.dims attribute
    auto dimsAttr = forOp->getAttrOfType<DenseI64ArrayAttr>(kSchedDimsAttr);
    if (!dimsAttr) {
      LDBG() << "Loop missing sched.dims attribute, skipping\n";
      return;
    }

    // Collect operations to schedule within the loop body
    SmallVector<Operation *> opsToSchedule = collectOpsToSchedule(forOp);
    if (opsToSchedule.empty())
      return;

    // Apply autoschedules
    if (failed(applyAutoschedules(opsToSchedule)))
      signalPassFailure();
  }
};

} // namespace
} // namespace amdgcn
} // namespace mlir::aster
