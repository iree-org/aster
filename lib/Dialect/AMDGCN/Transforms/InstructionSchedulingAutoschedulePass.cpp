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
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"

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

/// Apply autoschedules to operations that don't have explicit ones.
static void applyAutoschedules(SmallVector<Operation *> &opsToSchedule) {
  // Process each operation without a schedule (in reverse order)
  for (size_t i = opsToSchedule.size(); i > 0; --i) {
    Operation *op = opsToSchedule[i - 1];
    if (hasSchedule(op))
      continue;

    // Rule 1: Gather all consumers with schedules and select the earliest one
    // (first by delay, then by rate)
    SmallVector<Operation *> consumersWithSchedule =
        gatherConsumersWithSchedule(op, opsToSchedule);
    Operation *earliestConsumer =
        findConsumerWithEarliestSchedule(consumersWithSchedule);
    if (earliestConsumer) {
      // Apply Rule 1: inherit schedule from consumer with earliest schedule
      copyScheduleAttributes(earliestConsumer, op);
      int delay =
          earliestConsumer->getAttrOfType<IntegerAttr>(kSchedDelayAttr)
              ? earliestConsumer->getAttrOfType<IntegerAttr>(kSchedDelayAttr)
                    .getInt()
              : 0;
      int rate =
          earliestConsumer->getAttrOfType<IntegerAttr>(kSchedRateAttr)
              ? earliestConsumer->getAttrOfType<IntegerAttr>(kSchedRateAttr)
                    .getInt()
              : 1;
      LDBG() << "Operation '" << op->getName()
             << "' inherits schedule from consumer '"
             << earliestConsumer->getName() << "' (delay=" << delay
             << ", rate=" << rate << ")\n";
      continue;
    }

    // Rule 2: Apply default schedule (delay=0, rate=1, no permutation)
    OpBuilder b(op->getContext());
    op->setAttr(kSchedDelayAttr, b.getI32IntegerAttr(0));
    op->setAttr(kSchedRateAttr, b.getI32IntegerAttr(1));
    LDBG() << "Operation '" << op->getName()
           << "' gets default schedule (delay=0, rate=1)\n";
  }
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
    applyAutoschedules(opsToSchedule);
  }
};

} // namespace
} // namespace amdgcn
} // namespace mlir::aster
