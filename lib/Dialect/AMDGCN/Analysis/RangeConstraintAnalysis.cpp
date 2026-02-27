//===- RangeConstraintAnalysis.cpp - Range constraint analysis ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/IR/PrintingUtils.h"
#include "aster/IR/SSAMap.h"
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include <deque>

#define DEBUG_TYPE "amdgcn-range-constraint-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
/// Represents a range operation and its allocas.
struct RangeInfo {
  MakeRegisterRangeOp op;
  ValueRange allocas;
  int64_t idx;
  void print(llvm::raw_ostream &os) const {
    os << "range[" << idx << "] = " << op;
  }
};

/// Implements the range constraint analysis.
struct RangeConstraintAnalysisImpl {
  RangeConstraintAnalysisImpl(SmallVector<RangeConstraint> &constraints)
      : constraints(constraints) {}

  /// Run the analysis on the given operation. Returns failure if the analysis
  /// fails.
  LogicalResult run(Operation *op);

private:
  /// Compute the equivalence classes for the ranges. Returns the number of
  /// equivalence classes.
  int64_t computeRangeClasses(int64_t numAllocas);

  /// Merge the equivalence classes of the ranges. Returns failure if the merge
  /// is not possible.
  LogicalResult mergeClasses(int64_t numEqCLasses);

  /// Set the alignment for the given range. Returns failure if the alignment
  /// constraint is unsatisfiable.
  LogicalResult checkAlignment(RangeInfo &range);

  Operation *topOp;
  SmallVector<RangeInfo> rangeOps;
  SmallVector<RangeConstraint> &constraints;
  DenseMap<Value, int64_t> valueToConstraintIdx;
};
} // namespace

//===----------------------------------------------------------------------===//
// RangeConstraint
//===----------------------------------------------------------------------===//

void RangeConstraint::print(llvm::raw_ostream &os,
                            const mlir::aster::SSAMap &ssaMap) const {
  os << "range_constraint<alignment = " << alignment << ", allocations = [";
  SmallVector<std::pair<Value, int64_t>> ids;
  ssaMap.getIds(allocations, ids);
  llvm::interleaveComma(ids, os, [&](const std::pair<Value, int64_t> &entry) {
    os << entry.second << " = `" << ValueWithFlags(entry.first, true) << "`";
  });
  os << "]>";
}

//===----------------------------------------------------------------------===//
// Free functions
//===----------------------------------------------------------------------===//

static inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                            const RangeInfo &info) {
  info.print(os);
  return os;
}

/// Compute the common subrange of two value ranges. Returns the start index in
/// the allocas and constraints, and the size of the common subrange.
static std::tuple<int64_t, int64_t, int64_t>
commonSubrange(ValueRange allocas, ValueRange constraints,
               SmallPtrSetImpl<Value> &allocationsSet) {
  int64_t aBegin = -1;
  // Find the first element of the allocas that is in the allocations set.
  for (auto [i, alloc] : llvm::enumerate(allocas)) {
    if (allocationsSet.contains(alloc)) {
      aBegin = i;
      break;
    }
  }
  if (aBegin == -1)
    return {0, 0, 0};

  int64_t aSize = allocas.size();
  int64_t commonSize = 0, aPos = aBegin, cBegin = -1;

  // Get the common subrange of the allocas and the constraints.
  for (auto [i, alloc] : llvm::enumerate(constraints)) {
    if (aPos >= aSize)
      break;

    // Try to match the allocas and the constraints.
    if (allocas[aPos] == alloc) {
      if (cBegin == -1)
        cBegin = i;
      ++commonSize;
      ++aPos;
      continue;
    }

    // At this point the ranges stopped matching, return the common subrange.
    if (commonSize > 0)
      return {aBegin, cBegin, commonSize};
  }
  return {aBegin, cBegin, commonSize};
}

/// Merge the range operation with the range constraint. Returns failure if
/// the merge is not possible. Returns true if the merge succeeded, false
/// if there's no enough information to merge.
static FailureOr<bool> merge(RangeConstraint &constraint, ValueRange allocas,
                             int32_t alignment,
                             SmallPtrSetImpl<Value> &allocationsSet) {
  assert(allocas.size() > 0 && "allocas must not be empty");
  LDBG() << "Merging:\n  " << llvm::interleaved_array(allocas) << "\n  "
         << llvm::interleaved_array(constraint.allocations);

  // If the constraint is empty, set the alignment and append the allocas.
  if (constraint.allocations.empty()) {
    constraint.alignment = alignment;
    llvm::append_range(constraint.allocations, allocas);
    allocationsSet.insert_range(allocas);
    LDBG() << "  - Successfully merged";
    return true;
  }

  // Get the common subrange of the allocas and the constraint.
  auto [allocSubOff, cstrSubOff, commonSize] =
      commonSubrange(allocas, constraint.allocations, allocationsSet);

  LDBG() << "  Common subrange: " << allocSubOff << " - " << cstrSubOff << " - "
         << commonSize;

  // If there's no common subrange, return false as there's no enough
  // information to merge.
  if (commonSize == 0) {
    // NOTE: The ranges could be incompatible here, but for efficiency we
    // don't test, eventually the `mergeClasses` will fail due to no changes.
    LDBG() << "  - Cannot merge: no common ordered subrange";
    return false;
  }

  // If the common subrange is not at the beginning of either range, we cannot
  // merge.
  if (allocSubOff != 0 && cstrSubOff != 0) {
    LDBG() << "  - Failed to merge: common subrange is not at the beginning of "
              "either range";
    return failure();
  }

  // Get the start of the allocas and constraints in the merged range,
  // NOTE: If for a range, `off == 0` then there might be new elements to the
  // left in the merged range. Otherwise, the range lies at the beginning of the
  // merged range.
  int64_t allocOffInRange = allocSubOff == 0 ? cstrSubOff : 0;
  int64_t cstrOffInRange = cstrSubOff == 0 ? allocSubOff : 0;

  // Check the alignments can be respected in the merged range.
  if (allocOffInRange % alignment != 0 ||
      cstrOffInRange % constraint.alignment != 0) {
    LDBG() << "  - Failed to merge: alignments prevent merging the ranges";
    return failure();
  }

  // Make sure the merged range always respects both range alignments.
  constraint.alignment = std::lcm(alignment, constraint.alignment);

  // Get the allocas to the left and right of the common subrange.
  ValueRange lhsAllocas = allocas.slice(0, allocSubOff);
  ValueRange rhsAllocas = allocas.drop_front(allocSubOff + commonSize);

  // In this case the allocas are already in the constraint.
  if (lhsAllocas.empty() && rhsAllocas.empty()) {
    LDBG() << "  - Success, range was already in the constraint";
    return true;
  }

  // If there are allocas to the left and right, then the allocas contain the
  // constraint, therefore prefer vector.assign.
  if (!lhsAllocas.empty() && !rhsAllocas.empty()) {
    constraint.allocations.assign(allocas.begin(), allocas.end());
  } else {
    constraint.allocations.insert(constraint.allocations.begin(),
                                  lhsAllocas.begin(), lhsAllocas.end());
    constraint.allocations.insert(constraint.allocations.end(),
                                  rhsAllocas.begin(), rhsAllocas.end());
  }
  allocationsSet.insert_range(lhsAllocas);
  allocationsSet.insert_range(rhsAllocas);
  LDBG() << "  - Success, merged the ranges";
  return true;
}

//===----------------------------------------------------------------------===//
// RangeConstraintAnalysisImpl
//===----------------------------------------------------------------------===//

LogicalResult RangeConstraintAnalysisImpl::run(Operation *op) {
  topOp = op;
  int64_t numRanges = 0, numAllocas = 0;

  // Collect all the range operations, and count the number of allocas.
  WalkResult result = op->walk([&](Operation *op) {
    if (isa<AllocaOp>(op)) {
      ++numAllocas;
      return WalkResult::advance();
    }

    auto rOp = dyn_cast<MakeRegisterRangeOp>(op);
    if (!rOp)
      return WalkResult::advance();
    LDBG() << "Initializing range: " << rOp;

    // This analysis assumes that the range operation always has alloca as
    // inputs.
    FailureOr<ValueRange> allocas = getAllocasOrFailure(rOp);
    if (failed(allocas)) {
      LDBG() << "  - Failed to get allocas for the range";
      return WalkResult::interrupt();
    }
    rangeOps.push_back({rOp, *allocas, numRanges++});
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  // If there are no ranges, return success.
  if (numRanges == 0)
    return success();

  // Compute the equivalence classes for the ranges.
  int64_t numEqClasses = computeRangeClasses(numAllocas);

  // Merge the ranges into constraints.
  if (failed(mergeClasses(numEqClasses)))
    return failure();

  // Assert the alignment constraints are satisfiable.
  assert(llvm::all_of(rangeOps,
                      [&](RangeInfo &range) {
                        return succeeded(checkAlignment(range));
                      }) &&
         "alignment constraints should be satisfiable by construction");
  return success();
}

int64_t RangeConstraintAnalysisImpl::computeRangeClasses(int64_t numAllocas) {
  llvm::IntEqClasses eqClasses;
  eqClasses.grow(rangeOps.size());

  DenseMap<Value, uint32_t> allocaToConstraint;
  allocaToConstraint.reserve(numAllocas);

  LDBG() << "Computing range classes";

  // Compute the equivalence classes for the ranges.
  for (auto &&[i, info] : llvm::enumerate(rangeOps)) {
    LDBG() << "Analyzing range: " << info;
    for (Value alloc : info.allocas) {
      // Get or create the equivalence class for the alloca.
      uint32_t id = allocaToConstraint.insert({alloc, i}).first->getSecond();
      LDBG() << "  Alloca: " << alloc << " -> " << id << ", " << i;
      if (id == i)
        continue;
      LDBG() << "   Joining classes: " << id << " and " << i;
      // Join the equivalence classes of the range and the alloca.
      eqClasses.join(i, id);
    }
  }
  eqClasses.compress();

  // Assign the equivalence class to the range.
  for (RangeInfo &info : rangeOps)
    info.idx = eqClasses[info.idx];

  // Sort the ranges by the equivalence class.
  llvm::sort(rangeOps, [&](RangeInfo lhs, RangeInfo rhs) {
    return std::make_tuple(lhs.idx, lhs.allocas.size()) <
           std::make_tuple(rhs.idx, rhs.allocas.size());
  });

  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "Range classes:\n";
    llvm::interleave(
        rangeOps, os, [&](RangeInfo info) { info.print(os); }, "\n");
  });

  return eqClasses.getNumClasses();
}

LogicalResult RangeConstraintAnalysisImpl::mergeClasses(int64_t numEqClasses) {
  std::deque<RangeInfo> ranges;
  llvm::append_range(ranges, rangeOps);
  SmallVector<llvm::SmallPtrSet<Value, 4>> inserted(numEqClasses);
  constraints.assign(numEqClasses, RangeConstraint());

  // Merge the ranges until no more changes are made.
  while (true) {
    bool changed = false;
    int64_t it = 0, maxIt = ranges.size();

    while (!ranges.empty() && it++ < maxIt) {
      RangeInfo range = ranges.front();
      ranges.pop_front();

      // Merge the range with the constraint.
      FailureOr<bool> result = merge(
          constraints[range.idx], range.allocas,
          range.op.getType().getAsRange().alignment(), inserted[range.idx]);

      // If the merge failed, return failure.
      if (failed(result))
        return range.op.emitError() << "failed to merge range";

      // If the merge succeeded, update the value to constraint index.
      if (*result) {
        changed = true;
        maxIt = ranges.size();
        for (Value alloc : range.allocas)
          valueToConstraintIdx[alloc] = range.idx;
        continue;
      }

      // If the merge couldn't be done, push the range back to the queue.
      ranges.push_back(range);
    }
    // If there are no more ranges to merge, break.
    if (ranges.empty())
      break;

    // If there are no changes made, fail as the ranges are not mergeable.
    if (!changed)
      return topOp->emitError() << "range constraints failed to converge";
  }
  return success();
}

LogicalResult RangeConstraintAnalysisImpl::checkAlignment(RangeInfo &range) {
  int32_t alignCtr = range.op.getType().getAsRange().alignment();
  if (alignCtr <= 1)
    return success();

  RangeConstraint &constraint = constraints[range.idx];

  // Get the leading allocation (first alloca in the range).
  Value leadingAlloc = range.allocas.front();

  // Find the position of the leading allocation in the constraint.
  ArrayRef<Value> allocations = constraint.allocations;

  auto it = llvm::find(allocations, leadingAlloc);
  if (it == allocations.end())
    return range.op.emitError() << "leading allocation not found in constraint";
  ptrdiff_t pos = std::distance(allocations.begin(), it);

  // If we are checking the leading allocation, check the alignment constraint
  // is compatible.
  if (pos == 0 && constraint.alignment % alignCtr == 0)
    return success();

  // Check the alignment of the alloca itself is satisfiable.
  if (pos % alignCtr == 0)
    return success();

  return range.op.emitError()
         << "Unsatisfiable alignment constraint from: " << range.op.getResult();
}

//===----------------------------------------------------------------------===//
// RangeConstraintAnalysis
//===----------------------------------------------------------------------===//

FailureOr<RangeConstraintAnalysis>
RangeConstraintAnalysis::create(Operation *topOp) {
  if (!topOp)
    return failure();
  RangeConstraintAnalysis analysis;
  RangeConstraintAnalysisImpl impl(analysis.constraints);
  if (failed(impl.run(topOp)))
    return failure();

  // Sort the constraints by the number of allocations and alignment, so that
  // the constraints with more allocations and stricter alignment are allocated
  // first.
  llvm::stable_sort(analysis.constraints, [](const RangeConstraint &lhs,
                                             const RangeConstraint &rhs) {
    return std::make_pair(lhs.allocations.size(), lhs.alignment) >
           std::make_pair(rhs.allocations.size(), rhs.alignment);
  });
  return analysis;
}
