//===- ResolveAnyIterArgs.cpp - Resolve any-typed iter_args ---------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Specializes !aster_utils.any-typed values to their concrete types by locally
// analyzing the to_any/from_any chain. Handles:
//   - scf.for iter_args (including passthrough rotation patterns)
//   - cf block arguments passed via cf.br/cf.cond_br
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AsterUtils/IR/AsterUtilsDialect.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsOps.h"
#include "aster/Dialect/AsterUtils/IR/AsterUtilsTypes.h"
#include "aster/Dialect/AsterUtils/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

namespace mlir::aster {
namespace aster_utils {
#define GEN_PASS_DEF_RESOLVEANYITERARGS
#include "aster/Dialect/AsterUtils/Transforms/Passes.h.inc"
} // namespace aster_utils
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::aster_utils;

namespace {

//===----------------------------------------------------------------------===//
// Local type resolution
//===----------------------------------------------------------------------===//

/// Check that all uses of `val` are from_any ops producing type `type`.
/// Carves out yield ops from the check.
static bool allUsesAreFromAny(Value val, Type type) {
  return llvm::all_of(val.getUses(), [&](OpOperand &use) {
    if (isa<scf::YieldOp>(use.getOwner()))
      return true;
    auto fromAny = dyn_cast<FromAnyOp>(use.getOwner());
    return fromAny && fromAny.getResult().getType() == type;
  });
}

/// Validate passthrough consistency across resolved iter_args.
/// Fixpoint impl to gradually refine supported `resolvedTypes` and properly
/// capture rotation patterns. Entries that fail validation are reset to their
/// original type (meaning "no change").
static void
validatePassthroughConsistency(scf::ForOp forOp, scf::YieldOp yieldOp,
                               MutableArrayRef<Type> resolvedTypes) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (int64_t i = 0, n = resolvedTypes.size(); i < n; ++i) {
      Type origType = forOp.getRegionIterArg(i).getType();
      if (resolvedTypes[i] == origType)
        continue;
      Value yieldVal = yieldOp.getOperand(i);
      auto ba = dyn_cast<BlockArgument>(yieldVal);
      if (!ba || ba.getOwner() != forOp.getBody() ||
          ba == forOp.getInductionVar())
        continue;
      int64_t srcIdx = ba.getArgNumber() - 1;
      if (resolvedTypes[srcIdx] != resolvedTypes[i]) {
        resolvedTypes[i] = origType;
        changed = true;
      }
    }
  }
}

/// For each any-typed iter_arg, determine the concrete type from the
/// to_any / from_any chain. Returns a vector of types parallel to iter_args;
/// entries that equal the original type mean "no change".
static SmallVector<Type> resolveIterArgTypes(scf::ForOp forOp) {
  int64_t numIterArgs = forOp.getInitArgs().size();
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

  // Initialize with original types (meaning "no change").
  SmallVector<Type> resolvedTypes;
  resolvedTypes.reserve(numIterArgs);
  for (int64_t i = 0; i < numIterArgs; ++i)
    resolvedTypes.push_back(forOp.getRegionIterArg(i).getType());

  // 1. Determine candidate types from init_args and verify local uses.
  for (int64_t i = 0; i < numIterArgs; ++i) {
    Value blockArg = forOp.getRegionIterArg(i);
    if (!isa<AnyTypeType>(blockArg.getType()))
      continue;

    auto toAnyInit = forOp.getInitArgs()[i].getDefiningOp<ToAnyOp>();
    if (!toAnyInit)
      continue;

    // All block arg and result uses must be from_any (except yield).
    Type type = toAnyInit.getInput().getType();
    if (!allUsesAreFromAny(blockArg, type) ||
        !allUsesAreFromAny(forOp.getResult(i), type))
      continue;

    Value yieldVal = yieldOp.getOperand(i);
    // Yield a toAny is supported if `type` resolves identically.
    auto toAnyYield = yieldVal.getDefiningOp<ToAnyOp>();
    if (toAnyYield && toAnyYield.getInput().getType() != type)
      continue;
    // Directly yielding a BlockArgument: must be in the forOp body and not be
    // the induction variable.
    auto ba = dyn_cast<BlockArgument>(yieldVal);
    if (ba &&
        (ba.getOwner() != forOp.getBody() || ba == forOp.getInductionVar()))
      continue;
    // non-ToAny and non-BlockArgument are unsupported.
    if (!toAnyYield && !ba)
      continue;

    resolvedTypes[i] = type;
  }

  // 2. Validate passthrough consistency, including rotation patterns.
  // Reset resolvedType entries to null when not consistent.
  validatePassthroughConsistency(forOp, yieldOp, resolvedTypes);

  return resolvedTypes;
}

//===----------------------------------------------------------------------===//
// Generic ForOp iter_arg type replacement
//===----------------------------------------------------------------------===//

/// Callback: configure body cloning for iter_arg `idx` whose type changed.
/// Add entries to `mapping` (e.g., map unwrap op results to the new block arg)
/// and `skipOps` (ops to omit during cloning).
using BodySetupFn =
    function_ref<void(int64_t idx, Value oldBlockArg, Value newBlockArg,
                      IRMapping &mapping, DenseSet<Operation *> &skipOps)>;

/// Callback: produce the new yield operand for iter_arg `idx` whose type
/// changed. Return null to use mapping.lookupOrDefault on the original yield
/// value (passthrough).
using YieldSetupFn =
    function_ref<Value(int64_t idx, Value oldYieldVal, IRMapping &mapping)>;

/// Callback: replace uses of the old result for iter_arg `idx` whose type
/// changed.
using ResultReplaceFn =
    function_ref<void(int64_t idx, Value oldResult, Value newResult)>;

/// Replace an scf.for with new iter_arg types.
/// Uses clone + IRMapping (not mergeBlocks) so that callers can selectively
/// skip ops during body migration.
///
/// For each position i where newTypes[i] differs from the original iter_arg
/// type, the three callbacks are invoked. Positions where the type is unchanged
/// are carried through as-is.
static LogicalResult replaceForOpIterArgTypes(scf::ForOp forOp,
                                              ArrayRef<Type> newTypes,
                                              ArrayRef<Value> newInitArgs,
                                              BodySetupFn bodySetup,
                                              YieldSetupFn getNewYieldValue,
                                              ResultReplaceFn replaceResult) {
  int64_t numIterArgs = forOp.getInitArgs().size();
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  auto newForOp =
      scf::ForOp::create(builder, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), newInitArgs);

  Block *newBody = newForOp.getBody();
  Block *oldBody = forOp.getBody();
  if (newBody->mightHaveTerminator())
    newBody->getTerminator()->erase();

  builder.setInsertionPointToStart(newBody);
  IRMapping mapping;
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));

  // Map block args. For changed iter_args, let the caller configure additional
  // mappings and mark ops to skip.
  DenseSet<Operation *> skipOps;
  for (int64_t i = 0; i < numIterArgs; ++i) {
    Value oldBlockArg = oldBody->getArgument(i + 1);
    Value newBlockArg = newBody->getArgument(i + 1);
    mapping.map(oldBlockArg, newBlockArg);
    if (newTypes[i] != oldBlockArg.getType())
      bodySetup(i, oldBlockArg, newBlockArg, mapping, skipOps);
  }

  for (auto &op : oldBody->without_terminator()) {
    if (skipOps.contains(&op))
      continue;
    builder.clone(op, mapping);
  }

  // Build new yield.
  SmallVector<Value> newYieldOperands;
  for (int64_t i = 0; i < numIterArgs; ++i) {
    Value yieldVal = yieldOp.getOperand(i);
    if (newTypes[i] != oldBody->getArgument(i + 1).getType()) {
      Value newVal = getNewYieldValue(i, yieldVal, mapping);
      if (newVal) {
        newYieldOperands.push_back(newVal);
        continue;
      }
      // Null return: passthrough, use default mapping.
    }
    newYieldOperands.push_back(mapping.lookupOrDefault(yieldVal));
  }
  scf::YieldOp::create(builder, loc, newYieldOperands);

  // Replace uses of old results.
  for (int64_t i = 0; i < numIterArgs; ++i) {
    Value oldResult = forOp.getResult(i);
    Value newResult = newForOp.getResult(i);
    if (newTypes[i] != forOp.getRegionIterArg(i).getType()) {
      replaceResult(i, oldResult, newResult);
    } else {
      oldResult.replaceAllUsesWith(newResult);
    }
  }

  forOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ForOp any-type rewriting
//===----------------------------------------------------------------------===//

/// Rewrite a single scf.for op, specializing any-typed iter_args to their
/// concrete types by stripping to_any/from_any wrapper ops.
///
/// Example:
///   %init = to_any %x : i32                 %result = scf.for ...
///   %result = scf.for ...                     iter_args(%arg = %x) -> (i32) {
///     iter_args(%arg = %init) -> (!any) { =>    %sum = arith.addi %arg, %arg
///     %v = from_any %arg : i32                    scf.yield %sum
///     %sum = arith.addi %v, %v              }
///     scf.yield (to_any %sum : i32)
///   }
///   %out = from_any %result : i32
static LogicalResult rewriteForOp(scf::ForOp forOp) {
  SmallVector<Type> resolvedTypes = resolveIterArgTypes(forOp);

  // Check if any iter_arg type actually changed.
  int64_t numIterArgs = forOp.getInitArgs().size();
  if (!llvm::any_of(llvm::enumerate(resolvedTypes), [&](auto pair) {
        return pair.value() != forOp.getRegionIterArg(pair.index()).getType();
      }))
    return failure();

  // Compute new init args: unwrap to_any for changed iter_args.
  SmallVector<Value> newInitArgs;
  for (int64_t i = 0; i < numIterArgs; ++i) {
    if (resolvedTypes[i] != forOp.getRegionIterArg(i).getType()) {
      auto toAny = forOp.getInitArgs()[i].getDefiningOp<ToAnyOp>();
      assert(toAny && "expectedto_any init for resolved types");
      newInitArgs.push_back(toAny.getInput());
    } else {
      newInitArgs.push_back(forOp.getInitArgs()[i]);
    }
  }

  // Set up the generic replaceForOpIterArgTypes callbacks.
  return replaceForOpIterArgTypes(
      forOp, resolvedTypes, newInitArgs,
      /*bodySetup=*/
      [](int64_t, Value oldBlockArg, Value newBlockArg, IRMapping &mapping,
         DenseSet<Operation *> &skipOps) {
        // Skip from_any ops: map their results directly to the new
        // concrete-typed block arg.
        for (OpOperand &use : oldBlockArg.getUses()) {
          if (isa<scf::YieldOp>(use.getOwner()))
            continue;
          auto fromAny = cast<FromAnyOp>(use.getOwner());
          mapping.map(fromAny.getResult(), newBlockArg);
          skipOps.insert(fromAny);
        }
      },
      /*getNewYieldValue=*/
      [](int64_t, Value oldYieldVal, IRMapping &mapping) -> Value {
        // Unwrap to_any at yield; return null for passthrough.
        if (auto toAny = oldYieldVal.getDefiningOp<ToAnyOp>())
          return mapping.lookupOrDefault(toAny.getInput());
        return Value();
      },
      /*replaceResult=*/
      [](int64_t, Value oldResult, Value newResult) {
        // Bypass from_any users of the old result.
        for (OpOperand &use : llvm::make_early_inc_range(oldResult.getUses())) {
          auto fromAny = cast<FromAnyOp>(use.getOwner());
          fromAny.getResult().replaceAllUsesWith(newResult);
          fromAny.erase();
        }
      });
}

//===----------------------------------------------------------------------===//
// CF block argument resolution
//===----------------------------------------------------------------------===//

/// Callback for forEachPredecessorValue: receives the branch terminator, the
/// forwarded value, and its absolute operand index.
using PredecessorValueFn = function_ref<LogicalResult(
    BranchOpInterface terminator, Value predVal, int64_t operandIdx)>;

/// Call `fn` for each predecessor edge forwarding a value to block arg at
/// `argIdx`. Returns failure if any predecessor lacks BranchOpInterface,
/// doesn't forward enough operands, or `fn` returns failure.
static LogicalResult forEachPredecessorValue(Block *block, int64_t argIdx,
                                             PredecessorValueFn fn) {
  for (Block *pred : block->getPredecessors()) {
    auto terminator = dyn_cast<BranchOpInterface>(pred->getTerminator());
    if (!terminator)
      return failure();
    for (auto [i, succ] : llvm::enumerate(terminator->getSuccessors())) {
      if (succ != block)
        continue;
      OperandRange forwarded =
          terminator.getSuccessorOperands(i).getForwardedOperands();
      if (argIdx >= static_cast<int64_t>(forwarded.size()))
        return failure();
      int64_t operandIdx = forwarded.getBeginOperandIndex() + argIdx;
      if (failed(fn(terminator, forwarded[argIdx], operandIdx)))
        return failure();
    }
  }
  return success();
}

/// Determine the concrete type for an any-typed block argument by checking that
/// - all predecessors provide to_any of the same type, and
/// - all uses within the block are from_any of the same type (or passthrough).
static Type resolveBlockArgType(Block *block, int64_t argIdx) {
  BlockArgument blockArg = block->getArgument(argIdx);
  assert(isa<AnyTypeType>(blockArg.getType()) && "expected any-typed arg");

  // Determine candidate type from uses within this block.
  // Every non-terminator use must be from_any of the same type.
  // Terminator uses (branch forwarding to successors) are passthroughs subject
  // to a fixpoint iteration (for rotation patterns).
  Type candidateType;
  for (OpOperand &use : blockArg.getUses()) {
    Operation *owner = use.getOwner();
    if (owner->hasTrait<OpTrait::IsTerminator>())
      continue;
    auto fromAny = dyn_cast<FromAnyOp>(owner);
    if (!fromAny)
      return nullptr;
    Type useType = fromAny.getResult().getType();
    if (!candidateType) {
      candidateType = useType;
    } else if (candidateType != useType) {
      return nullptr;
    }
  }

  // Check all predecessors provide to_any of the same type for this arg.
  auto checkPredType = [&](Type predType) -> LogicalResult {
    if (!candidateType) {
      candidateType = predType;
      return success();
    }
    return success(candidateType == predType);
  };

  if (failed(forEachPredecessorValue(
          block, argIdx,
          [&](BranchOpInterface, Value predVal, int64_t) -> LogicalResult {
            if (auto toAny = predVal.getDefiningOp<ToAnyOp>())
              return checkPredType(toAny.getInput().getType());
            // Raw !any: unresolved passthrough, fixpoint may help later.
            if (isa<AnyTypeType>(predVal.getType()))
              return failure();
            // Concrete value from a previous fixpoint iteration.
            return checkPredType(predVal.getType());
          })))
    return nullptr;

  return candidateType;
}

/// Rewrite a single block argument from !aster_utils.any to its concrete type.
/// Updates the block arg type in-place, bypasses from_any users, and unwraps
/// to_any in predecessor branches. Returns failure if predecessor rewriting
/// fails (IR will be in a partially-modified state -- caller must signal pass
/// failure).
static LogicalResult rewriteBlockArg(Block *block, int64_t argIdx,
                                     Type concreteType) {
  BlockArgument blockArg = block->getArgument(argIdx);
  blockArg.setType(concreteType);

  // Bypass from_any users: replace from_any result with the now-concrete arg.
  for (OpOperand &use : llvm::make_early_inc_range(blockArg.getUses())) {
    auto fromAny = dyn_cast<FromAnyOp>(use.getOwner());
    if (!fromAny)
      continue;
    fromAny.getResult().replaceAllUsesWith(blockArg);
    fromAny.erase();
  }

  // Unwrap to_any in predecessor branches. Analysis should have verified all
  // predecessors are rewritable, so failure here indicates an internal error.
  return forEachPredecessorValue(
      block, argIdx,
      [](BranchOpInterface terminator, Value predVal,
         int64_t operandIdx) -> LogicalResult {
        if (auto toAny = predVal.getDefiningOp<ToAnyOp>())
          terminator->setOperand(operandIdx, toAny.getInput());
        return success();
      });
}

/// Single sweep over all non-entry blocks: resolve any-typed block args.
/// Returns failure if any rewrite leaves IR in a broken state. Sets
/// `anyChanged` to true if at least one block arg was rewritten.
static LogicalResult rewriteCFBlockArgs(Operation *op, bool &anyChanged) {
  LogicalResult result = success();
  op->walk([&](Block *block) {
    if (block->isEntryBlock())
      return;
    if (block->hasNoPredecessors())
      return;

    for (int64_t i = 0; i < block->getNumArguments(); ++i) {
      if (!isa<AnyTypeType>(block->getArgument(i).getType()))
        continue;
      Type resolved = resolveBlockArgType(block, i);
      if (!resolved)
        continue;
      if (failed(rewriteBlockArg(block, i, resolved))) {
        result = failure();
        return;
      }
      anyChanged = true;
    }
  });
  return result;
}

//===----------------------------------------------------------------------===//
// ResolveAnyIterArgs pass
//===----------------------------------------------------------------------===//

struct ResolveAnyIterArgs
    : public aster_utils::impl::ResolveAnyIterArgsBase<ResolveAnyIterArgs> {
public:
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void ResolveAnyIterArgs::runOnOperation() {
  // Iterate until fixpoint: rewriting may expose new resolvable patterns.
  // scf.for rewrites erase ops (must break + re-scan), CF rewrites are
  // in-place (full sweep is safe).
  bool changed = true;
  while (changed) {
    changed = false;

    // scf.for iter_args.
    SmallVector<scf::ForOp> worklist;
    getOperation()->walk([&](scf::ForOp forOp) {
      if (llvm::any_of(forOp.getInitArgs(), [](Value arg) {
            return isa<AnyTypeType>(arg.getType());
          }))
        worklist.push_back(forOp);
    });
    for (auto forOp : worklist) {
      if (succeeded(rewriteForOp(forOp))) {
        changed = true;
        break; // Re-scan after ForOp erasure.
      }
    }
    if (changed)
      continue;

    // cf block arguments.
    if (failed(rewriteCFBlockArgs(getOperation(), changed)))
      return signalPassFailure();
  }
}
