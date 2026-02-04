//===- ToRegisterSemantics.cpp ------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass converts value allocas to unallocated register semantics and
// updates InstOpInterface operations to use make_register_range.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/AllocaAliasAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-to-register-semantics"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_TOREGISTERSEMANTICS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
static constexpr std::string_view kConversionTag = "__to_register_semantics__";
//===----------------------------------------------------------------------===//
// ToRegisterSemantics pass
//===----------------------------------------------------------------------===//

struct ToRegisterSemantics
    : public amdgcn::impl::ToRegisterSemanticsBase<ToRegisterSemantics> {
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// AliasTable
//===----------------------------------------------------------------------===//

/// Alias table helper class.
struct AliasTable {
  explicit AliasTable(AllocaAliasAnalysis &analysis);

  /// Looks up the alloca IDs for this value. Results are cached.
  ArrayRef<AllocaID> lookup(Value value);

  /// Populates the allocas associated with value.
  void getAllocas(Value value, SmallVectorImpl<Value> &allocas);

  /// Remaps the aliases of `from` to make them the aliases of `to`.
  ArrayRef<AllocaID> remap(Value from, Value to);

  /// Remaps the allocations.
  ArrayRef<AllocaID> remapAlloca(AllocaOp from, AllocaOp to);

  /// Maps a value to a list of alloca IDs.
  void map(Value value, ArrayRef<AllocaID> allocaIds);

private:
  /// Looks up and caches the alloca IDs for a value, returning an
  /// iterator to the cached entry. Returns end() if no alloca IDs
  /// exist.
  DenseMap<Value, ArrayRef<AllocaID>>::iterator lookupImpl(Value value);

  AllocaAliasAnalysis *allocaAliasAnalysis;
  DenseMap<Value, ArrayRef<AllocaID>> valueToIds;
  SmallVector<AllocaOp> idsToValuesMap;
};

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert value allocas to unallocated allocas.
struct AllocaOpPattern : public OpRewritePattern<AllocaOp> {
  AllocaOpPattern(MLIRContext *ctx, AliasTable &allocTable)
      : OpRewritePattern<AllocaOp>(ctx), allocTable(allocTable) {}

  LogicalResult matchAndRewrite(AllocaOp op,
                                PatternRewriter &rewriter) const override;

private:
  AliasTable &allocTable;
};

/// Pattern to update InstOpInterface operations to use make_register_range.
struct InstOpPattern : public OpInterfaceRewritePattern<InstOpInterface> {
  InstOpPattern(MLIRContext *ctx, AliasTable &allocTable)
      : OpInterfaceRewritePattern<InstOpInterface>(ctx),
        allocTable(allocTable) {}

  LogicalResult matchAndRewrite(InstOpInterface instOp,
                                PatternRewriter &rewriter) const override;

private:
  AliasTable &allocTable;
};

/// Pattern to convert MakeRegisterRangeOp to use allocas from
/// AllocaAliasAnalysis.
struct MakeRegisterRangePass : public OpRewritePattern<MakeRegisterRangeOp> {
  MakeRegisterRangePass(MLIRContext *ctx, AliasTable &allocTable)
      : OpRewritePattern<MakeRegisterRangeOp>(ctx), allocTable(allocTable) {}

  LogicalResult matchAndRewrite(MakeRegisterRangeOp op,
                                PatternRewriter &rewriter) const override;

private:
  AliasTable &allocTable;
};

/// Pattern to replace SplitRegisterRangeOp with allocas from
/// AllocaAliasAnalysis.
struct SplitRegisterRangePattern
    : public OpRewritePattern<SplitRegisterRangeOp> {
  SplitRegisterRangePattern(MLIRContext *ctx, AliasTable &allocTable)
      : OpRewritePattern<SplitRegisterRangeOp>(ctx), allocTable(allocTable) {}

  LogicalResult matchAndRewrite(SplitRegisterRangeOp op,
                                PatternRewriter &rewriter) const override;

private:
  AliasTable &allocTable;
};

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//

/// Creates an UnrealizedConversionCastOp with the conversion tag.
static UnrealizedConversionCastOp createTaggedCast(PatternRewriter &rewriter,
                                                   Location loc, Type type,
                                                   Value input) {
  auto castOp = UnrealizedConversionCastOp::create(rewriter, loc, type, input);
  castOp->setDiscardableAttr(kConversionTag, rewriter.getUnitAttr());
  return castOp;
}

/// Validates that all allocas have unallocated semantics.
static LogicalResult validateUnallocatedAllocas(ArrayRef<Value> allocas,
                                                Operation *op,
                                                PatternRewriter &rewriter) {
  if (allocas.empty())
    return rewriter.notifyMatchFailure(op, "no allocations found");

  for (Value alloca : allocas) {
    auto rTy = cast<RegisterTypeInterface>(alloca.getType());
    if (!rTy.hasUnallocatedSemantics()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected an unallocated allocation");
    }
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// AliasTable implementation
//===----------------------------------------------------------------------===//

AliasTable::AliasTable(AllocaAliasAnalysis &analysis)
    : allocaAliasAnalysis(&analysis) {
  idsToValuesMap = llvm::map_to_vector(analysis.getAllocas(), [](Value value) {
    auto allocaOp = value.getDefiningOp<AllocaOp>();
    assert(allocaOp && "expected an alloca op");
    return allocaOp;
  });
}

DenseMap<Value, ArrayRef<AllocaID>>::iterator
AliasTable::lookupImpl(Value value) {
  // Check if already cached in valueToIds map
  auto it = valueToIds.find(value);
  if (it != valueToIds.end())
    return it;

  // Query the analysis
  ArrayRef<AllocaID> allocaIds = allocaAliasAnalysis->getAllocaIds(value);

  // If no alloca IDs, return end iterator
  if (allocaIds.empty())
    return valueToIds.end();

  // Cache the ArrayRef in the map (it points to data owned by the analysis)
  return valueToIds.insert({value, allocaIds}).first;
}

ArrayRef<AllocaID> AliasTable::lookup(Value value) {
  auto it = lookupImpl(value);
  return it != valueToIds.end() ? it->second : ArrayRef<AllocaID>();
}

void AliasTable::getAllocas(Value value, SmallVectorImpl<Value> &allocas) {
  // Use lookup to get the alloca IDs
  ArrayRef<AllocaID> allocaIds = lookup(value);

  // Convert each alloca ID to its corresponding alloca
  allocas.reserve(allocas.size() + allocaIds.size());
  for (AllocaID allocaId : allocaIds)
    allocas.push_back(idsToValuesMap[allocaId]);
}

ArrayRef<AllocaID> AliasTable::remap(Value from, Value to) {
  auto it = lookupImpl(from);
  if (it == valueToIds.end())
    return {};
  LDBG() << "Remapping " << from << " to " << to;
  ArrayRef<AllocaID> ids = it->second;
  valueToIds.erase(it);
  valueToIds.insert_or_assign(to, ids);
  return ids;
}

ArrayRef<AllocaID> AliasTable::remapAlloca(AllocaOp from, AllocaOp to) {
  ArrayRef<AllocaID> ids = remap(from, to);
  idsToValuesMap[ids.front()] = to;
  return ids;
}

void AliasTable::map(Value value, ArrayRef<AllocaID> allocaIds) {
  valueToIds.insert_or_assign(value, allocaIds);
}

//===----------------------------------------------------------------------===//
// AllocaOpPattern implementation
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpPattern::matchAndRewrite(AllocaOp op, PatternRewriter &rewriter) const {
  // Only convert value allocas
  auto regTy = cast<RegisterTypeInterface>(op.getType());
  if (!regTy.hasValueSemantics())
    return rewriter.notifyMatchFailure(op, "expected a value");

  // Create a new alloca with unallocated semantics.
  AllocaOp newAlloca =
      AllocaOp::create(rewriter, op.getLoc(), regTy.getAsUnallocated());

  // Create a tagged cast to the original type.
  auto castOp = createTaggedCast(rewriter, op.getLoc(), op.getType(),
                                 newAlloca.getResult());

  // Update the alloc table with the new alloca.
  allocTable.map(castOp.getResult(0), allocTable.remapAlloca(op, newAlloca));

  // Replace the original alloca with the new value.
  rewriter.replaceOp(op, castOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// InstOpPattern implementation
//===----------------------------------------------------------------------===//

/// Helper to handle an instruction operand.
static void handleInstOperand(Operation *op, Value &value,
                              PatternRewriter &rewriter, AliasTable &allocTable,
                              bool &changed) {
  auto rTy = dyn_cast<RegisterTypeInterface>(value.getType());

  // If the operand is not a value register, skip.
  if (!rTy || !rTy.hasValueSemantics())
    return;

  // Check if the operand is a conversion cast - unwrap it
  if (auto cOp = value.getDefiningOp<UnrealizedConversionCastOp>();
      cOp && cOp->getDiscardableAttr(kConversionTag)) {
    ValueRange ins = cOp.getInputs();
    assert(ins.size() == 1 && "ill-formed conversion");
    if (isa_and_nonnull<AllocaOp, MakeRegisterRangeOp>(
            ins[0].getDefiningOp())) {
      if (ins[0].getType() != value.getType()) {
        value = ins[0];
        changed = true;
      }
      return;
    }
  }

  // Get the allocas for the operand and create register range if needed
  SmallVector<Value, 4> allocas;
  allocTable.getAllocas(value, allocas);

  // Verify all allocas are unallocated
  if (failed(validateUnallocatedAllocas(allocas, op, rewriter)))
    return;

  // Get the new value
  Value newValue;
  if (allocas.size() > 1) {
    newValue = rewriter.create<MakeRegisterRangeOp>(value.getLoc(), allocas);
    changed = true;
  } else {
    newValue = allocas[0];
  }
  if (newValue.getType() == value.getType())
    return;

  value = newValue;
  changed = true;
}

LogicalResult InstOpPattern::matchAndRewrite(InstOpInterface instOp,
                                             PatternRewriter &rewriter) const {
  // Try updating the operands in place.
  bool changed = false;
  SmallVector<Value, 4> outs = llvm::to_vector(instOp.getInstOuts()),
                        ins = llvm::to_vector(instOp.getInstIns());
  for (Value &operand : outs)
    handleInstOperand(instOp, operand, rewriter, allocTable, changed);
  for (Value &operand : ins)
    handleInstOperand(instOp, operand, rewriter, allocTable, changed);

  // If no changes were made, return failure.
  if (!changed)
    return failure();

  // Clone the instruction with the updated operands.
  auto newInst = instOp.cloneInst(rewriter, outs, ins, std::nullopt);

  // Create tagged casts for all results
  SmallVector<Value, 4> newResults;
  for (auto [result, oldResult] :
       llvm::zip_equal(newInst->getResults(), instOp->getResults())) {
    if (result.getType() == oldResult.getType()) {
      newResults.push_back(result);
      continue;
    }

    auto castOp = createTaggedCast(rewriter, result.getLoc(),
                                   oldResult.getType(), result);
    newResults.push_back(castOp.getResult(0));
    // Update the alloc table with the new result.
    allocTable.map(castOp.getResult(0), allocTable.remap(oldResult, result));
  }

  rewriter.replaceOp(instOp, newResults);
  return success();
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangePass implementation
//===----------------------------------------------------------------------===//

LogicalResult
MakeRegisterRangePass::matchAndRewrite(MakeRegisterRangeOp op,
                                       PatternRewriter &rewriter) const {
  // Get the result type
  auto regTy = cast<RegisterTypeInterface>(op.getType());

  // Check if this is a value register that needs to be converted
  if (!regTy.hasValueSemantics())
    return rewriter.notifyMatchFailure(op, "expected a value register");

  // Collect allocas from all inputs using AliasTable
  SmallVector<Value, 4> allocas;
  for (Value input : op.getInputs())
    allocTable.getAllocas(input, allocas);

  // Verify all allocas are unallocated
  if (failed(validateUnallocatedAllocas(allocas, op, rewriter)))
    return failure();

  // Create new MakeRegisterRangeOp and wrap with tagged cast
  MakeRegisterRangeOp newOp =
      MakeRegisterRangeOp::create(rewriter, op.getLoc(), allocas);
  auto castOp =
      createTaggedCast(rewriter, op.getLoc(), op.getType(), newOp.getResult());

  // Update the alloc table with the new operations.
  allocTable.map(castOp.getResult(0), allocTable.remap(op, newOp));

  // Replace the original operation with the new operation.
  rewriter.replaceOp(op, castOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// SplitRegisterRangePattern implementation
//===----------------------------------------------------------------------===//

LogicalResult
SplitRegisterRangePattern::matchAndRewrite(SplitRegisterRangeOp op,
                                           PatternRewriter &rewriter) const {
  Value input = op.getInput();

  // Check if the input is a value register that needs to be converted
  auto regTy = cast<RegisterTypeInterface>(input.getType());
  if (!regTy.hasValueSemantics())
    return rewriter.notifyMatchFailure(op, "expected a value");

  // Get the allocations using AliasTable
  SmallVector<Value, 4> allocas;
  allocTable.getAllocas(input, allocas);

  // Verify all allocas are unallocated
  if (failed(validateUnallocatedAllocas(allocas, op, rewriter)))
    return failure();

  // Create tagged cast for each result
  for (auto &&[alloca, result] : llvm::zip_equal(allocas, op.getResults())) {
    auto castOp =
        createTaggedCast(rewriter, alloca.getLoc(), result.getType(), alloca);
    (void)allocTable.remap(result, castOp.getResult(0));
    alloca = castOp.getResult(0);
  }
  rewriter.replaceOp(op, allocas);
  return success();
}

//===----------------------------------------------------------------------===//
// ToRegisterSemantics pass implementation
//===----------------------------------------------------------------------===//

void ToRegisterSemantics::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();

  // Step 1: Configure dataflow with AllocaAliasAnalysis
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  solver.load<dataflow::DeadCodeAnalysis>();
  auto *allocaAliasAnalysis = solver.load<AllocaAliasAnalysis>();

  if (failed(solver.initializeAndRun(op))) {
    op->emitError("Failed to run alloca alias analysis");
    return signalPassFailure();
  }

  if (allocaAliasAnalysis->isIllFormedIR()) {
    op->emitError("Alloca alias analysis is ill-formed");
    return signalPassFailure();
  }

  // Step 2: Create AliasTable for caching and updating aliases
  AliasTable allocTable(*allocaAliasAnalysis);

  // Step 3: Apply greedy pattern rewriting
  RewritePatternSet patterns(ctx);
  patterns.add<AllocaOpPattern, InstOpPattern, MakeRegisterRangePass,
               SplitRegisterRangePattern>(ctx, allocTable);

  if (failed(applyPatternsAndFoldGreedily(
          op, std::move(patterns),
          GreedyRewriteConfig().setUseTopDownTraversal(true)))) {
    op->emitError("Failed to apply register semantics patterns");
    return signalPassFailure();
  }
}
