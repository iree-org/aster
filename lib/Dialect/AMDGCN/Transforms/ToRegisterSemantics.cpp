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
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"

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
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Pattern to convert value allocas to unallocated allocas.
struct AllocaOpPattern : public OpRewritePattern<AllocaOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(AllocaOp op,
                                PatternRewriter &rewriter) const override;
};

/// Pattern to handle dealloc_cast operations.
struct DeallocCastOpPattern : public OpRewritePattern<DeallocCastOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(DeallocCastOp op,
                                PatternRewriter &rewriter) const override;
};

/// Pattern to update InstOpInterface operations.
struct InstOpPattern : public OpInterfaceRewritePattern<InstOpInterface> {
  using Base::Base;
  LogicalResult matchAndRewrite(InstOpInterface instOp,
                                PatternRewriter &rewriter) const override;
};

/// Pattern to handle make_register_range operations.
struct MakeRegisterRangePattern : public OpRewritePattern<MakeRegisterRangeOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(MakeRegisterRangeOp op,
                                PatternRewriter &rewriter) const override;
};

/// Pattern to replace SplitRegisterRangeOp with allocas.
struct SplitRegisterRangePattern
    : public OpRewritePattern<SplitRegisterRangeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(SplitRegisterRangeOp op,
                                PatternRewriter &rewriter) const override;
};

/// Pattern to convert a generic operation to update non-value allocas.
template <typename OpTy>
struct GenericOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

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

/// Gets the non-value allocation behind a value, using the following rules:
/// - If the value is not a register, it returns failure.
/// - If the value is an alloca, it returns the alloca if it has non-value
/// semantics.
/// - If value is a make register range it checks the inputs and if all the
/// inputs are allocas with non-value semantics, it returns itself.
/// - If value is an unrealized conversion cast operation, it returns the input
/// if it is an allocation and has non-value semantics.
/// - Otherwise, it returns failure.
/// If unpackRange is true, it returns the range of non-value allocations,
/// otherwise it returns the single non-value allocation.
static FailureOr<ValueRange> getNonValueAllocationsOrFailure(Value value,
                                                             bool unpackRange) {
  // If the value is not a register, return an empty range.
  if (!isa<RegisterTypeInterface>(value.getType()))
    return failure();

  // Handle alloca operations.
  if (auto allocaOp = value.getDefiningOp<AllocaOp>()) {
    if (allocaOp.getType().hasValueSemantics())
      return failure();
    return allocaOp->getResults();
  }

  // Handle make register range operations.
  if (auto rangeOp = value.getDefiningOp<MakeRegisterRangeOp>()) {
    for (Value input : rangeOp.getInputs()) {
      auto allocaOp = input.getDefiningOp<AllocaOp>();
      if (!allocaOp || allocaOp.getType().hasValueSemantics())
        return failure();
    }
    return unpackRange ? ValueRange(rangeOp.getInputs())
                       : ValueRange(rangeOp->getResults());
  }

  // Handle unrealized conversion cast operations.
  if (auto cOp = value.getDefiningOp<UnrealizedConversionCastOp>();
      cOp && cOp->getDiscardableAttr(kConversionTag)) {
    if (cOp.getInputs().size() != 1)
      return failure();
    return getNonValueAllocationsOrFailure(cOp.getInputs().front(),
                                           unpackRange);
  }

  // Fail in all other cases.
  return failure();
}

//===----------------------------------------------------------------------===//
// AllocaOpPattern implementation
//===----------------------------------------------------------------------===//

LogicalResult
AllocaOpPattern::matchAndRewrite(AllocaOp op, PatternRewriter &rewriter) const {
  // Only convert value allocas
  auto regTy = cast<RegisterTypeInterface>(op.getType());
  if (!regTy.hasValueSemantics())
    return rewriter.notifyMatchFailure(op,
                                       "alloca does not have value semantics");

  // Create a new alloca with unallocated semantics.
  AllocaOp newAlloca =
      AllocaOp::create(rewriter, op.getLoc(), regTy.getAsUnallocated());

  // Create a tagged cast to the original type.
  auto castOp = createTaggedCast(rewriter, op.getLoc(), op.getType(),
                                 newAlloca.getResult());

  // Replace the original alloca with the new value.
  rewriter.replaceOp(op, castOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// DeallocCastOpPattern implementation
//===----------------------------------------------------------------------===//

LogicalResult
DeallocCastOpPattern::matchAndRewrite(DeallocCastOp op,
                                      PatternRewriter &rewriter) const {
  // Get the non-value allocation.
  FailureOr<ValueRange> result =
      getNonValueAllocationsOrFailure(op.getInput(), /*unpackRange=*/false);
  if (failed(result) || result->size() != 1)
    return failure();

  // Create a tagged cast to the original type.
  auto castOp =
      createTaggedCast(rewriter, op.getLoc(), op.getType(), result->front());

  // Replace the original operation with the new value.
  rewriter.replaceOp(op, castOp.getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// InstOpPattern implementation
//===----------------------------------------------------------------------===//

/// Helper to handle an instruction operand.
static void handleInstOperand(Operation *op, Value &value,
                              PatternRewriter &rewriter, bool &changed) {
  auto rTy = dyn_cast<RegisterTypeInterface>(value.getType());

  // If the operand is not a register, skip.
  if (!rTy)
    return;

  // Get the non-value allocation.
  FailureOr<ValueRange> result =
      getNonValueAllocationsOrFailure(value, /*unpackRange=*/false);
  if (failed(result) || result->size() != 1)
    return;

  // If the values are the same, skip.
  if (result->front() == value)
    return;

  // Update the value.
  value = result->front();
  changed = true;
}

LogicalResult InstOpPattern::matchAndRewrite(InstOpInterface instOp,
                                             PatternRewriter &rewriter) const {
  // Try getting the new operands.
  bool changed = false;
  SmallVector<Value, 4> outs = llvm::to_vector(instOp.getInstOuts()),
                        ins = llvm::to_vector(instOp.getInstIns());
  for (Value &operand : outs)
    handleInstOperand(instOp, operand, rewriter, changed);
  for (Value &operand : ins)
    handleInstOperand(instOp, operand, rewriter, changed);

  // Determine if we need to propagate outs to results.
  bool requiresResultsUpdate = false;
  for (Value result : instOp.getInstResults()) {
    auto regTy = dyn_cast<RegisterTypeInterface>(result.getType());
    if (!regTy || regTy.hasValueSemantics())
      continue;
    // If the result has uses, we need to propagate the outs to it.
    requiresResultsUpdate |= !result.getUses().empty();
  }

  // If no changes were made, return failure.
  if (!changed && !requiresResultsUpdate)
    return rewriter.notifyMatchFailure(instOp, "no changes can be made");

  // Clone the instruction with the updated operands.
  auto newInst = instOp.cloneInst(rewriter, outs, ins);

  SmallVector<Value, 4> newResults =
      llvm::to_vector_of<Value, 4>(newInst->getResults());

  // Forward the output operands to the new results.
  int64_t resPos = 0;
  if (ResultRange range = instOp.getInstResults(); !range.empty())
    resPos = range.front().getResultNumber();

  // Propagate non-value outs to results.
  for (Value out : newInst.getInstOuts()) {
    auto regTy = dyn_cast<RegisterTypeInterface>(out.getType());
    if (regTy.hasValueSemantics())
      continue;
    newResults[resPos++] = out;
  }

  // Create tagged casts for all results if the types don't match.
  for (auto &&[result, oldResult] :
       llvm::zip_equal(newResults, instOp->getResults())) {
    if (result.getType() == oldResult.getType())
      continue;

    // Create a tagged cast to the expected type.
    auto castOp = createTaggedCast(rewriter, result.getLoc(),
                                   oldResult.getType(), result);
    result = castOp.getResult(0);
  }

  // Replace the original instruction with the new results.
  rewriter.replaceOp(instOp, newResults);
  return success();
}

//===----------------------------------------------------------------------===//
// MakeRegisterRangePattern implementation
//===----------------------------------------------------------------------===//

LogicalResult
MakeRegisterRangePattern::matchAndRewrite(MakeRegisterRangeOp op,
                                          PatternRewriter &rewriter) const {
  // Collect non-value allocas from all inputs.
  SmallVector<Value, 4> allocas;
  int32_t numUpdates = 0;
  for (Value input : op.getInputs()) {
    // Get the non-value allocation.
    FailureOr<ValueRange> result =
        getNonValueAllocationsOrFailure(input, /*unpackRange=*/false);
    if (failed(result))
      return failure();
    assert(result->size() == 1 && "expected a single allocation");

    // Count the number of updates.
    numUpdates += (*result != ValueRange(input)) ? result->size() : 0;
    llvm::append_range(allocas, *result);
  }

  // If no updates can be made, or if the number of updates is not the same as
  // the number of inputs, return failure.
  if (numUpdates != static_cast<int32_t>(op.getInputs().size()))
    return failure();

  // Create new MakeRegisterRangeOp and wrap with a tagged cast.
  MakeRegisterRangeOp newOp =
      MakeRegisterRangeOp::create(rewriter, op.getLoc(), allocas);
  auto castOp =
      createTaggedCast(rewriter, op.getLoc(), op.getType(), newOp.getResult());

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

  // Get the non-value allocations.
  SmallVector<Value, 4> allocas;

  FailureOr<ValueRange> result =
      getNonValueAllocationsOrFailure(input, /*unpackRange=*/true);
  if (failed(result) || result->size() != op.getResults().size())
    return failure();
  llvm::append_range(allocas, *result);

  // Create tagged cast for each result if the types don't match.
  for (auto &&[alloca, result] : llvm::zip_equal(allocas, op.getResults())) {
    auto castOp =
        createTaggedCast(rewriter, alloca.getLoc(), result.getType(), alloca);
    alloca = castOp.getResult(0);
  }

  // Replace the original operation with the new results.
  rewriter.replaceOp(op, allocas);
  return success();
}

//===----------------------------------------------------------------------===//
// GenericOpPattern implementation
//===----------------------------------------------------------------------===//

template <typename OpTy>
LogicalResult
GenericOpPattern<OpTy>::matchAndRewrite(OpTy op,
                                        PatternRewriter &rewriter) const {
  bool changed = false;
  for (OpOperand &operand : op->getOpOperands()) {
    FailureOr<ValueRange> allocs =
        getNonValueAllocationsOrFailure(operand.get(), /*unpackRange=*/false);
    if (failed(allocs) || allocs->size() != 1)
      continue;
    if (allocs->front() == operand.get())
      continue;
    operand.set(allocs->front());
    changed = true;
  }
  if (changed)
    rewriter.modifyOpInPlace(op, []() {});
  return success(changed);
}

//===----------------------------------------------------------------------===//
// ToRegisterSemantics pass implementation
//===----------------------------------------------------------------------===//

void ToRegisterSemantics::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *ctx = op->getContext();
  RewritePatternSet patterns(ctx);
  patterns
      .add<AllocaOpPattern, InstOpPattern, MakeRegisterRangePattern,
           SplitRegisterRangePattern, GenericOpPattern<lsir::CmpIOp>,
           GenericOpPattern<amdgcn::RegInterferenceOp>, DeallocCastOpPattern>(
          ctx);
  if (failed(applyPatternsAndFoldGreedily(
          op, std::move(patterns),
          GreedyRewriteConfig()
              .setUseTopDownTraversal(true)
              .setRegionSimplificationLevel(
                  GreedySimplifyRegionLevel::Disabled)))) {
    op->emitError("failed to apply register semantics patterns");
    return signalPassFailure();
  }
}
