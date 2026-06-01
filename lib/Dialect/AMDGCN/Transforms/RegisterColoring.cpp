//===- RegisterColoring.cpp ----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "GraphColoring.h"
#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/AMDGCN/Transforms/Transforms.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-register-coloring"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_REGISTERCOLORING
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
static constexpr std::string_view kCastOpTag = "__amdgcn_register_coloring__";
//===----------------------------------------------------------------------===//
// RegisterColoring pass
//===----------------------------------------------------------------------===//
struct RegisterColoring
    : public amdgcn::impl::RegisterColoringBase<RegisterColoring> {
public:
  using Base::Base;
  void runOnOperation() override;

  /// Run the transformation on the given function.
  LogicalResult run(FunctionOpInterface funcOp);
};

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//
struct InstRewritePattern : public OpInterfaceRewritePattern<InstOpInterface> {
  using Base::Base;

  LogicalResult matchAndRewrite(InstOpInterface op,
                                PatternRewriter &rewriter) const override;
};

struct MakeRegisterRangeOpPattern
    : public OpRewritePattern<MakeRegisterRangeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MakeRegisterRangeOp op,
                                PatternRewriter &rewriter) const override;
};

struct RegInterferenceOpPattern : public OpRewritePattern<RegInterferenceOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(RegInterferenceOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

LogicalResult
InstRewritePattern::matchAndRewrite(InstOpInterface op,
                                    PatternRewriter &rewriter) const {
  if (!op.getInstResults().empty())
    return rewriter.notifyMatchFailure(
        op, "expected instruction with register semantics");

  auto handleOperand = [&](Value operand) -> Value {
    auto cOp =
        dyn_cast_or_null<UnrealizedConversionCastOp>(operand.getDefiningOp());
    if (!cOp || !cOp->getDiscardableAttr(kCastOpTag))
      return nullptr;
    return cOp.getInputs().front();
  };

  bool mutated = false;
  SmallVector<Value> newOuts = llvm::to_vector(op.getInstOuts());
  SmallVector<Value> newIns = llvm::to_vector(op.getInstIns());
  for (Value &v : newOuts)
    if (Value nV = handleOperand(v)) {
      v = nV;
      mutated = true;
    }
  for (Value &v : newIns)
    if (Value nV = handleOperand(v)) {
      v = nV;
      mutated = true;
    }
  if (!mutated)
    return failure();

  // Create the new instruction.
  InstOpInterface newInst = op.cloneInst(rewriter, newOuts, newIns);
  if (!newInst)
    return failure();

  // Replace the original instruction with the new results.
  rewriter.replaceOp(op, newInst->getResults());
  return success();
}

LogicalResult
MakeRegisterRangeOpPattern::matchAndRewrite(MakeRegisterRangeOp op,
                                            PatternRewriter &rewriter) const {
  SmallVector<Value> ins;
  for (Value v : op.getInputs()) {
    auto cOp = dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp());
    if (!cOp || !cOp->getDiscardableAttr(kCastOpTag))
      return failure();
    ins.push_back(cOp.getInputs().front());
  }
  auto newOp = MakeRegisterRangeOp::create(rewriter, op.getLoc(), ins);
  auto cOp = rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op.getType(), newOp.getResult());
  cOp->setDiscardableAttr(kCastOpTag, rewriter.getUnitAttr());
  return success();
}

LogicalResult
RegInterferenceOpPattern::matchAndRewrite(RegInterferenceOp op,
                                          PatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Phase 2 helpers
//===----------------------------------------------------------------------===//

/// Emit one AllocaOp and a tagged cast for value, using allocatedType as the
/// physical register type, then replace all uses of value with the cast result.
/// Advances ip past the new AllocaOp so subsequent calls insert in IR order.
static void emitAllocAndReplace(Value value,
                                AMDGCNRegisterTypeInterface allocatedType,
                                IRRewriter &rewriter,
                                OpBuilder::InsertPoint &ip) {
  assert(value.getDefiningOp() && isa<AllocaOp>(value.getDefiningOp()) &&
         "expected alloca op for coalescing class member");
  auto allocaOp = cast<AllocaOp>(value.getDefiningOp());
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.restoreInsertionPoint(ip);

  AllocaOp newAlloca =
      AllocaOp::create(rewriter, allocaOp.getLoc(), allocatedType);

  // Cast back to the original unallocated type so existing uses remain valid
  // until the downstream rewrite patterns peel the cast.
  auto cOp = UnrealizedConversionCastOp::create(
      rewriter, allocaOp.getLoc(), value.getType(), newAlloca.getResult());
  cOp->setDiscardableAttr(kCastOpTag, rewriter.getUnitAttr());

  // Advance ip past the new AllocaOp so subsequent allocas are inserted in
  // textual IR order.
  rewriter.setInsertionPointAfter(newAlloca.getOperation());
  ip = rewriter.saveInsertionPoint();

  rewriter.replaceAllUsesWith(value, cOp.getResult(0));
}

/// Derive the allocated type for range position pos relative to the leader
/// type. The leader type carries physical register begin; position pos uses
/// begin+pos.
static AMDGCNRegisterTypeInterface
getOffsetType(AMDGCNRegisterTypeInterface leaderType, int32_t pos) {
  if (pos == 0)
    return leaderType;
  Register reg =
      leaderType.getAsRange().begin().getWithOffset(static_cast<int16_t>(pos));
  return cast<AMDGCNRegisterTypeInterface>(leaderType.cloneRegisterType(reg));
}

/// A pre-colored member at slot `pos` in its leader's range pins the range
/// base to (member.register - pos). Returns nullopt when the implied base is
/// negative (invalid position).
static std::optional<AMDGCNRegisterTypeInterface>
getPinnedRangeBase(AMDGCNRegisterTypeInterface fixedMemberTy, int32_t pos) {
  int32_t baseReg =
      static_cast<int32_t>(fixedMemberTy.getAsRange().begin().getRegister()) -
      pos;
  if (baseReg < 0)
    return std::nullopt;
  return cast<AMDGCNRegisterTypeInterface>(
      fixedMemberTy.cloneRegisterType(Register(static_cast<int16_t>(baseReg))));
}

/// Result of scanning a coalesced range for pre-colored members.
struct RangePinAnalysis {
  std::optional<AMDGCNRegisterTypeInterface> base;
  bool hasUnallocatedMember = false;
};

/// Scan all members of a coalesced range to find the implied physical base.
/// Returns failure on conflicting fixed registers within the same range.
static FailureOr<RangePinAnalysis>
analyzePinnedRange(int32_t leaderIdx, int32_t numRegs,
                   const RegisterInterferenceGraph &graph,
                   const CoalescingInfo &coalescingInfo, Location loc) {
  RangePinAnalysis result;
  for (int32_t pos = 0; pos < numRegs; ++pos) {
    int32_t qid = leaderIdx + pos;
    for (int32_t j = coalescingInfo.memberOffsets[qid],
                 end = coalescingInfo.memberOffsets[qid + 1];
         j < end; ++j) {
      auto memberTy = cast<AMDGCNRegisterTypeInterface>(
          graph.getValue(coalescingInfo.memberData[j]).getType());
      if (!memberTy.hasAllocatedSemantics()) {
        result.hasUnallocatedMember = true;
        continue;
      }
      auto candidate = getPinnedRangeBase(memberTy, pos);
      if (!candidate)
        return RangePinAnalysis{};
      if (result.base && *result.base != *candidate) {
        emitError(loc)
            << "coalesced class contains conflicting fixed registers";
        return failure();
      }
      result.base = *candidate;
    }
  }
  return result;
}

/// Pre-colored members in a coalesced class pin the leader's physical range.
/// Run before graph coloring so the colorer treats whole pinned ranges as
/// already-allocated.
static LogicalResult pinCoalescedClassesToFixedRegisters(
    int32_t numNodes, ArrayRef<NodeConstraint> nodeConsts,
    const RegisterInterferenceGraph &graph,
    const CoalescingInfo &coalescingInfo,
    SmallVectorImpl<AMDGCNRegisterTypeInterface> &types,
    llvm::SmallBitVector &wasUnallocated, Location loc) {
  for (int32_t i = 0; i < numNodes; ++i) {
    if (nodeConsts[i].numRegs == 0)
      continue;
    int32_t numRegs = nodeConsts[i].numRegs;

    FailureOr<RangePinAnalysis> pin =
        analyzePinnedRange(i, numRegs, graph, coalescingInfo, loc);
    if (failed(pin))
      return failure();
    if (!pin->base)
      continue;

    for (int32_t pos = 0; pos < numRegs; ++pos)
      types[i + pos] = getOffsetType(*pin->base, pos);
    if (pin->hasUnallocatedMember)
      wasUnallocated.set(static_cast<unsigned>(i));
  }
  return success();
}

/// Coalescing fan-out: expand each changed quotient node to all members of its
/// equivalence classes in the original interference graph.
static void rewriteCoalescing(RegisterInterferenceGraph &graph,
                              CoalescingInfo &coalescingInfo,
                              ArrayRef<NodeConstraint> nodeConsts,
                              ArrayRef<AMDGCNRegisterTypeInterface> types,
                              const llvm::SmallBitVector &wasUnallocated,
                              IRRewriter &rewriter,
                              OpBuilder::InsertPoint &ip) {
  int32_t m = static_cast<int32_t>(types.size());
  for (int32_t i = 0; i < m; ++i) {
    if (!wasUnallocated[static_cast<unsigned>(i)])
      continue;
    int32_t numRegs = nodeConsts[i].numRegs;
    // Non-leader range positions in the quotient graph (numRegs == 0) are
    // handled by their leader's fan-out loop; skip them here.
    if (numRegs == 0)
      continue;

    for (int32_t pos = 0; pos < numRegs; ++pos) {
      // Members for quotient slot i+pos are stored in memberData at the
      // slice [memberOffsets[i+pos], memberOffsets[i+pos+1]).
      int32_t qid = i + pos;
      AMDGCNRegisterTypeInterface allocatedType = getOffsetType(types[i], pos);
      for (int32_t j = coalescingInfo.memberOffsets[qid],
                   end = coalescingInfo.memberOffsets[qid + 1];
           j < end; ++j)
        emitAllocAndReplace(graph.getValue(coalescingInfo.memberData[j]),
                            allocatedType, rewriter, ip);
    }
  }
}

//===----------------------------------------------------------------------===//
// RegisterColoring pass
//===----------------------------------------------------------------------===//

LogicalResult RegisterColoring::run(FunctionOpInterface funcOp) {
  // Parse build mode option.
  RegisterInterferenceGraph::BuildMode buildMode;
  if (this->buildMode == "full") {
    buildMode = RegisterInterferenceGraph::BuildMode::Full;
  } else if (this->buildMode == "minimal") {
    buildMode = RegisterInterferenceGraph::BuildMode::Minimal;
  } else {
    return funcOp.emitError()
           << "build-mode must be \"full\" or \"minimal\", got \""
           << this->buildMode << "\"";
  }

  // Create the range constraint analysis.
  FailureOr<RangeConstraintAnalysis> rangeAnalysis =
      RangeConstraintAnalysis::create(funcOp);
  if (failed(rangeAnalysis))
    return funcOp.emitError() << "failed to run range constraint analysis";

  // Create the dataflow solver but don't run an analysis.
  // The coalescing-priority query in OptimizeInterferenceGraph::collectMov is
  // answered on demand via hasReachingLoadDefinition (per-MOV backwards walk),
  // eliminating the global dataflow fixpoint that previously dominated compile
  // time.
  SymbolTableCollection symbolTable;
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));

  // Create the interference graph.
  FailureOr<RegisterInterferenceGraph> graph =
      RegisterInterferenceGraph::create(funcOp, solver, symbolTable,
                                        *rangeAnalysis, buildMode);
  if (failed(graph))
    return funcOp.emitError() << "failed to create register interference graph";

  // Optimize the graph and get the coalescing classes.
  FailureOr<CoalescingInfo> coalescingInfo =
      CoalescingInfo::optimizeGraph(funcOp, *graph);
  if (failed(coalescingInfo))
    return funcOp.emitError() << "failed to optimize interference graph";

  //===--------------------------------------------------------------------===//
  // Phase 1: Materialize constraints and types; run colorGraph.
  //===--------------------------------------------------------------------===//

  ArrayRef<RangeConstraint *> qConsts = coalescingInfo->constraints;
  int32_t m = static_cast<int32_t>(qConsts.size());

  SmallVector<NodeConstraint> nodeConsts;
  SmallVector<AMDGCNRegisterTypeInterface> types;
  nodeConsts.reserve(m);
  types.reserve(m);
  llvm::SmallBitVector wasUnallocated(static_cast<unsigned>(m));

  // Track the end of the active range so that non-leader positions (those
  // falling inside a preceding leader's range) are marked with numRegs == 0.
  //
  // This relies on the invariant that, after compress(), the quotient IDs for
  // the non-leader positions of a range are consecutive and immediately
  // follow the leader's quotient ID. The invariant holds because:
  //  1. Ranges are constructed with consecutive original node IDs.
  //  2. Coalescing always merges the smaller range into the larger (or equal)
  //     one, so the larger range's original IDs are always smaller and
  //     IntEqClasses::join makes them the leader.
  //  3. compress() scans original IDs in ascending order, so the leading
  //     range's positions receive consecutive quotient IDs.
  int32_t rangeEnd = 0;
  for (int32_t i = 0; i < m; ++i) {
    RangeConstraint *c = qConsts[i];
    if (c != nullptr) {
      nodeConsts.push_back(
          {static_cast<int32_t>(c->allocations.size()), c->alignment});
      rangeEnd = i + static_cast<int32_t>(c->allocations.size());
    } else if (i < rangeEnd) {
      // Non-leader position within an active range; the leader's colorNode
      // call will write types[i] and mark this node visited.
      nodeConsts.push_back({0, 1});
    } else {
      nodeConsts.push_back({1, 1});
    }
    Value v = coalescingInfo->values[i];
    auto regTy = cast<AMDGCNRegisterTypeInterface>(v.getType());
    if (regTy.hasValueSemantics()) {
      emitError(v.getLoc()) << "found unexpected value register";
      return funcOp.emitError() << "failed to run register allocator";
    }
    types.push_back(regTy);
    if (!regTy.hasAllocatedSemantics())
      wasUnallocated.set(static_cast<unsigned>(i));
  }

  // Pre-colored members in a coalesced class pin the leader's physical range.
  if (failed(pinCoalescedClassesToFixedRegisters(
          m, nodeConsts, *graph, *coalescingInfo, types, wasUnallocated,
          funcOp.getLoc())))
    return failure();

  if (failed(colorGraph(coalescingInfo->graph, nodeConsts, types,
                        funcOp.getLoc(), numVGPRs, numAGPRs, numSGPRs)))
    return funcOp.emitError() << "failed to run register allocator";

  //===--------------------------------------------------------------------===//
  // Phase 2: IR rewrite.
  //===--------------------------------------------------------------------===//

  Region &body = funcOp.getFunctionBody();
  if (!body.empty()) {
    IRRewriter rewriter(funcOp->getContext());
    Block *entryBlock = &body.front();
    rewriter.setInsertionPointToStart(entryBlock);
    OpBuilder::InsertPoint ip = rewriter.saveInsertionPoint();

    rewriteCoalescing(*graph, *coalescingInfo, nodeConsts, types,
                      wasUnallocated, rewriter, ip);
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<InstRewritePattern, MakeRegisterRangeOpPattern,
               RegInterferenceOpPattern>(&getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPatternsGreedily(
          funcOp, frozenPatterns,
          GreedyRewriteConfig().setRegionSimplificationLevel(
              GreedySimplifyRegionLevel::Disabled)))) {
    return funcOp.emitError() << "failed to apply patterns";
  }
  return success();
}

void RegisterColoring::runOnOperation() {
  Operation *op = getOperation();
  WalkResult walkResult =
      op->walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
        if (failed(run(funcOp)))
          return WalkResult::interrupt();
        return WalkResult::skip();
      });
  if (walkResult.wasInterrupted())
    return signalPassFailure();

  // Set post-condition: all registers have allocated semantics.
  if (auto kernelOp = dyn_cast<KernelOp>(op))
    kernelOp.addNormalForms({AllRegistersAllocatedAttr::get(&getContext())});
}
