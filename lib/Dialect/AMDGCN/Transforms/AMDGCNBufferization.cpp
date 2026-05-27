//===- AMDGCNBufferization.cpp --------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/DPSAnalysis.h"
#include "aster/Analysis/LivenessAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/IR/CFG.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include <cstdint>
#include <limits>
#include <tuple>

#define DEBUG_TYPE "amdgcn-bufferization"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_AMDGCNBUFFERIZATION
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

/// Returns true if `type` is a special register (SCC, VCC, ...) with value
/// semantics, i.e. a register that must be tunnelled through an SGPR carrier.
static bool isSpecialReg(Type type) {
  if (!type.hasTrait<SpecialRegTrait>())
    return false;
  return cast<RegisterTypeInterface>(type).hasValueSemantics();
}

/// Returns the SGPR carrier type for a given special register type.
static RegisterTypeInterface getSGPRCarrier(Type sregTy) {
  auto regTy = cast<RegisterTypeInterface>(sregTy);
  int64_t sizeInBits = regTy.getSizeInBits();
  assert(sizeInBits > 0 &&
         sizeInBits <= 32 * std::numeric_limits<int16_t>::max() &&
         "register size out of range");
  int16_t words = static_cast<int16_t>((sizeInBits + 31) / 32);
  return getSGPR(sregTy.getContext(), words);
}

namespace {
//===----------------------------------------------------------------------===//
// AMDGCNBufferization pass
//===----------------------------------------------------------------------===//
struct AMDGCNBufferization
    : public amdgcn::impl::AMDGCNBufferizationBase<AMDGCNBufferization> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Allocator
//===----------------------------------------------------------------------===//
/// `amdgcn.alloca` allocator.
struct Allocator {
  Allocator(Block *entryBB) : entryBB(entryBB) {
    assert(entryBB != nullptr && "entry block cannot be null");
  }

  /// Get or create an alloca for the given type. For special-registers this
  /// returns a unique allocation.
  Value getOrCreateAlloc(OpBuilder &builder, Location loc, Type type);

private:
  Block *entryBB;
  DenseMap<Type, Value> sregAllocMap;
};

//===----------------------------------------------------------------------===//
// BufferizationImpl
//===----------------------------------------------------------------------===//
/// Register bufferization implementation.
/// The handling of block arguments is inspired by:
///   Benoit Boissinot, Alain Darte, Fabrice Rastello, Benoît Dupont de
///   Dinechin, Christophe Guillon. Revisiting Out-of-SSA Translation for
///   Correctness, Code Quality, and Efficiency. [Research Report] 2008, pp.14.
///   ⟨inria-00349925v1⟩
struct BufferizationImpl {
  BufferizationImpl(Allocator &allocator, DominanceInfo &domInfo,
                    DPSAnalysis &dpsAnalysis,
                    DPSClobberingAnalysis &dpsLiveness)
      : allocator(allocator), domInfo(domInfo), dpsAnalysis(dpsAnalysis),
        dpsLiveness(dpsLiveness) {}

  /// Run the bufferization transform.
  void run(RewriterBase &rewriter, FunctionOpInterface op);

  /// Insert de-clobbering allocas for the given operation.
  void handleInstruction(RewriterBase &rewriter, InstOpInterface op);

  /// Insert phi-breaking copies for the given block argument.
  void handleBlockArgument(RewriterBase &rewriter, BlockArgument arg);

  /// Insert phi-forwards.
  void handlePhiForwards(RewriterBase &rewriter);

  /// Insert phi-forwards for a group of phi-forwards.
  void handlePhiForwardGroup(RewriterBase &rewriter, int64_t start,
                             int64_t end);

  /// Remove register values from the terminators and the given blocks.
  void handleBlocksAndTerminators(RewriterBase &rewriter,
                                  ArrayRef<Block *> blocks);
  /// An alloca allocator.
  Allocator &allocator;
  /// The dominance info.
  DominanceInfo &domInfo;
  /// The entry block of the function.
  Block *entryBlock = nullptr;
  /// The DPS analysis.
  DPSAnalysis &dpsAnalysis;
  /// The DPS liveness analysis.
  DPSClobberingAnalysis &dpsLiveness;
  /// The set of branch operations.
  DenseSet<BranchOpInterface> branchOps;
  /// The set of phi-node replacements.
  SmallVector<std::pair<BlockArgument, Value>> phiReplacements;
  /// A list containing, the branch operation to forward from, the successor
  /// index, the value to forward, the block to forward to, the allocation to
  /// use, and the argument number.
  SmallVector<
      std::tuple<BranchOpInterface, int32_t, Value, Block *, Value, int64_t>>
      phiForwards;
  /// A map from the processed block to a unique deterministic ID.
  DenseMap<Block *, int64_t> blockToId;
};
} // namespace

//===----------------------------------------------------------------------===//
// Allocator
//===----------------------------------------------------------------------===//

Value Allocator::getOrCreateAlloc(OpBuilder &builder, Location loc, Type type) {
  auto rTy = cast<RegisterTypeInterface>(type);

  // For non-special registers, just create a new allocation.
  if (!rTy.hasTrait<SpecialRegTrait>())
    return createAllocation(builder, loc, rTy);

  // For special registers, check if the allocation is already cached.
  Value &alloc = sregAllocMap[type];
  if (alloc)
    return alloc;

  // Create a new allocation and cache it.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entryBB);
  alloc = createAllocation(builder, loc, rTy);

  // For composite special registers, map the sub-allocations to the cache.
  if (bitEnumContainsAll(rTy.getProps(), RegisterProps::IsComposite)) {
    auto rOp = alloc.getDefiningOp<MakeRegisterRangeOp>();
    assert(rOp && "expected allocation to be a range");
    for (Value v : rOp.getInputs())
      sregAllocMap[v.getType()] = v;
  }
  return alloc;
}

//===----------------------------------------------------------------------===//
// BufferizationImpl
//===----------------------------------------------------------------------===//

void BufferizationImpl::run(RewriterBase &rewriter, FunctionOpInterface op) {
  entryBlock = &op.getFunctionBody().getBlocks().front();
  // Insert de-clobbering allocas for all instructions that needed.
  op.walk([&](InstOpInterface op) {
    rewriter.setInsertionPoint(op);
    handleInstruction(rewriter, op);
  });

  SmallVector<Block *> blocksToUpdate;
  // Insert phi-breaking copies for all blocks that needed.
  op.walk([&](Block *block) {
    blockToId[block] = blockToId.size();
    rewriter.setInsertionPointToStart(block);
    bool needsUpdate = false;
    for (BlockArgument arg : block->getArguments()) {
      auto regTy = dyn_cast<RegisterTypeInterface>(arg.getType());
      if (!regTy || !regTy.hasValueSemantics())
        continue;
      handleBlockArgument(rewriter, arg);
      needsUpdate = true;
    }
    if (needsUpdate)
      blocksToUpdate.push_back(block);
  });
  handlePhiForwards(rewriter);
  handleBlocksAndTerminators(rewriter, blocksToUpdate);
}

void BufferizationImpl::handleInstruction(RewriterBase &rewriter,
                                          InstOpInterface instOp) {
  ResultRange results = instOp.getInstResults();
  if (results.empty())
    return;

  LDBG() << "- Handling instruction: " << instOp;
  rewriter.setInsertionPoint(instOp);

  OperandRange outs = instOp.getInstOuts();
  MutableArrayRef<OpOperand> operands =
      instOp->getOpOperands().slice(outs.getBeginOperandIndex(), outs.size());
  ArrayRef<bool> resultInfo = dpsLiveness.getClobberingInfo(instOp);
  assert(results.size() == resultInfo.size() &&
         "expected number of results to match clobbering info size");
  // `pos` tracks position within register-value-semantic outs only (not all
  // outs). resultInfo has one entry per value-semantic out, matching results.
  int64_t pos = 0;
  for (auto &&[idx, out] : llvm::enumerate(operands)) {
    auto regTy = dyn_cast<RegisterTypeInterface>(out.get().getType());
    if (!regTy || !regTy.hasValueSemantics())
      continue;

    if (!resultInfo[pos++])
      continue;

    Value newAlloca =
        allocator.getOrCreateAlloc(rewriter, instOp.getLoc(), regTy);
    out.set(newAlloca);
    LDBG() << "-- De-clobbering out operand: " << idx;
  }
}

void BufferizationImpl::handleBlockArgument(RewriterBase &rewriter,
                                            BlockArgument arg) {
  const DPSAnalysis::ProvenanceSet *provenance = dpsAnalysis.getProvenance(arg);
  assert(provenance != nullptr && "block argument must have provenance");

  auto regTy = cast<RegisterTypeInterface>(arg.getType());
  Location loc = arg.getLoc();
  Block *block = arg.getOwner();

  // Insert allocas to handle the breakage of the phi-node. For special
  // registers (SCC, VCC, ...) the phi carrier must be an SGPR because
  // unallocated SCC/VCC are not valid intermediaries.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(entryBlock);
  RegisterTypeInterface carrierTy;
  if (isSpecialReg(arg.getType())) {
    // Use the unallocated SGPR slot so the phi-forward copy has a resource
    // side-effect and is not eliminated as dead code.
    carrierTy = getSGPRCarrier(arg.getType()).getAsUnallocated();
  } else {
    carrierTy = regTy.getAsUnallocated();
  }
  Value commonAlloc = createAllocation(rewriter, loc, carrierTy);
  Value argAlloc = allocator.getOrCreateAlloc(rewriter, loc, regTy);

  // Save the branch, value to forward, destination block and allocation for
  // each phi-node to handle later.
  for (auto [branchOp, value, index] : *provenance) {
    auto brOp = cast<BranchOpInterface>(branchOp);
    branchOps.insert(brOp);
    phiForwards.push_back(
        {brOp, index, value, block, commonAlloc, arg.getArgNumber()});
  }

  // Insert a copy at the first possible insertion point in the block, which is
  // either before the first use or before the terminator if there are no uses.
  rewriter.setInsertionPoint(block->getTerminator());
  {
    SmallVector<Block::iterator> possibleIps;
    // Get all the uses of the argument and add them if they are in the same
    // block.
    for (OpOperand &use : arg.getUses()) {
      if (use.getOwner()->getBlock() != block)
        continue;
      possibleIps.push_back(Block::iterator(use.getOwner()));
    }

    // Sort the possible insertion points based on dominance order, so that we
    // can insert the copy as close as possible to the first use.
    llvm::sort(possibleIps, [&](Block::iterator lhs, Block::iterator rhs) {
      return domInfo.properlyDominates(block, lhs, block, rhs);
    });

    // If there are possible insertion points, insert the copy before the first
    // one.
    if (!possibleIps.empty())
      rewriter.setInsertionPoint(block, possibleIps.front());
  }

  auto cpy = lsir::CopyOp::create(rewriter, loc, argAlloc, commonAlloc);
  phiReplacements.push_back({arg, cpy.getTargetRes()});
}

void BufferizationImpl::handlePhiForwards(RewriterBase &rewriter) {
  auto getCmpTuple = [&](const std::tuple<BranchOpInterface, int32_t, Value,
                                          Block *, Value, int64_t> &elem) {
    BranchOpInterface brOp = std::get<0>(elem);
    int32_t index = std::get<1>(elem);
    Block *block = std::get<3>(elem);
    int64_t argNum = std::get<5>(elem);
    return std::make_tuple(brOp.getOperation(), index, blockToId[block],
                           argNum);
  };

  // Sort the phiForwards by the branch operation, the successor index, the
  // block to forward to, and the argument number.
  llvm::sort(phiForwards,
             [&](const std::tuple<BranchOpInterface, int32_t, Value, Block *,
                                  Value, int64_t> &a,
                 const std::tuple<BranchOpInterface, int32_t, Value, Block *,
                                  Value, int64_t> &b) {
               return getCmpTuple(a) < getCmpTuple(b);
             });

  auto it = phiForwards.begin();
  while (it != phiForwards.end()) {
    // Get the iterator to the first element with a different branch operation
    // or block.
    auto nextIt = it;
    while (++nextIt != phiForwards.end() &&
           std::get<0>(*nextIt) == std::get<0>(*it) &&
           std::get<1>(*nextIt) == std::get<1>(*it)) {
    }
    // Insert phi-forwards for the group.
    handlePhiForwardGroup(rewriter, it - phiForwards.begin(),
                          nextIt - phiForwards.begin());
    it = nextIt;
  }
}

void BufferizationImpl::handlePhiForwardGroup(RewriterBase &rewriter,
                                              int64_t start, int64_t end) {
  // Get the origin block and region.
  Block *prdBlock = std::get<0>(phiForwards[start])->getBlock();
  Region *prdRegion = prdBlock->getParent();

  SuccessorOperands succOperands =
      std::get<0>(phiForwards[start])
          .getSuccessorOperands(std::get<1>(phiForwards[start]));

  SmallVector<Value> fwdValues = llvm::to_vector(
      llvm::make_filter_range(succOperands.getForwardedOperands(), [](Value v) {
        auto regTy = dyn_cast<RegisterTypeInterface>(v.getType());
        return !regTy || !regTy.hasValueSemantics();
      }));
  succOperands.getMutableForwardedOperands().clear();

  // Create a new block to insert the phi-forwards.
  Block *newBlock =
      rewriter.createBlock(prdRegion, ++Region::iterator(prdBlock));
  rewriter.setInsertionPointToEnd(newBlock);

  // Create copies and set the successors.
  for (int64_t i = start; i < end; ++i) {
    auto [brOp, index, value, block, alloc, argNum] = phiForwards[i];
    lsir::CopyOp::create(rewriter, alloc.getLoc(), alloc, value);
    brOp->setSuccessor(newBlock, index);
  }

  // Create a branch op to the block to forward to.
  lsir::BranchOp::create(rewriter, std::get<0>(phiForwards[start])->getLoc(),
                         std::get<3>(phiForwards[start]), fwdValues);
}

void BufferizationImpl::handleBlocksAndTerminators(RewriterBase &rewriter,
                                                   ArrayRef<Block *> blocks) {
  auto isRegValType = [](Value value) {
    auto regTy = dyn_cast<RegisterTypeInterface>(value.getType());
    return regTy && regTy.hasValueSemantics();
  };

  // For each branch op, remove successor operands with register value
  // semantics.
  for (BranchOpInterface branchOp : branchOps) {
    for (auto [idx, succ] : llvm::enumerate(branchOp->getSuccessors())) {
      SuccessorOperands succOperands = branchOp.getSuccessorOperands(idx);
      assert(succOperands.getProducedOperandCount() == 0 &&
             "expected no produced operands");
      MutableOperandRange forwarded =
          succOperands.getMutableForwardedOperands();
      int64_t start = 0;
      while (start < forwarded.size()) {
        if (!isRegValType(forwarded[start].get())) {
          ++start;
          continue;
        }
        forwarded.erase(start);
      }
    }
  }

  // Replace the phi-nodes.
  for (auto [arg, value] : phiReplacements)
    rewriter.replaceAllUsesWith(arg, value);

  // Erase block arguments with register value semantics.
  for (Block *block : blocks)
    block->eraseArguments(isRegValType);
}

//===----------------------------------------------------------------------===//
// AMDGCNBufferization pass
//===----------------------------------------------------------------------===//

/// Canonicalize outs of special register-typed instructions to shared
/// allocations.
static void canonicalizeSRegAllocas(IRRewriter &rewriter, Allocator &allocator,
                                    FunctionOpInterface op) {
  SetVector<AllocaOpInterface> allocas;
  op.walk([&](InstOpInterface op) {
    OperandRange outs = op.getInstOuts();
    if (outs.empty())
      return;
    MutableArrayRef<OpOperand> operands =
        op->getOpOperands().slice(outs.getBeginOperandIndex(), outs.size());
    // Go through each out-operand and canonicalize it to a shared allocation.
    for (OpOperand &out : operands) {
      if (!out.get().getType().hasTrait<SpecialRegTrait>())
        continue;
      // If the out-operand is an alloca, add it to the list of allocas to
      // erase.
      if (auto maybeAlloc =
              dyn_cast_if_present<AllocaOpInterface>(out.get().getDefiningOp()))
        allocas.insert(maybeAlloc);

      // Get the shared allocation for the special register type.
      Value alloc = allocator.getOrCreateAlloc(rewriter, op.getLoc(),
                                               out.get().getType());
      out.set(alloc);
    }
  });

  // Erase the allocas.
  for (AllocaOpInterface alloc : allocas) {
    if (alloc->use_empty())
      rewriter.eraseOp(alloc);
  }
}

//===----------------------------------------------------------------------===//
// SRegBufferization
//===----------------------------------------------------------------------===//

/// Returns true if the use of a special register value requires
/// materialization (i.e., the use demands the physical special register).
static bool requiresMaterialization(OpOperand &use) {
  Operation *owner = use.getOwner();

  // lsir.cond_br always requires the physical special register.
  if (auto brOp = dyn_cast<lsir::CondBranchOp>(owner);
      brOp &&
      brOp.getConditionMutable().getOperandNumber() == use.getOperandNumber())
    return true;

  // For InstOpInterface, check if this operand is a special operand.
  auto instOp = dyn_cast<InstOpInterface>(owner);
  if (!instOp)
    return false;

  SmallVector<SpecialOperand> specials;
  instOp.getSpecialOperands(specials);
  for (SpecialOperand &sp : specials) {
    if (sp.get() == &use)
      return true;
  }
  return false;
}

namespace {
/// Tracks the current definition of each special register type and detects
/// clobbering. Values that are clobbered are promoted to SGPRs.
struct SRegBufferization : CFGWalker<SRegBufferization> {
  SRegBufferization(IRRewriter &rewriter, Allocator &allocator,
                    DominanceInfo &domInfo)
      : rewriter(rewriter), allocator(allocator), domInfo(domInfo) {}

  LogicalResult run(FunctionOpInterface funcOp);
  LogicalResult visitOp(Operation *op);
  LogicalResult visitControlFlowEdge(const BranchPoint &branchPoint,
                                     const Successor &successor);

  /// Push a scope before descending into a branch so that definitions made
  /// inside the branch are automatically undone when the scope is destroyed.
  LogicalResult handleBranch(const BranchPoint &branchPoint,
                             const Successor &successor);

private:
  /// Record a new definition of a special register, promoting any live prior
  /// value that would be clobbered.
  void recordDefinition(Value newDef, Type sregTy, Operation *insertBefore);

  /// Promote a special register value to an SGPR. Inserts a copy before
  /// `insertBefore` and rewrites uses dominated by the clobber point.
  void promoteValue(Value sregVal, Operation *insertBefore);

  /// Rewrite a single use of a promoted value, either amending or
  /// materializing.
  void rewriteUse(OpOperand &use, Value sgprVal, Type sregTy);

  IRRewriter &rewriter;
  Allocator &allocator;
  DominanceInfo &domInfo;
  /// Maps a special register type to its current live definition.
  llvm::ScopedHashTable<Type, Value> currentDef;
};
} // namespace

void SRegBufferization::recordDefinition(Value newDef, Type sregTy,
                                         Operation *insertBefore) {
  Value prevDef = currentDef.lookup(sregTy);
  // If there is a live previous definition, promote it before it is
  // clobbered.
  if (prevDef && prevDef != newDef)
    promoteValue(prevDef, insertBefore);
  currentDef.insert(sregTy, newDef);
}

void SRegBufferization::promoteValue(Value sregVal, Operation *insertBefore) {
  // Collect uses that need rewriting. Only rewrite uses that are dominated by
  // the clobber point: same-block uses after the clobber, or uses in blocks
  // dominated by the clobber's block.
  Block *clobberBlock = insertBefore->getBlock();
  SmallVector<OpOperand *> usesToRewrite;
  for (OpOperand &use : sregVal.getUses()) {
    Operation *owner = use.getOwner();
    if (owner->getBlock() == clobberBlock) {
      if (owner->isBeforeInBlock(insertBefore))
        continue;
      usesToRewrite.push_back(&use);
    } else if (domInfo.dominates(clobberBlock, owner->getBlock())) {
      usesToRewrite.push_back(&use);
    }
  }

  // Nothing to promote if the clobbered value has no remaining uses.
  if (usesToRewrite.empty())
    return;

  Type sregTy = sregVal.getType();
  RegisterTypeInterface sgprTy = getSGPRCarrier(sregTy);
  Location loc = sregVal.getLoc();

  // Create an SGPR alloca and copy the special register value into it.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertBefore);
  Value sgprSlot = createAllocation(rewriter, loc, sgprTy);
  lsir::CopyOp copyOp = lsir::CopyOp::create(rewriter, loc, sgprSlot, sregVal);
  Value sgprVal = copyOp.getTargetRes();

  for (OpOperand *use : usesToRewrite)
    rewriteUse(*use, sgprVal, sregTy);
}

void SRegBufferization::rewriteUse(OpOperand &use, Value sgprVal, Type sregTy) {
  if (!requiresMaterialization(use)) {
    // The use accepts an SGPR directly -- just amend the operand.
    use.set(sgprVal);
    return;
  }

  // The use requires the physical special register. Materialize it by copying
  // the SGPR value back into the special register alloca.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(use.getOwner());
  Value sregSlot =
      allocator.getOrCreateAlloc(rewriter, sgprVal.getLoc(), sregTy);
  lsir::CopyOp materialize =
      lsir::CopyOp::create(rewriter, sgprVal.getLoc(), sregSlot, sgprVal);
  use.set(materialize.getTargetRes());
}

LogicalResult SRegBufferization::visitOp(Operation *op) {
  auto instOp = dyn_cast<InstOpInterface>(op);
  if (!instOp)
    return success();

  // Scan all results for special register definitions.
  for (OpResult result : instOp.getInstResults()) {
    Type resultTy = result.getType();
    if (!isSpecialReg(resultTy))
      continue;
    recordDefinition(result, resultTy, op);
  }

  return success();
}

LogicalResult
SRegBufferization::visitControlFlowEdge(const BranchPoint &branchPoint,
                                        const Successor &successor) {
  if (!successor.isBlock())
    return success();

  Block *block = successor.getTarget<Block *>();
  // Block arguments with special register types are definitions.
  for (BlockArgument arg : block->getArguments()) {
    if (!isSpecialReg(arg.getType()))
      continue;
    recordDefinition(arg, arg.getType(), &block->front());
  }
  return success();
}

LogicalResult SRegBufferization::handleBranch(const BranchPoint &branchPoint,
                                              const Successor &successor) {
  // Push a new scope so definitions made inside this branch are undone when
  // the scope is destroyed, restoring the predecessor's state for siblings.
  llvm::ScopedHashTableScope<Type, Value> scope(currentDef);
  return CFGWalker<SRegBufferization>::handleBranch(branchPoint, successor);
}

LogicalResult SRegBufferization::run(FunctionOpInterface funcOp) {
  return walk(funcOp);
}

/// Run the bufferization transform on the given function.
static LogicalResult runOnFunction(FunctionOpInterface op,
                                   DominanceInfo &domInfo) {
  IRRewriter rewriter(op->getContext());
  Allocator allocator(&op.getFunctionBody().getBlocks().front());
  canonicalizeSRegAllocas(rewriter, allocator, op);

  // Create the dataflow solver and load the liveness analysis.
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  SymbolTableCollection symbolTable;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<LivenessAnalysis>(symbolTable);

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(op)))
    return op->emitError() << "failed to run liveness analysis";

  // Run the DPS analysis.
  FailureOr<DPSAnalysis> dpsResult = DPSAnalysis::create(op);
  if (failed(dpsResult))
    return op->emitError() << "failed to run DPS analysis";

  // Run the DPS liveness analysis.
  FailureOr<DPSClobberingAnalysis> livenessResult =
      DPSClobberingAnalysis::create(*dpsResult, solver, op);
  if (failed(livenessResult))
    return op->emitError() << "failed to run DPS liveness analysis";

  // Run the bufferization transform.
  BufferizationImpl impl(allocator, domInfo, *dpsResult, *livenessResult);
  impl.run(rewriter, op);

  // Promote special register values that are clobbered by later definitions.
  SRegBufferization sregImpl(rewriter, allocator, domInfo);
  if (failed(sregImpl.run(op)))
    return op->emitError() << "failed to run special register bufferization";

  return success();
}

void AMDGCNBufferization::runOnOperation() {
  Operation *moduleOp = getOperation();
  auto &domInfo = getAnalysis<DominanceInfo>();

  // Walk through the functions and run the bufferization transform.
  WalkResult result = moduleOp->walk([&](FunctionOpInterface op) {
    if (op.empty())
      return WalkResult::skip();

    if (failed(runOnFunction(op, domInfo)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  // Run CSE to clean up any redundant copies inserted by bufferization.
  IRRewriter rewriter(moduleOp->getContext());
  mlir::eliminateCommonSubExpressions(rewriter, domInfo, moduleOp);

  // Set post-condition: no register-typed block arguments remain.
  if (auto kernelOp = dyn_cast<KernelOp>(moduleOp))
    kernelOp.addNormalForms(
        {NoRegisterBlockArgsAttr::get(moduleOp->getContext())});
}
