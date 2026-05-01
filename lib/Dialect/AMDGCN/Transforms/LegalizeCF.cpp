//===- LegalizeCF.cpp - Legalize CF ops to AMDGCN instructions ------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass legalizes CF dialect operations (cf.cond_br, cf.br) and lsir.cmpi
// to AMDGCN scalar branch and compare instructions. It runs after register
// allocation when operands are in physical registers.
//
// Transformations:
//   - lsir.cmpi (SGPR/i32 operands) -> s_cmp_* (sets SCC flag)
//   - lsir.cmpi (VGPR operands) -> v_cmp_* (sets VCC flag)
//   - cf.cond_br -> s_cbranch_scc1/scc0 or s_cbranch_vccnz/vccz + s_branch
//   - cf.br -> s_branch
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNAttrs.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_LEGALIZECF
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// LegalizeCF pass
//===----------------------------------------------------------------------===//

struct LegalizeCF : public amdgcn::impl::LegalizeCFBase<LegalizeCF> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Verify i1 lifetime constraints for SCC/VCC registers:
  /// 1. No i1 value is used across block boundaries (flag reg not preserved).
  /// 2. No two lsir.cmp ops have overlapping lifetimes within a block
  /// (clobber).
  LogicalResult verifyI1Lifetimes(Operation *op);

  /// Get or create the lowered compare instruction + alloca for an lsir.cmpi.
  /// Selects s_cmp_* (SCC) for scalar operands, v_cmp_* (VCC) for vector.
  /// On first call for a given cmpOp, creates the alloca and compare at the
  /// original lsir.cmpi location. On subsequent calls, returns the cached
  /// result.
  Value getOrCreateLoweredCmp(lsir::CmpIOp cmpOp, IRRewriter &rewriter);

  /// Lower lsir.cmpi + cf.cond_br pattern to AMDGCN compare + branch.
  LogicalResult lowerCondBranch(cf::CondBranchOp condBr);

  /// Lower cf.br to s_branch.
  LogicalResult lowerBranch(cf::BranchOp br);

  /// Lower lsir.cmpi + lsir.select(i1) pattern to s_cmp + s_cselect_b32
  /// or v_cmp + v_cndmask_b32.
  LogicalResult lowerSelect(lsir::SelectOp selectOp);

  /// Map from original lsir.cmpi to the SCC/VCC alloca value from the lowered
  /// compare. Used to deduplicate compare lowering on fan-out.
  DenseMap<Operation *, Value> loweredCmpMap;
};

/// Create the appropriate s_cmp_* instruction for a scalar compare.
static void createScalarCompare(IRRewriter &rewriter, Location loc,
                                arith::CmpIPredicate predicate, Value scc,
                                Value lhs, Value rhs) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    SCmpEqI32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ne:
    SCmpLgI32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::slt:
    SCmpLtI32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::sle:
    SCmpLeI32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::sgt:
    SCmpGtI32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::sge:
    SCmpGeI32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ult:
    SCmpLtU32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ule:
    SCmpLeU32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ugt:
    SCmpGtU32::create(rewriter, loc, scc, lhs, rhs);
    return;
  case arith::CmpIPredicate::uge:
    SCmpGeU32::create(rewriter, loc, scc, lhs, rhs);
    return;
  }
  llvm_unreachable("Unknown CmpIPredicate");
}

/// Create the appropriate v_cmp_* instruction for a vector compare.
/// The 32-bit VOPC encoding requires rhs (src1) to be a VGPR.
/// If operands need swapping, the predicate should be flipped first.
static void createVectorCompare(IRRewriter &rewriter, Location loc,
                                arith::CmpIPredicate predicate, Value vcc,
                                Value lhs, Value rhs) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    VCmpEqI32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ne:
    VCmpNeI32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::slt:
    VCmpLtI32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::sle:
    VCmpLeI32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::sgt:
    VCmpGtI32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::sge:
    VCmpGeI32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ult:
    VCmpLtU32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ule:
    VCmpLeU32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::ugt:
    VCmpGtU32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  case arith::CmpIPredicate::uge:
    VCmpGeU32::create(rewriter, loc, vcc, lhs, rhs);
    return;
  }
  llvm_unreachable("Unknown CmpIPredicate");
}

/// Swap a comparison predicate (a < b becomes b > a).
static arith::CmpIPredicate swapPredicate(arith::CmpIPredicate pred) {
  switch (pred) {
  case arith::CmpIPredicate::eq:
    return arith::CmpIPredicate::eq;
  case arith::CmpIPredicate::ne:
    return arith::CmpIPredicate::ne;
  case arith::CmpIPredicate::slt:
    return arith::CmpIPredicate::sgt;
  case arith::CmpIPredicate::sle:
    return arith::CmpIPredicate::sge;
  case arith::CmpIPredicate::sgt:
    return arith::CmpIPredicate::slt;
  case arith::CmpIPredicate::sge:
    return arith::CmpIPredicate::sle;
  case arith::CmpIPredicate::ult:
    return arith::CmpIPredicate::ugt;
  case arith::CmpIPredicate::ule:
    return arith::CmpIPredicate::uge;
  case arith::CmpIPredicate::ugt:
    return arith::CmpIPredicate::ult;
  case arith::CmpIPredicate::uge:
    return arith::CmpIPredicate::ule;
  }
  llvm_unreachable("Unknown CmpIPredicate");
}

/// Returns true if either operand of the compare is a VGPR.
static bool hasVGPROperand(lsir::CmpIOp cmpOp) {
  return isa<VGPRType>(cmpOp.getLhs().getType()) ||
         isa<VGPRType>(cmpOp.getRhs().getType());
}

Value LegalizeCF::getOrCreateLoweredCmp(lsir::CmpIOp cmpOp,
                                        IRRewriter &rewriter) {
  auto it = loweredCmpMap.find(cmpOp);
  if (it != loweredCmpMap.end())
    return it->second;

  // Create the lowered compare at the original lsir.cmpi location.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(cmpOp);
  Location loc = cmpOp.getLoc();

  bool isVector = hasVGPROperand(cmpOp);
  if (isVector) {
    // Vector compare: v_cmp_* writes to VCC.
    // The 32-bit VOPC encoding requires src1 (rhs) to be a VGPR.
    // If rhs is not a VGPR, swap operands and flip the predicate.
    Value lhs = cmpOp.getLhs();
    Value rhs = cmpOp.getRhs();
    arith::CmpIPredicate pred = cmpOp.getPredicate();
    if (!isa<VGPRType>(rhs.getType())) {
      assert(isa<VGPRType>(lhs.getType()) &&
             "at least one operand must be a VGPR for vector compare");
      std::swap(lhs, rhs);
      pred = swapPredicate(pred);
    }
    Type vccType = VCCType::get(rewriter.getContext(), Register(0));
    Value vcc = AllocaOp::create(rewriter, loc, vccType);
    createVectorCompare(rewriter, loc, pred, vcc, lhs, rhs);
    loweredCmpMap[cmpOp] = vcc;
    return vcc;
  }

  // Scalar compare: s_cmp_* writes to SCC.
  Type sccType = SCCType::get(rewriter.getContext(), Register(0));
  Value scc = AllocaOp::create(rewriter, loc, sccType);
  createScalarCompare(rewriter, loc, cmpOp.getPredicate(), scc, cmpOp.getLhs(),
                      cmpOp.getRhs());

  loweredCmpMap[cmpOp] = scc;
  return scc;
}

LogicalResult LegalizeCF::lowerCondBranch(cf::CondBranchOp condBr) {
  // Get the condition - must come from lsir.cmpi
  Value condition = condBr.getCondition();
  auto cmpOp = condition.getDefiningOp<lsir::CmpIOp>();
  if (!cmpOp) {
    return condBr.emitError()
           << "cf.cond_br condition must come from lsir.cmpi for legalization";
  }

  // Note: We just drop block arguments as they are allocated and all values
  // flow through side effects.
  // TODO: In the future, this is better done as a RA legalization once we have
  // a side-effecting representation of instructions without return values.
  for (auto &brOpRange :
       {condBr.getTrueDestOperands(), condBr.getFalseDestOperands()}) {
    for (Value operand : brOpRange) {
      Type type = operand.getType();
      if (!isa<SGPRType, VGPRType, SGPRType, VGPRType>(type)) {
        return condBr.emitError()
               << "cf.br operand must have an allocated register type";
      }
    }
  }

  IRRewriter rewriter(condBr);
  rewriter.setInsertionPoint(condBr);

  Value flagReg = getOrCreateLoweredCmp(cmpOp, rewriter);
  bool isVector = isa<VCCType>(flagReg.getType());

  // Create conditional branch based on which destination is the next physical
  // block. The fallthrough target must be the next block.
  Location loc = condBr.getLoc();
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();
  Block *currentBlock = condBr->getBlock();
  Block *nextBlock = currentBlock->getNextNode();

  // Select branch opcodes based on whether the compare wrote SCC or VCC.
  OpCode branchTrue =
      isVector ? OpCode::S_CBRANCH_VCCNZ : OpCode::S_CBRANCH_SCC1;
  OpCode branchFalse =
      isVector ? OpCode::S_CBRANCH_VCCZ : OpCode::S_CBRANCH_SCC0;

  // amdgcn::CBranchOp takes a label; later, the actual 16-bit PC-relative
  // offset is computed by the LLVM assembler (MC layer) when it assembles this
  // text into binary machine code. This is happening outside of aster.
  if (falseDest == nextBlock) {
    // Branch to trueDest if flag set, fallthrough to falseDest
    CBranchOp::create(rewriter, loc, branchTrue, flagReg, trueDest, falseDest);
  } else if (trueDest == nextBlock) {
    // Branch to falseDest if flag clear, fallthrough to trueDest
    CBranchOp::create(rewriter, loc, branchFalse, flagReg, falseDest, trueDest);
  } else {
    // TODO: neither destination is the next block, we need more sophisticated
    // logic to insert explicit branch and create a new block. For this to
    // happen we need to first stabilize reg-alloc output guarantees (i.e. the
    // BBarg erasure needs to happen in the absence of SSA values flowing).
    // For now, emit an error if we reach such a case. The current behavior is
    // enough to model `scf.for` loops.
    return condBr.emitError()
           << "neither cf.cond_br destination is the next physical block; "
           << "block reordering not yet implemented";
  }

  // Erase the original cf.cond_br
  rewriter.eraseOp(condBr);

  return success();
}

LogicalResult LegalizeCF::lowerBranch(cf::BranchOp br) {
  // Note: We just drop block arguments as they are allocated and all values
  // flow through side effects.
  // TODO: In the future, this is better done as a RA legalization once we have
  // a side-effecting representation of instructions without return values.
  for (Value operand : br.getDestOperands()) {
    Type type = operand.getType();
    if (!isa<SGPRType, VGPRType, SGPRType, VGPRType>(type)) {
      return br.emitError()
             << "cf.br operand must have an allocated register type";
    }
  }

  Location loc = br.getLoc();
  IRRewriter rewriter(br);
  rewriter.setInsertionPoint(br);

  // Create unconditional branch
  BranchOp::create(rewriter, loc, OpCode::S_BRANCH, br.getDest());

  // Erase the original cf.br
  rewriter.eraseOp(br);

  return success();
}

LogicalResult LegalizeCF::lowerSelect(lsir::SelectOp selectOp) {
  Value condition = selectOp.getCondition();
  // Only handle i1-conditioned selects (from lsir.cmpi).
  // Register-conditioned selects are handled elsewhere.
  if (!condition.getType().isInteger(1))
    return success();

  auto cmpOp = condition.getDefiningOp<lsir::CmpIOp>();
  if (!cmpOp) {
    return selectOp.emitError()
           << "lsir.select with i1 condition must come from lsir.cmpi";
  }

  Location loc = selectOp.getLoc();
  IRRewriter rewriter(selectOp);
  rewriter.setInsertionPoint(selectOp);

  // Ensure the compare is lowered.
  Value flagReg = getOrCreateLoweredCmp(cmpOp, rewriter);
  bool isVector = isa<VCCType>(flagReg.getType());

  Value dst = selectOp.getDst();
  if (isVector) {
    // v_cndmask_b32: vdst = VCC[lane] ? src1 : src0 (note: reversed order!)
    // src0 = false_value, src1 = true_value, src2 = VCC
    // VOP2 src0 accepts i32/SGPR/VGPR, but src1 must be VGPR.
    Value trueVal = selectOp.getTrueValue();
    Value falseVal = selectOp.getFalseValue();
    if (!isa<VGPRType>(trueVal.getType())) {
      // src1 is not a VGPR. Materialize it into dst (an allocated VGPR)
      // via v_mov_b32, then use dst as src1.
      VMovB32::create(rewriter, loc, dst, trueVal);
      trueVal = dst;
    }
    VCndmaskB32::create(rewriter, loc, dst, falseVal, trueVal, flagReg);
  } else {
    // s_cselect_b32: sdst = SCC ? src0 : src1.
    // src0 = true_value (selected when SCC=1), src1 = false_value.
    SCselectB32::create(rewriter, loc, dst, selectOp.getTrueValue(),
                        selectOp.getFalseValue(), flagReg);
  }

  // Replace uses of the select result with the dst (which now holds the
  // s_cselect_b32 or v_cndmask_b32 result via side effect).
  if (selectOp->getNumResults() > 0)
    rewriter.replaceOp(selectOp, dst);
  else
    rewriter.eraseOp(selectOp);

  return success();
}

/// Find the last user of `value` in `block`, by operation order.
/// Returns nullptr if no user exists in the block.
static Operation *findLastUserInBlock(Value value, Block *block) {
  Operation *lastUser = nullptr;
  for (Operation *user : value.getUsers()) {
    if (user->getBlock() != block)
      continue;
    if (!lastUser || lastUser->isBeforeInBlock(user))
      lastUser = user;
  }
  return lastUser;
}

LogicalResult LegalizeCF::verifyI1Lifetimes(Operation *op) {
  LogicalResult result = success();

  op->walk([&](Block *block) {
    // Track the currently-live i1 value and where its lifetime ends.
    Operation *activeI1Op = nullptr;
    Operation *activeI1OpLastUserOp = nullptr;

    for (Operation &innerOp : *block) {
      Value i1;
      if (auto cmpOp = dyn_cast<lsir::CmpIOp>(&innerOp))
        i1 = cmpOp.getResult();
      else if (auto cmpOp = dyn_cast<lsir::CmpFOp>(&innerOp))
        i1 = cmpOp.getResult();
      else
        continue;

      // Check cross-block usage: all users of this cmpi must be in the same
      // block. SCC/VCC are not preserved across block boundaries.
      for (Operation *user : i1.getUsers()) {
        if (user->getBlock() != block) {
          innerOp.emitError()
              << "has consumer in a different block; flag register (SCC/VCC) "
                 "is not preserved across block boundaries";
          result = failure();
          return WalkResult::interrupt();
        }
      }

      // Check overlap: any cmpi (even dead ones) clobbers the flag register,
      // so if a previous i1 is still live, this is an error.
      if (activeI1Op && activeI1OpLastUserOp &&
          !activeI1OpLastUserOp->isBeforeInBlock(&innerOp)) {
        innerOp.emitError()
            << "would clobber flag register from earlier compare; i1 "
               "lifetimes must not overlap";
        result = failure();
        return WalkResult::interrupt();
      }

      // Dead cmpi (no users) is benign for tracking purposes -- it clobbers
      // SCC but has no consumers that could be affected by a future clobber.
      // Don't update activeI1 so it doesn't block subsequent live cmpi ops.
      if (i1.use_empty())
        continue;

      // Start tracking this cmpi's lifetime.
      activeI1Op = &innerOp;
      activeI1OpLastUserOp = findLastUserInBlock(i1, block);
    }
    return WalkResult::advance();
  });

  return result;
}

void LegalizeCF::runOnOperation() {
  Operation *op = getOperation();

  // Pre-condition: all registers must be allocated before CF legalization.
  if (auto kernelOp = dyn_cast<KernelOp>(op)) {
    if (!kernelOp.hasNormalForm(
            AllRegistersAllocatedAttr::get(op->getContext()))) {
      op->emitError() << "amdgcn-legalize-cf requires "
                         "#amdgcn.all_registers_allocated normal form";
      return signalPassFailure();
    }
  }

  // Precondition: verify i1 lifetimes are non-overlapping and block-local.
  // SCC/VCC are implicit flag registers with no spill capability, so
  // overlapping lifetimes or cross-block usage would produce silent
  // miscompilation.
  if (failed(verifyI1Lifetimes(op))) {
    signalPassFailure();
    return;
  }

  // Construct allocated register to alloca map.
  // After canonicalize + CSE (run in the backend pipeline before this pass),
  // there must be exactly one alloca per concrete register type. CSE
  // deduplicates allocas of the same type since allocated allocas are Pure.
  DenseMap<RegisterTypeInterface, AllocaOp> allocatedRegisterToAllocaMap;
  bool hasDuplicates = false;
  op->walk([&](AllocaOp alloca) {
    auto regType = cast<RegisterTypeInterface>(alloca.getType());
    if (!regType.hasAllocatedSemantics()) {
      alloca.emitOpError("alloca must have a fixed register type "
                         "(register coloring must run before LegalizeCF)");
      hasDuplicates = true;
      return;
    }
    auto [it, inserted] =
        allocatedRegisterToAllocaMap.try_emplace(regType, alloca);
    if (!inserted) {
      alloca.emitOpError("duplicate alloca for register type ")
          << alloca.getType()
          << " (canonicalize + CSE should deduplicate allocated allocas)";
      hasDuplicates = true;
    }
  });
  if (hasDuplicates) {
    signalPassFailure();
    return;
  }

  // Collect all operations to lower.
  SmallVector<lsir::SelectOp> selects;
  SmallVector<cf::CondBranchOp> condBranches;
  SmallVector<cf::BranchOp> branches;
  op->walk([&](Operation *innerOp) {
    if (auto selectOp = dyn_cast<lsir::SelectOp>(innerOp))
      selects.push_back(selectOp);
    else if (auto condBr = dyn_cast<cf::CondBranchOp>(innerOp))
      condBranches.push_back(condBr);
    else if (auto br = dyn_cast<cf::BranchOp>(innerOp))
      branches.push_back(br);
  });

  // Lower i1-conditioned selects first (they reference lsir.cmpi which may
  // also be used by cond_br).
  for (lsir::SelectOp selectOp : selects) {
    if (failed(lowerSelect(selectOp))) {
      signalPassFailure();
      return;
    }
  }

  // Lower conditional branches (they may reference lsir.cmpi)
  for (cf::CondBranchOp condBr : condBranches) {
    if (failed(lowerCondBranch(condBr))) {
      signalPassFailure();
      return;
    }
  }

  // Lower unconditional branches
  for (cf::BranchOp br : branches) {
    if (failed(lowerBranch(br))) {
      signalPassFailure();
      return;
    }
  }

  // Erase original lsir.cmpi ops that were lowered. Collect first, then
  // clear the map before erasing to avoid dangling pointers during iteration.
  SmallVector<Operation *> cmpsToErase;
  for (auto &[cmpOp, scc] : loweredCmpMap) {
    assert(cmpOp->use_empty() &&
           "lsir.cmpi still has uses after all consumers lowered");
    cmpsToErase.push_back(cmpOp);
  }
  loweredCmpMap.clear();
  for (Operation *cmpOp : cmpsToErase)
    cmpOp->erase();

  // Iterate all blocks in all regions of the function and replace block
  // arguments with the corresponding alloca.
  //
  // For register range block arguments, we decompose them to individual
  // registers since ranges are composite constructs without their own allocas.
  // Each range block arg is replaced by reconstructing the range from its
  // constituent allocas using make_register_range at the block entry.
  //
  // This is a simple way of legalizing block arguments, late in the pipeline.
  //
  // Note and caveat: taking the alloc is fine because at this point values do
  // not flow through SSA values anymore, except i1  cf.cond_br conditions.
  // While this is correct, it is easily confusing since SSA and side-effects
  // are mixed in the same representation.
  //
  // TODO: In the very short future, this is better done as a RA legalization
  // once we have a side-effecting representation of instructions without return
  // values.
  op->walk([&](Block *block) {
    IRRewriter rewriter(op->getContext());

    // Drop all block arguments, if any.
    for (int i = block->getNumArguments() - 1; i >= 0; --i) {
      // Always erase index i; indices shift after each erase.
      BlockArgument arg = block->getArgument(i);
      RegisterTypeInterface regType =
          cast<RegisterTypeInterface>(arg.getType());

      // Simple case: non-range register type
      if (!regType.isRegisterRange()) {
        auto it = allocatedRegisterToAllocaMap.find(regType);
        if (it == allocatedRegisterToAllocaMap.end()) {
          block->getParentOp()->emitError()
              << "Alloca not found for register type " << regType;
          signalPassFailure();
          return WalkResult::interrupt();
        }
        arg.replaceAllUsesWith(it->second);
        block->eraseArgument(i);
        continue;
      }

      // Complex case: register range type - decompose to constituents
      RegisterRange range = regType.getAsRange();
      Register beginReg = range.begin();
      int16_t rangeSize = range.size();

      if (beginReg.isRelocatable()) {
        block->getParentOp()->emitError()
            << "Cannot legalize relocatable register range block argument";
        signalPassFailure();
        return WalkResult::interrupt();
      }

      // Collect allocas for all constituent registers
      SmallVector<Value> constituentAllocas;
      constituentAllocas.reserve(rangeSize);

      auto rangeRegType = cast<AMDGCNRegisterTypeInterface>(regType);
      RegisterKind regKind = rangeRegType.getRegisterKind();

      for (int16_t offset = 0; offset < rangeSize; ++offset) {
        Register reg = beginReg.getWithOffset(offset);

        RegisterTypeInterface constituentType;
        MLIRContext *ctx = block->getParentOp()->getContext();
        switch (regKind) {
        case RegisterKind::SGPR:
          constituentType = SGPRType::get(ctx, reg);
          break;
        case RegisterKind::VGPR:
          constituentType = VGPRType::get(ctx, reg);
          break;
        case RegisterKind::AGPR:
          constituentType = AGPRType::get(ctx, reg);
          break;
        default:
          block->getParentOp()->emitError()
              << "Unsupported register kind for range block argument";
          signalPassFailure();
          return WalkResult::interrupt();
        }

        auto it = allocatedRegisterToAllocaMap.find(constituentType);
        if (it == allocatedRegisterToAllocaMap.end()) {
          block->getParentOp()->emitError()
              << "Alloca not found for constituent register " << constituentType
              << " in range " << regType;
          signalPassFailure();
          return WalkResult::interrupt();
        }
        constituentAllocas.push_back(it->second);
      }

      rewriter.setInsertionPointToStart(block);
      Value reconstructedRange = MakeRegisterRangeOp::create(
          rewriter, arg.getLoc(), constituentAllocas);
      arg.replaceAllUsesWith(reconstructedRange);
      block->eraseArgument(i);
    }
    return WalkResult::advance();
  });

  // Set post-condition: no CF branches remain.
  if (auto kernelOp = dyn_cast<KernelOp>(op))
    kernelOp.addNormalForms({NoCfBranchesAttr::get(op->getContext())});
}

} // namespace
