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
//   - lsir.cmpi (i32 operands) -> s_cmp_* (sets SCC flag)
//   - cf.cond_br -> s_cbranch_scc1/scc0 + s_branch
//   - cf.br -> s_branch
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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
  /// Lower lsir.cmpi + cf.cond_br pattern to AMDGCN compare + branch.
  LogicalResult lowerCondBranch(cf::CondBranchOp condBr);

  /// Lower cf.br to s_branch.
  LogicalResult lowerBranch(cf::BranchOp br);
};

/// Map arith::CmpIPredicate to the appropriate s_cmp_* opcode.
static OpCode getCompareOpCode(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return OpCode::S_CMP_EQ_I32;
  case arith::CmpIPredicate::ne:
    return OpCode::S_CMP_LG_I32;
  case arith::CmpIPredicate::slt:
    return OpCode::S_CMP_LT_I32;
  case arith::CmpIPredicate::sle:
    return OpCode::S_CMP_LE_I32;
  case arith::CmpIPredicate::sgt:
    return OpCode::S_CMP_GT_I32;
  case arith::CmpIPredicate::sge:
    return OpCode::S_CMP_GE_I32;
  case arith::CmpIPredicate::ult:
    return OpCode::S_CMP_LT_U32;
  case arith::CmpIPredicate::ule:
    return OpCode::S_CMP_LE_U32;
  case arith::CmpIPredicate::ugt:
    return OpCode::S_CMP_GT_U32;
  case arith::CmpIPredicate::uge:
    return OpCode::S_CMP_GE_U32;
  }
  llvm_unreachable("Unknown CmpIPredicate");
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
      if (!isa<SGPRType, VGPRType, SGPRRangeType, VGPRRangeType>(type)) {
        return condBr.emitError()
               << "cf.br operand must have an allocated register type";
      }
    }
  }

  Location loc = condBr.getLoc();
  IRRewriter rewriter(condBr);
  rewriter.setInsertionPoint(condBr);

  // Create SCC allocation for the compare result
  Type sccType = SCCType::get(rewriter.getContext());
  Value scc = AllocaOp::create(rewriter, loc, sccType);

  // Create the s_cmp_* instruction (sets SCC)
  OpCode cmpOpCode = getCompareOpCode(cmpOp.getPredicate());
  amdgcn::CmpIOp::create(rewriter, loc, cmpOpCode, scc, cmpOp.getLhs(),
                         cmpOp.getRhs());

  // Create conditional branch based on which destination is the next physical
  // block. The fallthrough target must be the next block.
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();
  Block *currentBlock = condBr->getBlock();
  Block *nextBlock = currentBlock->getNextNode();

  // amdgcn::CBranchOp takes a label; later, the actual 16-bit PC-relative
  // offset is computed by the LLVM assembler (MC layer) when it assembles this
  // text into binary machine code. This is happening outside of aster.
  if (falseDest == nextBlock) {
    // Use S_CBRANCH_SCC1: branch to trueDest if SCC=1, fallthrough to falseDest
    CBranchOp::create(rewriter, loc, OpCode::S_CBRANCH_SCC1, scc, trueDest,
                      falseDest);
  } else if (trueDest == nextBlock) {
    // Use S_CBRANCH_SCC0: branch to falseDest if SCC=0, fallthrough to trueDest
    CBranchOp::create(rewriter, loc, OpCode::S_CBRANCH_SCC0, scc, falseDest,
                      trueDest);
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

  // Erase the lsir.cmpi if it has no other uses
  if (cmpOp.use_empty())
    rewriter.eraseOp(cmpOp);

  return success();
}

LogicalResult LegalizeCF::lowerBranch(cf::BranchOp br) {
  // Note: We just drop block arguments as they are allocated and all values
  // flow through side effects.
  // TODO: In the future, this is better done as a RA legalization once we have
  // a side-effecting representation of instructions without return values.
  for (Value operand : br.getDestOperands()) {
    Type type = operand.getType();
    if (!isa<SGPRType, VGPRType, SGPRRangeType, VGPRRangeType>(type)) {
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

void LegalizeCF::runOnOperation() {
  Operation *op = getOperation();

  // Construct allocated register to alloca map.
  // Assumes all operands are in allocated registers and there is exactly one
  // alloca per register type.
  DenseMap<RegisterTypeInterface, AllocaOp> allocatedRegisterToAllocaMap;
  op->walk([&](AllocaOp alloca) {
    auto it = allocatedRegisterToAllocaMap.find(alloca.getType());
    assert(it == allocatedRegisterToAllocaMap.end() && "Alloca already exists");
    assert(!cast<RegisterTypeInterface>(alloca.getType()).isRelocatable() &&
           "Alloca must have a fixed register type");
    allocatedRegisterToAllocaMap.insert({alloca.getType(), alloca});
  });

  // Collect all cf.cond_br and cf.br operations
  SmallVector<cf::CondBranchOp> condBranches;
  SmallVector<cf::BranchOp> branches;
  op->walk([&](Operation *innerOp) {
    if (auto condBr = dyn_cast<cf::CondBranchOp>(innerOp))
      condBranches.push_back(condBr);
    else if (auto br = dyn_cast<cf::BranchOp>(innerOp))
      branches.push_back(br);
  });

  // Lower conditional branches first (they may reference lsir.cmpi)
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
}

} // namespace
