//===- ConvertLDSBuffers.cpp - Convert LDS Buffer Operations -------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ConvertLDSBuffers pass which converts LDS buffer
// operations to their final form after allocation.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_CONVERTLDSBUFFERS
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;

namespace {
struct ConvertLDSBuffers
    : public amdgcn::impl::ConvertLDSBuffersBase<ConvertLDSBuffers> {
public:
  using Base::Base;
  void runOnOperation() override;

private:
  /// Replace one AllocLDSOp's get_lds_offset uses with the assigned byte offset
  /// and queue the alloc/dealloc for erasure.
  void processAllocOp(RewriterBase &rewriter, amdgcn::AllocLDSOp allocOp,
                      SmallVectorImpl<Operation *> &deadOps);
};

/// Build the i32 byte-offset constant at the current insertion point.
Value createOffsetConstant(RewriterBase &rewriter, amdgcn::AllocLDSOp allocOp,
                           uint32_t off) {
  return arith::ConstantIntOp::create(rewriter, allocOp.getLoc(),
                                      static_cast<int64_t>(off), 32)
      .getResult();
}

/// Materialize the byte offset in a real register (amdgcn.alloca + s_mov_b32 /
/// v_mov_b32) at the current insertion point. The alloca gives it an allocation
/// the register-allocation DPS analysis tracks.
Value materializeOffsetRegister(RewriterBase &rewriter, Location loc,
                                Type regType, Value i32Const) {
  Value alloca = amdgcn::AllocaOp::create(rewriter, loc, regType);
  switch (amdgcn::getOperandKind(regType)) {
  case amdgcn::OperandKind::SGPR:
    return amdgcn::SMovB32::create(rewriter, loc, alloca, i32Const)
        .getDst0Res();
  case amdgcn::OperandKind::VGPR:
    return amdgcn::VMovB32::create(rewriter, loc, alloca, i32Const)
        .getDst0Res();
  default:
    llvm_unreachable("lds offset must materialize into an sgpr/vgpr");
  }
}

/// Replace a register-typed get_lds_offset (after codegen) with the byte offset
/// materialized in a real register. The downstream amdgcn-inline-constant-movs
/// pass reclaims the s_mov where the offset only reaches an inline-literal-
/// accepting VALU consumer.
void foldRegisterOffset(RewriterBase &rewriter, amdgcn::GetLDSOffsetOp ldsOffOp,
                        Value i32Const) {
  rewriter.setInsertionPoint(ldsOffOp);
  Value reg = materializeOffsetRegister(
      rewriter, ldsOffOp.getLoc(), ldsOffOp.getResult().getType(), i32Const);
  rewriter.replaceAllUsesWith(ldsOffOp.getResult(), reg);
}

/// Replace an index/i32 get_lds_offset (early fold, before codegen) with an
/// inline constant of its own type.
void foldIntegerOffset(RewriterBase &rewriter, amdgcn::GetLDSOffsetOp ldsOffOp,
                       uint32_t off, Value i32Const) {
  Type resTy = ldsOffOp.getResult().getType();
  Value c = i32Const;
  if (resTy != i32Const.getType())
    c = arith::ConstantOp::create(
        rewriter, ldsOffOp.getLoc(), resTy,
        rewriter.getIntegerAttr(resTy, static_cast<int64_t>(off)));
  rewriter.replaceAllUsesWith(ldsOffOp.getResult(), c);
}
} // namespace

void ConvertLDSBuffers::processAllocOp(RewriterBase &rewriter,
                                       amdgcn::AllocLDSOp allocOp,
                                       SmallVectorImpl<Operation *> &deadOps) {
  // Skip allocations without a concrete offset.
  std::optional<uint32_t> off = allocOp.getOffset();
  if (!off || ShapedType::isDynamic(allocOp.getStaticSize()))
    return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(allocOp);
  Value i32Const = createOffsetConstant(rewriter, allocOp, *off);

  for (Operation *user :
       llvm::make_early_inc_range(allocOp.getBuffer().getUsers())) {
    if (auto deallocOp = dyn_cast<amdgcn::DeallocLDSOp>(user)) {
      rewriter.eraseOp(deallocOp);
      continue;
    }
    auto ldsOffOp = dyn_cast<amdgcn::GetLDSOffsetOp>(user);
    if (!ldsOffOp)
      continue;
    if (!ldsOffOp.getResult().use_empty()) {
      if (isa<RegisterTypeInterface>(ldsOffOp.getResult().getType()))
        foldRegisterOffset(rewriter, ldsOffOp, i32Const);
      else
        foldIntegerOffset(rewriter, ldsOffOp, *off, i32Const);
    }
    deadOps.push_back(ldsOffOp);
  }

  // Erase the alloc once nothing references the buffer; otherwise queue it.
  if (allocOp.getBuffer().use_empty())
    rewriter.eraseOp(allocOp);
  else
    deadOps.push_back(allocOp);
}

void ConvertLDSBuffers::runOnOperation() {
  Operation *op = getOperation();

  IRRewriter rewriter(op->getContext());
  SmallVector<amdgcn::AllocLDSOp> allocOps;
  op->walk([&](amdgcn::AllocLDSOp allocOp) { allocOps.push_back(allocOp); });

  // Fold each buffer's offsets, then erase the dead get_lds_offset / alloc ops.
  // get_lds_offset ops precede their alloc in deadOps, so a single forward pass
  // erases each once it is use-empty.
  SmallVector<Operation *> deadOps;
  for (amdgcn::AllocLDSOp allocOp : allocOps)
    processAllocOp(rewriter, allocOp, deadOps);

  for (Operation *deadOp : deadOps)
    if (deadOp->use_empty())
      rewriter.eraseOp(deadOp);
}
