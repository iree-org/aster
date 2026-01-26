//===- WaitAnalysis.cpp - Wait dependency analysis -----------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include <algorithm>

#define DEBUG_TYPE "wait-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Merge two sorted sets of TokenState. Returns true if target changed.
/// Deduplicates by (kind, id), taking minimum position for duplicates.
static bool merge(SmallVectorImpl<TokenState> &target,
                  ArrayRef<TokenState> source) {
  if (source.empty())
    return false;
  if (target.empty()) {
    target.append(source.begin(), source.end());
    return true;
  }

  size_t oldSize = target.size();
  SmallVector<TokenState> result;
  result.reserve(target.size() + source.size());

  // Two-pointer merge with deduplication by (kind, id).
  auto i = target.begin(), ie = target.end();
  auto j = source.begin(), je = source.end();
  while (i != ie && j != je) {
    if (*i < *j) {
      result.push_back(*i++);
    } else if (*j < *i) {
      result.push_back(*j++);
    } else {
      // Same (kind, id): merge positions and advance both.
      TokenState merged = *i++;
      merged.merge(*j++);
      result.push_back(merged);
    }
  }
  result.append(i, ie);
  result.append(j, je);

  bool changed = result.size() != oldSize || result != target;
  target = std::move(result);
  return changed;
}

/// Get the defining block of a value.
static Block *getDefiningBlock(Value value) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return blockArg.getOwner();
  return value.getDefiningOp()->getBlock();
}

//===----------------------------------------------------------------------===//
// TokenState
//===----------------------------------------------------------------------===//

bool TokenState::merge(const TokenState &other) {
  assert(kind == other.kind && id == other.id &&
         "cannot merge incompatible tokens");
  Position prevPos = position;
  position = std::min(position, other.position);
  return position != prevPos;
}

void TokenState::print(raw_ostream &os) const {
  os << "{";
  if (token) {
    token.printAsOperand(os, OpPrintingFlags());
    os << ", " << id;
  } else {
    os << "<escaped>";
  }
  os << ", " << position;
  os << ", " << stringifyMemoryInstructionKind(kind);
  os << "}";
}

TokenState &TokenState::operator++() {
  if (position < kMaxPosition)
    ++position;
  return *this;
}

//===----------------------------------------------------------------------===//
// WaitCnt
//===----------------------------------------------------------------------===//

WaitCnt WaitCnt::fromOp(WaitOp waitOp) {
  return WaitCnt(waitOp.getVmCnt(), waitOp.getLgkmCnt());
}

int32_t WaitCnt::getCount(MemoryInstructionKind kind) const {
  switch (kind) {
  case MemoryInstructionKind::Flat:
    return vmCnt;
  case MemoryInstructionKind::Constant:
  case MemoryInstructionKind::Shared:
    return lgkmCnt;
  default:
    return -1;
  }
}

void WaitCnt::updateCount(MemoryInstructionKind kind, Position count) {
  switch (kind) {
  case MemoryInstructionKind::Flat:
    vmCnt = std::min(vmCnt, count);
    break;
  case MemoryInstructionKind::Constant:
  case MemoryInstructionKind::Shared:
    lgkmCnt = std::min(lgkmCnt, count);
    break;
  default:
    break;
  }
}

void WaitCnt::updateCount(ArrayRef<TokenState> tokens) {
  for (const TokenState &tok : tokens)
    updateCount(tok.getKind(), tok.getPosition());
}

void WaitCnt::setOpCounts(WaitOp waitOp) const {
  waitOp.setVmCnt(vmCnt);
  waitOp.setLgkmCnt(lgkmCnt);
}

void WaitCnt::print(llvm::raw_ostream &os) const {
  os << "{vm_cnt: ";
  if (vmCnt == kMaxPosition)
    os << "nowait";
  else
    os << vmCnt;
  os << ", lgkm_cnt: ";
  if (lgkmCnt == kMaxPosition)
    os << "nowait";
  else
    os << lgkmCnt;
  os << "}";
}

void WaitCnt::handleWait(ArrayRef<TokenState> reachingTokens,
                         ValueRange dependencies,
                         SmallVectorImpl<TokenState> &waitedTokens,
                         SmallVectorImpl<TokenState> &impliedTokens,
                         SmallVectorImpl<TokenState> &nextReachingTokens,
                         llvm::function_ref<TokenState(Value)> getState) {
  waitedTokens.clear();
  bool hasLgkm = false;

  // Find dependencies in the reaching set.
  for (Value v : dependencies) {
    TokenState tok = getState(v);
    auto lb = llvm::lower_bound(reachingTokens, tok);
    if (lb == reachingTokens.end() || lb->getToken() != v) {
      LDBG_OS([&](auto &os) {
        os << "  Dep not in reaching tokens: ";
        v.printAsOperand(os, OpPrintingFlags());
      });
      continue;
    }
    waitedTokens.push_back(*lb);
    auto k = tok.getKind();
    hasLgkm |= k == MemoryInstructionKind::Constant ||
               k == MemoryInstructionKind::Shared;
  }

  // If LGKM deps exist and reaching has both DS and SMEM, wait for all lgkm.
  if (hasLgkm) {
    bool hasDS = false, hasSMem = false;
    for (const TokenState &tok : reachingTokens) {
      hasDS |= tok.getKind() == MemoryInstructionKind::Shared;
      hasSMem |= tok.getKind() == MemoryInstructionKind::Constant;
    }
    if (hasDS && hasSMem)
      lgkmCnt = 0;
  }

  updateCount(waitedTokens);

  // Keep only tokens within wait count.
  llvm::erase_if(waitedTokens, [&](const TokenState &tok) {
    int32_t cnt = getCount(tok.getKind());
    bool invalid = tok.getPosition() > cnt || cnt <= 0;
    if (invalid && tok.getToken())
      LDBG_OS([&](auto &os) {
        os << "  Invalidating: ";
        tok.getToken().printAsOperand(os, OpPrintingFlags());
      });
    return invalid;
  });

  // Partition reaching tokens into next (position < count) and implied.
  for (const TokenState &tok : reachingTokens) {
    int32_t cnt = getCount(tok.getKind());
    (tok.getPosition() < cnt ? nextReachingTokens : impliedTokens)
        .push_back(tok);
  }
}

bool WaitCnt::handleEscapedTokens(SmallVectorImpl<TokenState> &results,
                                  SmallVectorImpl<TokenState> &escapedTokens) {
  LDBG() << "  Escaped tokens: " << llvm::interleaved_array(escapedTokens);
  // Compute minimum position for each memory kind.
  Position vmPos = kMaxPosition, lgkmSPos = kMaxPosition,
           lgkmDPos = kMaxPosition;
  for (const TokenState &tok : escapedTokens) {
    switch (tok.getKind()) {
    case MemoryInstructionKind::Flat:
      vmPos = std::min(vmPos, tok.getPosition());
      break;
    case MemoryInstructionKind::Constant:
      lgkmSPos = std::min(lgkmSPos, tok.getPosition());
      break;
    case MemoryInstructionKind::Shared:
      lgkmDPos = std::min(lgkmDPos, tok.getPosition());
      break;
    default:
      break;
    }
  }
  escapedTokens.clear();
  if (vmPos != kMaxPosition)
    escapedTokens.push_back(TokenState::unknownVMem(vmPos));
  if (lgkmSPos != kMaxPosition)
    escapedTokens.push_back(TokenState::unknownSMem(lgkmSPos));
  if (lgkmDPos != kMaxPosition)
    escapedTokens.push_back(TokenState::unknownDMem(lgkmDPos));
  llvm::sort(escapedTokens);
  return merge(results, escapedTokens);
}

//===----------------------------------------------------------------------===//
// WaitState
//===----------------------------------------------------------------------===//

void WaitOpInfo::print(llvm::raw_ostream &os) const {
  os << "{counts: " << counts
     << ", waited_tokens: " << llvm::interleaved_array(waitedTokens)
     << ", implied_tokens: " << llvm::interleaved_array(impliedTokens) << "}";
}

void WaitState::print(raw_ostream &os) const {
  if (isEmpty()) {
    os << "<Empty>";
    return;
  }
  ArrayRef<TokenState> tokens = reachingTokens;
  os << "unhandled tokens = " << llvm::interleaved_array(tokens);
  if (!waitOpInfo.has_value()) {
    return;
  }
  os << ", wait information = " << *waitOpInfo;
}

ChangeResult WaitState::join(const WaitState &lattice) {
  assert(!waitOpInfo.has_value() &&
         "this join should not be called on wait ops");

  if (lattice.isEmpty())
    return ChangeResult::NoChange;

  if (isEmpty()) {
    reachingTokens = lattice.reachingTokens;
    return ChangeResult::Change;
  }
  return merge(reachingTokens, lattice.reachingTokens) ? ChangeResult::Change
                                                       : ChangeResult::NoChange;
}

ChangeResult
WaitState::joinWait(ValueRange deps, const WaitState &before,
                    WaitCnt waitCounts,
                    llvm::function_ref<TokenState(Value)> getState) {
  LDBG() << "  Merging wait: reaching="
         << llvm::interleaved_array(before.reachingTokens)
         << " deps=" << llvm::interleaved_array(deps) << " cnt=" << waitCounts;

  if (!waitOpInfo)
    waitOpInfo = WaitOpInfo(waitCounts);
  else {
    waitOpInfo->counts = waitCounts;
    waitOpInfo->waitedTokens.clear();
    waitOpInfo->impliedTokens.clear();
  }

  SmallVector<TokenState> newReachingToks;
  waitOpInfo->counts.handleWait(
      before.reachingTokens, deps, waitOpInfo->waitedTokens,
      waitOpInfo->impliedTokens, newReachingToks, getState);

  bool changed = reachingTokens != newReachingToks;
  if (changed)
    reachingTokens = std::move(newReachingToks);
  LDBG() << "  Wait result: " << *waitOpInfo;
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

ChangeResult WaitState::addTokens(ArrayRef<TokenState> tokens) {
  for (TokenState &token : reachingTokens)
    ++token;
  return merge(reachingTokens, tokens) ? ChangeResult::Change
                                       : ChangeResult::NoChange;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::WaitState)

//===----------------------------------------------------------------------===//
// WaitAnalysis
//===----------------------------------------------------------------------===//

MemoryInstructionKind WaitAnalysis::getMemoryKindFromToken(Value token) {
  if (auto readToken = dyn_cast<ReadTokenType>(token.getType()))
    return readToken.getKind();
  if (auto writeToken = dyn_cast<WriteTokenType>(token.getType()))
    return writeToken.getKind();
  return MemoryInstructionKind::None;
}

TokenState WaitAnalysis::createTokenState(Value token) {
  auto [it, inserted] = tokenIDs.try_emplace(token, tokenIDs.size());
  assert(it->second != TokenState::kUnknownID &&
         "token ID conflicts with unknown ID");
  return TokenState(token, it->second, getMemoryKindFromToken(token), 0);
}

LogicalResult WaitAnalysis::handleWaitOp(WaitOp waitOp, const WaitState &before,
                                         WaitState *after) {
  auto getState = [&](Value token) { return createTokenState(token); };
  propagateIfChanged(after, after->joinWait(waitOp.getDependencies(), before,
                                            WaitCnt::fromOp(waitOp), getState));
  return success();
}

LogicalResult WaitAnalysis::handleOp(Operation *op, const WaitState &before,
                                     WaitState *after) {
  ChangeResult changed = after->join(before);
  SmallVector<TokenState> producedTokens;
  for (OpResult result : op->getResults()) {
    if (getMemoryKindFromToken(result) == MemoryInstructionKind::None)
      continue;
    producedTokens.push_back(createTokenState(result));
  }
  if (!producedTokens.empty()) {
    llvm::sort(producedTokens);
    producedTokens.erase(llvm::unique(producedTokens), producedTokens.end());
    changed = after->addTokens(producedTokens) | changed;
  }
  propagateIfChanged(after, changed);
  return success();
}

LogicalResult WaitAnalysis::visitOperation(Operation *op,
                                           const WaitState &before,
                                           WaitState *after) {
  LDBG() << "Visit op: " << OpWithFlags(op, OpPrintingFlags().skipRegions())
         << " before: " << before;

  if (auto waitOp = dyn_cast<WaitOp>(op))
    return handleWaitOp(waitOp, before, after);

  return handleOp(op, before, after);
}

/// Filter tokens by dominance, moving non-dominating tokens to escapedTokens.
bool WaitAnalysis::filterByDominance(
    SmallVectorImpl<TokenState> &results, SmallVectorImpl<TokenState> &scratch,
    SmallVectorImpl<TokenState> &escapedTokens,
    ArrayRef<TokenState> predecessorTokens,
    llvm::function_ref<bool(Value)> dominates) {
  scratch.clear();
  scratch.reserve(predecessorTokens.size());
  for (const TokenState &tok : predecessorTokens) {
    if (tok.getID() == TokenState::kUnknownID || dominates(tok.getToken()))
      scratch.push_back(tok);
    else
      escapedTokens.push_back(tok);
  }
  return merge(results, scratch);
}

/// Map control-flow operands to successor values, removing mapped tokens from
/// escapedTokens.
bool WaitAnalysis::mapControlFlowOperands(
    SmallVectorImpl<TokenState> &results, SmallVectorImpl<TokenState> &scratch,
    SmallVectorImpl<TokenState> &escapedTokens,
    ArrayRef<TokenState> predecessorTokens, ValueRange operands,
    ValueRange successorValues) {
  scratch.clear();
  scratch.reserve(operands.size());
  for (auto [operand, value] : llvm::zip_equal(operands, successorValues)) {
    auto it = llvm::find_if(predecessorTokens, [&](const TokenState &s) {
      return s.getToken() == operand;
    });
    if (it == predecessorTokens.end())
      continue;
    // Remove from escaped tokens those that flow through control-flow.
    if (auto lb = llvm::find(escapedTokens, *it); lb != escapedTokens.end())
      *lb = TokenState();
    // Create new token state with new value but preserving kind/position.
    auto [idIt, _] = tokenIDs.try_emplace(value, tokenIDs.size());
    scratch.push_back(
        TokenState(value, idIt->second, it->getKind(), it->getPosition()));
  }
  llvm::sort(scratch);
  return merge(results, scratch);
}

void WaitAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                      Block *predecessor,
                                      const WaitState &before,
                                      WaitState *after) {
  LDBG() << "Visit block: " << block << " before: " << before;
  auto terminator = cast<BranchOpInterface>(predecessor->getTerminator());
  bool changed = false;
  SmallVector<TokenState> scratch;
  SmallVector<TokenState> &tokens = after->reachingTokens;
  escapedTokens.clear();

  // Filter tokens by dominance.
  auto dominates = [&](Value v) {
    return domInfo.properlyDominates(getDefiningBlock(v), block);
  };
  changed |= filterByDominance(tokens, scratch, escapedTokens,
                               before.reachingTokens, dominates);

  // Map operands flowing through branch.
  for (auto [i, succ] : llvm::enumerate(terminator->getSuccessors())) {
    if (succ != block)
      continue;
    changed |= mapControlFlowOperands(
        tokens, scratch, escapedTokens, before.reachingTokens,
        terminator.getSuccessorOperands(i).getForwardedOperands(),
        block->getArguments());
  }

  changed |= WaitCnt::handleEscapedTokens(tokens, escapedTokens);
  propagateIfChanged(after,
                     changed ? ChangeResult::Change : ChangeResult::NoChange);
}

void WaitAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const WaitState &before,
    WaitState *after) {
  LDBG() << "Visit branch: "
         << OpWithFlags(branch, OpPrintingFlags().skipRegions())
         << " from=" << (regionFrom ? (int)*regionFrom : -1)
         << " to=" << (regionTo ? (int)*regionTo : -1) << " before: " << before;

  bool changed = false;
  ArrayRef<TokenState> predecessorTokens = before.reachingTokens;
  SmallVector<TokenState> scratch;
  SmallVector<TokenState> &tokens = after->reachingTokens;
  escapedTokens.clear();

  RegionSuccessor successor =
      regionTo ? RegionSuccessor(&branch->getRegion(*regionTo))
               : RegionSuccessor::parent();

  // Filter tokens by dominance.
  auto dominates = [&](Value v) {
    if (successor.isParent())
      return domInfo.dominates(v, branch);
    return domInfo.properlyDominates(getDefiningBlock(v),
                                     &successor.getSuccessor()->front());
  };
  changed |= filterByDominance(tokens, scratch, escapedTokens,
                               predecessorTokens, dominates);

  // Map operands through region branch.
  auto mapBranchOperands = [&](RegionBranchPoint point) {
    if (!successor.isParent() && successor.getSuccessor()->empty())
      return;
    changed |= mapControlFlowOperands(
        tokens, scratch, escapedTokens, predecessorTokens,
        branch.getSuccessorOperands(point, successor),
        branch.getSuccessorInputs(successor));
  };

  if (!regionFrom) {
    mapBranchOperands(RegionBranchPoint::parent());
  } else {
    for (Block &block : branch->getRegion(*regionFrom)) {
      if (!block.empty())
        if (auto term =
                dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
          mapBranchOperands(RegionBranchPoint(term));
    }
  }

  changed |= WaitCnt::handleEscapedTokens(tokens, escapedTokens);
  propagateIfChanged(after,
                     changed ? ChangeResult::Change : ChangeResult::NoChange);
}

void WaitAnalysis::setToEntryState(WaitState *lattice) {
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

void WaitAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const WaitState &before, WaitState *after) {
  llvm_unreachable("inter-procedural analysis not supported");
}
