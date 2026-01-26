//===- WaitAnalysis.cpp - Wait dependency analysis ------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/WaitAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"

#define DEBUG_TYPE "wait-analysis"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

/// Merge two sets of TokenState, modifying target. Returns whether the target
/// set was changed. This assumes both sets are sorted.
static bool merge(SmallVectorImpl<TokenState> &target,
                  ArrayRef<TokenState> source) {
  // Early exit if source is empty.
  if (source.empty())
    return false;

  // Early exit if target is empty.
  if (target.empty()) {
    target.append(source.begin(), source.end());
    return true;
  }

  // Early exit if ranges don't overlap, but still need to append source
  if (target.back() < source.front()) {
    target.append(source.begin(), source.end());
    return true;
  }
  if (source.back() < target.front()) {
    target.insert(target.begin(), source.begin(), source.end());
    return true;
  }

  size_t oldSize = target.size();
  SmallVector<TokenState> temp;
  temp.reserve(target.size() + source.size());
  int64_t i = 0, j = 0, n = target.size(), m = source.size();
  bool changed = false;
  while (i < n) {
    if (j >= m) {
      temp.push_back(target[i++]);
      continue;
    }
    if (target[i] < source[j]) {
      temp.push_back(target[i++]);
    } else if (source[j] < target[i]) {
      temp.push_back(source[j++]);
    } else {
      TokenState merged = target[i++];
      changed |= merged.merge(source[j++]);
      temp.push_back(merged);
    }
  }
  while (j < m)
    temp.push_back(source[j++]);
  target = std::move(temp);
  return changed || target.size() != oldSize;
}

/// Get a fingerprint of the wait state for change detection.
static std::tuple<int32_t, int32_t, int32_t, WaitCnt>
getStateFingerprint(const WaitState &state) {
  const std::optional<WaitOpInfo> &info = state.waitOpInfo;
  return std::tuple<int32_t, int32_t, int32_t, WaitCnt>(
      state.reachingTokens.size(), info ? info->waitedTokens.size() : -1,
      info ? info->impliedTokens.size() : -1, info ? info->counts : WaitCnt());
}

/// Get the memory instruction kind from a token value's type.
/// Returns MemoryInstructionKind::None if the value is not a token type
static MemoryInstructionKind getMemoryKindFromToken(Value token) {
  Type type = token.getType();
  if (auto readToken = dyn_cast<ReadTokenType>(type))
    return readToken.getKind();
  if (auto writeToken = dyn_cast<WriteTokenType>(type))
    return writeToken.getKind();
  return MemoryInstructionKind::None;
}

//===----------------------------------------------------------------------===//
// TokenState
//===----------------------------------------------------------------------===//

bool TokenState::merge(const TokenState &other) {
  assert(token == other.token && "cannot merge different tokens");
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
  // Clear the waited tokens.
  waitedTokens.clear();

  int32_t numLgkmToks = 0;

  // Compute which dependencies are in the reaching set.
  for (Value v : dependencies) {
    TokenState tok = getState(v);
    auto lb = llvm::lower_bound(reachingTokens, tok);
    if (lb == reachingTokens.end() || lb->getToken() != v) {
      LDBG_OS([&](raw_ostream &os) {
        os << "  Wait dependency: ";
        v.printAsOperand(os, OpPrintingFlags());
        os << " not in the reaching set";
      });
      continue;
    }
    waitedTokens.push_back(*lb);

    // Count the number of DS and SMem tokens.
    switch (tok.getKind()) {
    case MemoryInstructionKind::Constant:
    case MemoryInstructionKind::Shared:
      numLgkmToks++;
      break;
    default:
      break;
    }
  }

  int32_t numDSToks = 0;
  int32_t numSMemToks = 0;
  // If there are LGKM tokens, count the number of DS and SMEM tokens in the
  // reaching tokens.
  if (numLgkmToks > 0) {
    for (const TokenState &tok : reachingTokens) {
      switch (tok.getKind()) {
      case MemoryInstructionKind::Constant:
        numSMemToks++;
        break;
      case MemoryInstructionKind::Shared:
        numDSToks++;
        break;
      default:
        break;
      }
    }
  }

  // If there are both DS and SMEM tokens, the only consistent wait is to wait
  // for all lgkm tokens.
  if (numDSToks > 0 && numSMemToks > 0) {
    lgkmCnt = 0;
  }

  // Update the wait counts based on the waited tokens.
  updateCount(waitedTokens);

  // Invalidate tokens that are dominated by the wait counts.
  for (TokenState &token : waitedTokens) {
    int32_t count = getCount(token.getKind());
    // Skip tokens that match the wait count and are greater than zero.
    if (token.getPosition() <= count && count > 0)
      continue;
    LDBG_OS([&](raw_ostream &os) {
      if (token.getToken() == nullptr)
        return;
      os << "  Invalidating dependency token: ";
      token.getToken().printAsOperand(os, OpPrintingFlags());
    });
    token = TokenState();
  }

  // Remove invalidated tokens.
  waitedTokens.erase(
      std::remove(waitedTokens.begin(), waitedTokens.end(), TokenState()),
      waitedTokens.end());

  // Compute implied tokens and next reaching tokens.
  for (const TokenState &token : reachingTokens) {
    int32_t count = getCount(token.getKind());
    // Preserve tokens that are not waited on.
    if (token.getPosition() < count)
      nextReachingTokens.push_back(token);
    // Collect implied tokens.
    if (token.getPosition() >= count)
      impliedTokens.push_back(token);
  }
}

bool WaitCnt::handleEscapedTokens(SmallVectorImpl<TokenState> &results,
                                  SmallVectorImpl<TokenState> &escapedTokens) {
  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "  Handling escaped tokens: "
       << llvm::interleaved_array(escapedTokens);
  });
  std::array<int32_t, 4> cnt = {
      static_cast<int32_t>(TokenState::kMaxPosition),
      static_cast<int32_t>(TokenState::kMaxPosition),
      static_cast<int32_t>(TokenState::kMaxPosition),
      static_cast<int32_t>(TokenState::kMaxPosition),
  };
  auto getEscCnt = [&](MemoryInstructionKind kind) -> int32_t & {
    int32_t i = static_cast<int32_t>(kind);
    assert(i >= 0 && i < 4 && "invalid memory kind");
    return cnt[i];
  };
  for (TokenState &tok : escapedTokens) {
    if (tok.getKind() == MemoryInstructionKind::None)
      continue;
    getEscCnt(tok.getKind()) =
        std::min(getEscCnt(tok.getKind()), tok.getPosition());
  }
  escapedTokens.clear();
  if (getEscCnt(MemoryInstructionKind::Flat) != TokenState::kMaxPosition)
    escapedTokens.push_back(
        TokenState::unknownVMem(getEscCnt(MemoryInstructionKind::Flat)));
  if (getEscCnt(MemoryInstructionKind::Constant) != TokenState::kMaxPosition)
    escapedTokens.push_back(
        TokenState::unknownSMem(getEscCnt(MemoryInstructionKind::Constant)));
  if (getEscCnt(MemoryInstructionKind::Shared) != TokenState::kMaxPosition)
    escapedTokens.push_back(
        TokenState::unknownDMem(getEscCnt(MemoryInstructionKind::Shared)));
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
  // Get a fingerprint for change detection.
  auto oldFingerprint = getStateFingerprint(*this);
  LDBG_OS([&](raw_ostream &os) {
    os << "  Merging wait dependencies:\n";
    os << "  Reaching tokens: "
       << llvm::interleaved_array(before.reachingTokens) << "\n";
    os << "  Wait dependencies: " << llvm::interleaved_array(deps) << "\n";
    os << "  Wait counts: " << waitCounts;
  });
  if (waitOpInfo.has_value()) {
    waitOpInfo->counts = waitCounts;
    waitOpInfo->waitedTokens.clear();
    waitOpInfo->impliedTokens.clear();
  } else {
    waitOpInfo = WaitOpInfo(waitCounts);
  }
  SmallVector<TokenState> newReachingToks;
  waitOpInfo->counts.handleWait(
      before.reachingTokens, deps, waitOpInfo->waitedTokens,
      waitOpInfo->impliedTokens, newReachingToks, getState);
  bool changed = oldFingerprint != getStateFingerprint(*this);
  if (reachingTokens != newReachingToks) {
    changed = true;
    reachingTokens = std::move(newReachingToks);
  }
  LDBG() << "  Wait information: " << *waitOpInfo;
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

#define DUMP_STATE_HELPER(name, obj, extra)                                    \
  LDBG_OS([&](raw_ostream &os) {                                               \
    os << "Visiting " name ": " << obj << "\n";                                \
    os << "  Incoming lattice: ";                                              \
    before.print(os);                                                          \
    extra                                                                      \
  });                                                                          \
  auto _atExit = llvm::make_scope_exit([&]() {                                 \
    LDBG_OS([&](raw_ostream &os) {                                             \
      os << "  Outgoing lattice: ";                                            \
      after->print(os);                                                        \
    });                                                                        \
  });

/// Check if a value's defining block dominates a given block.
static bool dominatesSuccessor(DominanceInfo &domInfo, Value value,
                               Block *block) {
  Block *defBlock = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    defBlock = blockArg.getOwner();
  } else {
    defBlock = value.getDefiningOp()->getBlock();
  }
  return domInfo.properlyDominates(defBlock, block);
}

/// Check if a value dominates a given succesor.
static bool dominatesSuccessor(DominanceInfo &domInfo, Value value,
                               RegionBranchOpInterface op,
                               RegionSuccessor successor) {
  if (successor.isParent())
    return domInfo.dominates(value, op);
  Block *defBlock = nullptr;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    defBlock = blockArg.getOwner();
  } else {
    defBlock = value.getDefiningOp()->getBlock();
  }
  return domInfo.properlyDominates(defBlock,
                                   &successor.getSuccessor()->front());
}

/// Add tokens from predecessor to results based on dominance.
static bool addTokensByDominance(SmallVectorImpl<TokenState> &results,
                                 SmallVectorImpl<TokenState> &scratch,
                                 SmallVectorImpl<TokenState> &escapedTokens,
                                 ArrayRef<TokenState> predecessorTokens,
                                 llvm::function_ref<bool(Value)> dominates) {
  scratch.reserve(predecessorTokens.size());
  for (const TokenState &tok : predecessorTokens) {
    if (tok.getID() == TokenState::kUnknownID) {
      // Unknown tokens always propagate.
      scratch.push_back(tok);
      continue;
    }

    // Only include tokens whose defining block dominates the successor.
    if (!dominates(tok.getToken())) {
      // Add a potentially escaped token.
      escapedTokens.push_back(tok);
      continue;
    }
    scratch.push_back(tok);
  }
  return merge(results, scratch);
}

/// Walk all terminators in a region and invoke a function on each.
static void
walkTerminators(Region *region,
                std::function<void(RegionBranchTerminatorOpInterface)> &&func) {
  for (Block &block : *region) {
    if (block.empty())
      continue;
    if (auto terminator =
            dyn_cast<RegionBranchTerminatorOpInterface>(block.back()))
      func(terminator);
  }
}

TokenState WaitAnalysis::getState(Value token, TokenState::ID position) {
  return TokenState(token, getID(token), getMemoryKindFromToken(token),
                    position);
}

bool WaitAnalysis::mapControlFlowOperands(
    SmallVectorImpl<TokenState> &results, SmallVectorImpl<TokenState> &scratch,
    SmallVectorImpl<TokenState> &escapedTokens,
    ArrayRef<TokenState> predecessorTokens, ValueRange successorOperands,
    ValueRange successorValues) {
  scratch.clear();
  scratch.reserve(successorOperands.size());
  for (auto operandValue :
       llvm::zip_equal(successorOperands, successorValues)) {
    Value operand = std::get<0>(operandValue);
    Value value = std::get<1>(operandValue);

    LDBG_OS([&](llvm::raw_ostream &os) {
      os << "  Checking propagated value from: ";
      operand.printAsOperand(os, OpPrintingFlags());
      os << " to ";
      value.printAsOperand(os, OpPrintingFlags());
    });

    // Find the token in predecessorTokens.
    auto it = llvm::find_if(predecessorTokens, [&](const TokenState &state) {
      return state.getToken() == operand;
    });
    if (it == predecessorTokens.end())
      continue;

    // Remove from escaped tokens those tokens that flow through control-flow.
    if (auto lb = llvm::lower_bound(escapedTokens, *it);
        lb != escapedTokens.end() && *lb == *it) {
      LDBG() << "  Removing escaped token: " << *lb;
      *lb = TokenState();
    }

    scratch.push_back(
        TokenState(value, getID(value), it->getKind(), it->getPosition()));
  }
  llvm::sort(scratch);
  return merge(results, scratch);
}

LogicalResult WaitAnalysis::visitOperation(Operation *op,
                                           const WaitState &before,
                                           WaitState *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()), {});

  // Handle a wait op.
  if (auto waitOp = dyn_cast<WaitOp>(op)) {
    auto getState = [&](Value token) { return this->getState(token, 0); };
    propagateIfChanged(after,
                       after->joinWait(waitOp.getDependencies(), before,
                                       WaitCnt::fromOp(waitOp), getState));
    return success();
  }

  // Handle other operations.
  ChangeResult changed = after->join(before);
  SmallVector<TokenState> producedTokens;

  // Collect produced tokens.
  for (OpResult result : op->getResults()) {
    if (getMemoryKindFromToken(result) == MemoryInstructionKind::None)
      continue;
    producedTokens.push_back(getState(result, 0));
  }

  // Add produced tokens to the reaching set.
  if (!producedTokens.empty()) {
    llvm::sort(producedTokens);
    producedTokens.erase(llvm::unique(producedTokens), producedTokens.end());
    changed = after->addTokens(producedTokens) | changed;
  }
  propagateIfChanged(after, changed);
  return success();
}

void WaitAnalysis::visitBlockTransfer(Block *block, ProgramPoint *point,
                                      Block *predecessor,
                                      const WaitState &before,
                                      WaitState *after) {
  DUMP_STATE_HELPER("block", block, {});
  auto terminator = cast<BranchOpInterface>(predecessor->getTerminator());
  bool changed = false;
  SmallVector<TokenState> scratch;
  SmallVector<TokenState> &tokens = after->reachingTokens;
  escapedTokens.clear();

  // Get tokens reaching the beginning of the block.
  changed |= addTokensByDominance(
      tokens, scratch, escapedTokens, before.reachingTokens,
      [&](Value v) { return dominatesSuccessor(domInfo, v, block); });
  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "  Initial escaped tokens: "
       << llvm::interleaved_array(escapedTokens);
  });

  // Propagate tokens from the predecessor to this block.
  for (auto [i, succ] : llvm::enumerate(terminator->getSuccessors())) {
    if (succ != block)
      continue;
    changed |= mapControlFlowOperands(
        tokens, scratch, escapedTokens, before.reachingTokens,
        terminator.getSuccessorOperands(i).getForwardedOperands(),
        block->getArguments());
  }

  // Handle escaped tokens.
  changed |= WaitCnt::handleEscapedTokens(tokens, escapedTokens);
  propagateIfChanged(after,
                     changed ? ChangeResult::Change : ChangeResult::NoChange);
}

void WaitAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const WaitState &before,
    WaitState *after) {
  DUMP_STATE_HELPER(
      "branch op", OpWithFlags(branch, OpPrintingFlags().skipRegions()), {
        os << "\n  Branching from: " << (regionFrom ? *regionFrom : -1)
           << " to " << (regionTo ? *regionTo : -1);
      });
  bool changed = false;
  ArrayRef<TokenState> predecessorTokens = before.reachingTokens;
  SmallVector<TokenState> scratch;
  SmallVector<TokenState> &tokens = after->reachingTokens;
  escapedTokens.clear();

  // Determine the successor.
  RegionSuccessor successor =
      regionTo ? RegionSuccessor(&branch->getRegion(*regionTo))
               : RegionSuccessor::parent();

  // Get the reaching tokens that are control-flow independent.
  changed |= addTokensByDominance(
      tokens, scratch, escapedTokens, predecessorTokens, [&](Value v) {
        return dominatesSuccessor(domInfo, v, branch, successor);
      });
  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "  Initial escaped tokens: "
       << llvm::interleaved_array(escapedTokens);
  });

  // Branch from parent.
  if (!regionFrom) {
    changed |= mapControlFlowOperands(
        tokens, scratch, escapedTokens, predecessorTokens,
        branch.getSuccessorOperands(RegionBranchPoint::parent(), successor),
        branch.getSuccessorInputs(successor));
  } else {
    // Branch from a region.
    walkTerminators(&branch->getRegion(*regionFrom),
                    [&](RegionBranchTerminatorOpInterface terminator) {
                      changed |= mapControlFlowOperands(
                          tokens, scratch, escapedTokens, predecessorTokens,
                          branch.getSuccessorOperands(
                              RegionBranchPoint(terminator), successor),
                          branch.getSuccessorInputs(successor));
                    });
  }

  // Handle escaped tokens.
  changed |= WaitCnt::handleEscapedTokens(tokens, escapedTokens);
  propagateIfChanged(after,
                     changed ? ChangeResult::Change : ChangeResult::NoChange);
}

void WaitAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const WaitState &before, WaitState *after) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()), {});
  assert(false && "we don't support inter-procedural analysis");
}

void WaitAnalysis::setToEntryState(WaitState *lattice) {
  auto fingerprint = getStateFingerprint(*lattice);
  lattice->reachingTokens.clear();
  lattice->waitOpInfo.reset();
  auto newFingerprint = getStateFingerprint(*lattice);
  propagateIfChanged(lattice, fingerprint == newFingerprint
                                  ? ChangeResult::NoChange
                                  : ChangeResult::Change);
}

#undef DUMP_STATE_HELPER
