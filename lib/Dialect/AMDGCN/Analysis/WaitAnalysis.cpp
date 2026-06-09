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
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/IR/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseSet.h"
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

/// Map a token value to its hardware wait-counter kind. Returns nullopt if the
/// value is not a dependency token.
static std::optional<WaitCounterKind> getCounterKind(Value token) {
  Type type = token.getType();
  bool isWrite = false;
  MemoryInstructionKind kind;
  if (auto readToken = dyn_cast<ReadTokenType>(type)) {
    kind = readToken.getKind();
  } else if (auto writeToken = dyn_cast<WriteTokenType>(type)) {
    kind = writeToken.getKind();
    isWrite = true;
  } else {
    return std::nullopt;
  }
  switch (kind) {
  case MemoryInstructionKind::Flat:
    return isWrite ? WaitCounterKind::Store : WaitCounterKind::Load;
  case MemoryInstructionKind::Shared:
    return WaitCounterKind::Ds;
  case MemoryInstructionKind::Constant:
    return WaitCounterKind::Km;
  case MemoryInstructionKind::Tensor:
    return WaitCounterKind::Tensor;
  default:
    return std::nullopt;
  }
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

static StringRef kindName(WaitCounterKind kind, WaitCounterModel model) {
  if (model != WaitCounterModel::GFX12_50) {
    // CDNA: load+store share "flat", ds = "shared", km = "constant".
    switch (kind) {
    case WaitCounterKind::Load:
    case WaitCounterKind::Store:
      return "flat";
    case WaitCounterKind::Ds:
      return "shared";
    case WaitCounterKind::Km:
      return "constant";
    default:
      break;
    }
  }
  return stringifyWaitCounterKind(kind);
}

void TokenState::print(raw_ostream &os, WaitCounterModel model) const {
  os << "{";
  if (token) {
    token.printAsOperand(os, OpPrintingFlags());
    os << ", " << id;
  } else {
    os << "<escaped>";
  }
  os << ", " << position;
  os << ", " << kindName(kind, model);
  os << "}";
}

TokenState &TokenState::operator++() {
  if (position < kMaxPosition)
    ++position;
  return *this;
}

//===----------------------------------------------------------------------===//
// WaitCounterModel
//===----------------------------------------------------------------------===//

WaitCounterModel mlir::aster::amdgcn::getWaitCounterModel(ISAVersion isa) {
  switch (isa) {
  case ISAVersion::CDNA3:
    return WaitCounterModel::CDNA3;
  case ISAVersion::CDNA4:
    return WaitCounterModel::CDNA4;
  case ISAVersion::GFX12_50:
    return WaitCounterModel::GFX12_50;
  default:
    llvm_unreachable("unsupported ISA for WaitCounterModel");
  }
}

WaitCounterModel mlir::aster::amdgcn::getWaitCounterModelForOp(Operation *op) {
  auto moduleOp = dyn_cast<amdgcn::ModuleOp>(op);
  if (!moduleOp)
    moduleOp = op->getParentOfType<amdgcn::ModuleOp>();
  return moduleOp ? getWaitCounterModel(getIsaForTarget(moduleOp.getTarget()))
                  : WaitCounterModel::CDNA3;
}

//===----------------------------------------------------------------------===//
// WaitCnt
//===----------------------------------------------------------------------===//

/// Return all counter kinds that share the same physical hardware counter as
/// `kind` (including `kind` itself). On CDNA4, load+store share vmcnt and
/// ds+km share lgkmcnt. On GFX12_50 every kind is independent.
static llvm::SmallDenseSet<WaitCounterKind, 4>
aliasingCounters(WaitCounterKind kind, WaitCounterModel model) {
  llvm::SmallDenseSet<WaitCounterKind, 4> result;
  if (model == WaitCounterModel::GFX12_50) {
    result.insert(kind);
    return result;
  }
  switch (kind) {
  case WaitCounterKind::Vm:
  case WaitCounterKind::Load:
  case WaitCounterKind::Store:
    result.insert(WaitCounterKind::Load);
    result.insert(WaitCounterKind::Store);
    break;
  case WaitCounterKind::Lgkm:
  case WaitCounterKind::Ds:
  case WaitCounterKind::Km:
    result.insert(WaitCounterKind::Ds);
    result.insert(WaitCounterKind::Km);
    break;
  case WaitCounterKind::Tensor:
    result.insert(WaitCounterKind::Tensor);
    break;
  }
  return result;
}

WaitCnt WaitCnt::fromOp(WaitCntOpInterface waitOp, WaitCounterModel model) {
  WaitCnt c;
  // Query every counter kind the op might carry. For each reported value,
  // compute the min across the value and all existing aliasing counters (since
  // they share a physical counter), then set all aliases to that min.
  for (size_t i = 0; i < kNumKinds; ++i) {
    auto opKind = static_cast<WaitCounterKind>(i);
    uint16_t v = waitOp.getCounterValue(opKind);
    if (v == WaitCntOpInterface::kNoWaitCount)
      continue;
    auto aliases = aliasingCounters(opKind, model);
    Position mn = static_cast<Position>(v);
    for (WaitCounterKind k : aliases)
      mn = std::min<int32_t>(mn, c.getCount(k));
    for (WaitCounterKind k : aliases)
      c.updateCount(k, mn);
  }
  return c;
}

void WaitCnt::updateCount(ArrayRef<TokenState> tokens) {
  for (const TokenState &tok : tokens)
    updateCount(tok.getKind(), tok.getPosition());
}

void WaitCnt::print(llvm::raw_ostream &os, WaitCounterModel model) const {
  auto p = [&](const char *name, WaitCounterKind kind, bool sep) {
    os << name << ": ";
    int32_t c = getCount(kind);
    if (c == kMaxPosition)
      os << "nowait";
    else
      os << c;
    if (sep)
      os << ", ";
  };
  os << "{";
  if (model == WaitCounterModel::GFX12_50) {
    p("load_cnt", WaitCounterKind::Load, true);
    p("store_cnt", WaitCounterKind::Store, true);
    p("ds_cnt", WaitCounterKind::Ds, true);
    p("km_cnt", WaitCounterKind::Km, true);
    p("tensor_cnt", WaitCounterKind::Tensor, false);
  } else {
    p("vm_cnt", WaitCounterKind::Load, true);
    p("lgkm_cnt", WaitCounterKind::Ds, false);
  }
  os << "}";
}

void WaitCnt::handleWait(ArrayRef<TokenState> reachingTokens,
                         ValueRange dependencies,
                         SmallVectorImpl<TokenState> &waitedTokens,
                         SmallVectorImpl<TokenState> &impliedTokens,
                         SmallVectorImpl<TokenState> &nextReachingTokens,
                         llvm::function_ref<TokenState(Value)> getState,
                         WaitCounterModel model) {
  // Clear the waited tokens.
  waitedTokens.clear();

  bool hasLgkmDeps = false;

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

    // Count the number of DS and SMem tokens (the lgkm group: ds + km kinds).
    if (tok.getKind() == WaitCounterKind::Km ||
        tok.getKind() == WaitCounterKind::Ds) {
      hasLgkmDeps = true;
    }
  }

  bool hasSmemToks = false;

  // If there are LGKM tokens, check whether any SMEM (km) tokens are reaching.
  if (hasLgkmDeps) {
    for (const TokenState &tok : reachingTokens) {
      // End early if a km token has been found.
      if (hasSmemToks)
        break;

      if (tok.getKind() == WaitCounterKind::Km)
        hasSmemToks = true;
    }
  }

  // SMEM (km) returns out of order, so the only consistent wait is to drain
  // the km counter fully. All aliasing counters must also drain.
  if (hasSmemToks) {
    for (WaitCounterKind k : aliasingCounters(WaitCounterKind::Km, model))
      updateCount(k, 0);
  }

  // Update the wait counts based on the waited tokens.
  updateCount(waitedTokens);

  // Collapse aliasing counters to min: kinds sharing a physical counter must
  // report the same (tightest) wait count.
  for (size_t i = 0; i < kNumKinds; ++i) {
    auto kind = static_cast<WaitCounterKind>(i);
    auto aliases = aliasingCounters(kind, model);
    if (aliases.size() <= 1)
      continue;
    Position mn = kMaxPosition;
    for (WaitCounterKind k : aliases)
      mn = std::min<int32_t>(mn, getCount(k));
    for (WaitCounterKind k : aliases)
      updateCount(k, mn);
  }

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

//===----------------------------------------------------------------------===//
// WaitState
//===----------------------------------------------------------------------===//

void WaitOpInfo::print(llvm::raw_ostream &os, WaitCounterModel model) const {
  os << "{counts: ";
  counts.print(os, model);
  os << ", waited_tokens: [";
  llvm::interleave(
      waitedTokens, os, [&](const TokenState &t) { t.print(os, model); }, ", ");
  os << "], implied_tokens: [";
  llvm::interleave(
      impliedTokens, os, [&](const TokenState &t) { t.print(os, model); },
      ", ");
  os << "]}";
}

void WaitState::print(raw_ostream &os, WaitCounterModel model) const {
  if (isEmpty()) {
    os << "<Empty>";
    return;
  }
  os << "unhandled tokens = [";
  llvm::interleave(
      reachingTokens, os, [&](const TokenState &t) { t.print(os, model); },
      ", ");
  os << "]";
  if (!waitOpInfo.has_value())
    return;
  os << ", wait information = ";
  waitOpInfo->print(os, model);
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

ChangeResult WaitState::joinWait(ValueRange deps, const WaitState &before,
                                 WaitCnt waitCounts,
                                 llvm::function_ref<TokenState(Value)> getState,
                                 WaitCounterModel model) {
  // Get a fingerprint for change detection.
  auto oldFingerprint = getStateFingerprint(*this);
  LDBG_OS([&](raw_ostream &os) {
    os << "  Merging wait dependencies:\n";
    os << "  Reaching tokens: "
       << llvm::interleaved_array(before.reachingTokens) << "\n";
    os << "  Wait dependencies: " << llvm::interleaved_array(deps) << "\n";
    os << "  Wait counts: " << waitCounts;
  });

  // Update or create the wait op info.
  if (waitOpInfo.has_value()) {
    waitOpInfo->counts = waitCounts;
    waitOpInfo->waitedTokens.clear();
    waitOpInfo->impliedTokens.clear();
  } else {
    waitOpInfo = WaitOpInfo(waitCounts);
  }

  // Compute the new reaching tokens after the wait.
  SmallVector<TokenState> newReachingToks;
  waitOpInfo->counts.handleWait(
      before.reachingTokens, deps, waitOpInfo->waitedTokens,
      waitOpInfo->impliedTokens, newReachingToks, getState, model);
  bool changed = oldFingerprint != getStateFingerprint(*this);

  // Update the reaching tokens.
  if (reachingTokens != newReachingToks) {
    changed = true;
    reachingTokens = std::move(newReachingToks);
  }
  LDBG() << "  Wait information: " << *waitOpInfo;
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

ChangeResult WaitState::addTokens(ArrayRef<TokenState> tokens,
                                  WaitCounterModel model) {
  // Determine which counter kinds are affected (directly or via aliasing).
  bool affected[WaitCnt::kNumKinds] = {};
  for (const TokenState &tok : tokens)
    for (WaitCounterKind k : aliasingCounters(tok.getKind(), model))
      affected[static_cast<size_t>(k)] = true;

  // Only increment positions of reaching tokens whose counter kind is affected;
  // independent counters do not interfere.
  for (TokenState &token : reachingTokens)
    if (affected[static_cast<size_t>(token.getKind())])
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

/// Handle escaped tokens by converting them to unknown tokens at their
/// dominant positions, and merging them into the results.
static bool handleEscapedTokens(SmallVectorImpl<TokenState> &results,
                                SmallVectorImpl<TokenState> &escapedTokens) {
  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "  Handling escaped tokens: "
       << llvm::interleaved_array(escapedTokens);
  });
  // One escaped position per counter kind, indexed by WaitCounterKind.
  std::array<int32_t, WaitCnt::kNumKinds> cnt;
  cnt.fill(static_cast<int32_t>(TokenState::kMaxPosition));
  auto getEscCnt = [&](WaitCounterKind kind) -> int32_t & {
    int32_t i = static_cast<int32_t>(kind);
    assert(i >= 0 && i < static_cast<int32_t>(WaitCnt::kNumKinds) &&
           "invalid wait counter kind");
    return cnt[i];
  };
  for (TokenState &tok : escapedTokens) {
    // mapControlFlowOperands nulls out matched tokens with TokenState().
    // Skip these sentinels (id == -1 is not a valid token ID).
    if (tok.getID() == -1)
      continue;
    getEscCnt(tok.getKind()) =
        std::min(getEscCnt(tok.getKind()), tok.getPosition());
  }
  escapedTokens.clear();
  for (size_t i = 0; i < WaitCnt::kNumKinds; ++i) {
    auto kind = static_cast<WaitCounterKind>(i);
    if (getEscCnt(kind) != TokenState::kMaxPosition)
      escapedTokens.push_back(TokenState::unknown(kind, getEscCnt(kind)));
  }
  llvm::sort(escapedTokens);
  return merge(results, escapedTokens);
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

TokenState WaitAnalysis::getState(Value token, TokenState::ID position) {
  std::optional<WaitCounterKind> kind = getCounterKind(token);
  assert(kind && "expected a dependency token");
  return TokenState(token, getID(token), *kind, position);
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

  // Handle a wait op (amdgcn.wait or amdgcn.wait_gfx1250) uniformly via the
  // interface. Both ops' only operands are their token dependencies.
  if (auto waitOp = dyn_cast<WaitCntOpInterface>(op)) {
    auto getState = [&](Value token) { return this->getState(token, 0); };
    propagateIfChanged(after, after->joinWait(op->getOperands(), before,
                                              WaitCnt::fromOp(waitOp, model),
                                              getState, model));
    return success();
  }

  // Handle other operations.
  ChangeResult changed = after->join(before);
  SmallVector<TokenState> producedTokens;

  // Collect produced tokens.
  for (OpResult result : op->getResults()) {
    if (!getCounterKind(result))
      continue;
    producedTokens.push_back(getState(result, 0));
  }

  // Add produced tokens to the reaching set.
  if (!producedTokens.empty()) {
    llvm::sort(producedTokens);
    producedTokens.erase(llvm::unique(producedTokens), producedTokens.end());
    changed = after->addTokens(producedTokens, model) | changed;
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
  changed |= handleEscapedTokens(tokens, escapedTokens);
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
  changed |= handleEscapedTokens(tokens, escapedTokens);
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
