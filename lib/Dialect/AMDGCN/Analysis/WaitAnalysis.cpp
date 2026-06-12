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
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/IR/Utils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include <optional>

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
  os << ", " << stringifyWaitCounterKind(kind);
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

static WaitCounterModel getWaitCounterModel(ISAVersion isa) {
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

ISAVersion mlir::aster::amdgcn::getIsaForOp(amdgcn::ModuleOp op) {
  if (!op || !op.getTargetAttr())
    return ISAVersion::CDNA3;
  return getIsaForTarget(op.getTarget());
}
//===----------------------------------------------------------------------===//
// WaitCnt
//===----------------------------------------------------------------------===//

static void printCount(llvm::raw_ostream &os, const char *name,
                       TokenState::Position c, bool sep) {
  os << name << ": ";
  if (c == TokenState::kMaxPosition)
    os << "nowait";
  else
    os << c;
  if (sep)
    os << ", ";
}

void WaitCntCdna3::print(llvm::raw_ostream &os) const {
  os << "{";
  printCount(os, "vm_cnt", vmcnt, true);
  printCount(os, "lgkm_cnt", lgkmcnt, false);
  os << "}";
}

void WaitCntGfx1250::print(llvm::raw_ostream &os) const {
  os << "{";
  printCount(os, "load_cnt", loadcnt, true);
  printCount(os, "store_cnt", storecnt, true);
  printCount(os, "ds_cnt", dscnt, true);
  printCount(os, "km_cnt", kmcnt, true);
  printCount(os, "tensor_cnt", tensorcnt, false);
  os << "}";
}

/// Find the dependency tokens that are in the reaching set, i.e. the tokens a
/// wait op actually waits on.
static SmallVector<TokenState>
findWaitedTokens(ArrayRef<TokenState> reaching, ValueRange deps,
                 llvm::function_ref<TokenState(Value)> getState) {
  SmallVector<TokenState> waited;
  for (Value v : deps) {
    TokenState tok = getState(v);
    auto lb = llvm::lower_bound(reaching, tok);
    if (lb == reaching.end() || lb->getToken() != v) {
      LDBG_OS([&](raw_ostream &os) {
        os << "  Wait dependency: ";
        v.printAsOperand(os, OpPrintingFlags());
        os << " not in the reaching set";
      });
      continue;
    }
    waited.push_back(*lb);
  }
  return waited;
}

/// Prune `waited` tokens that the final counts fully cover and split `reaching`
/// into implied (count-dominated, returned via `implied`) vs still-pending
/// (returned).
/// `countOf` maps a token to its physical counter's wait count to adapt to
/// different counter models.
static SmallVector<TokenState>
resolveReaching(ArrayRef<TokenState> reaching,
                llvm::function_ref<int32_t(const TokenState &)> countOf,
                SmallVectorImpl<TokenState> &waited,
                SmallVectorImpl<TokenState> &implied) {
  assert(implied.empty() &&
         "implied output must be cleared before resolveReaching");
  for (TokenState &token : waited) {
    int32_t count = countOf(token);
    if (token.getPosition() <= count && count > 0)
      continue;
    token = TokenState();
  }
  waited.erase(std::remove(waited.begin(), waited.end(), TokenState()),
               waited.end());
  SmallVector<TokenState> next;
  for (const TokenState &token : reaching) {
    int32_t count = countOf(token);
    if (token.getPosition() < count)
      next.push_back(token);
    if (token.getPosition() >= count)
      implied.push_back(token);
  }
  return next;
}

//===----------------------------------------------------------------------===//
// WaitState
//===----------------------------------------------------------------------===//

void WaitOpInfo::print(llvm::raw_ostream &os) const {
  os << "{counts: ";
  std::visit([&](const auto &c) { c.print(os); }, counts);
  os << ", waited_tokens: [";
  llvm::interleave(
      waitedTokens, os, [&](const TokenState &t) { t.print(os); }, ", ");
  os << "], implied_tokens: [";
  llvm::interleave(
      impliedTokens, os, [&](const TokenState &t) { t.print(os); }, ", ");
  os << "]}";
}

void WaitState::print(raw_ostream &os, ISAVersion isaVersion) const {
  (void)isaVersion;
  if (isEmpty()) {
    os << "<Empty>";
    return;
  }
  os << "unhandled tokens = [";
  llvm::interleave(
      reachingTokens, os, [&](const TokenState &t) { t.print(os); }, ", ");
  os << "]";
  if (!waitOpInfo.has_value())
    return;
  os << ", wait information = ";
  waitOpInfo->print(os);
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

MLIR_DEFINE_EXPLICIT_TYPE_ID(mlir::aster::amdgcn::WaitState)

//===----------------------------------------------------------------------===//
// WaitAnalysis
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// WaitAnalysisCdna -- two physical counters (vmcnt, lgkmcnt)
//===----------------------------------------------------------------------===//

// CDNA3/CDNA4 token kinds: vmem (Vm) -> vmcnt; ds (LDS) and scalar_read/km
// (scalar) -> lgkmcnt. ds and scalar_read stay distinct so only scalar reads
// force a full lgkm drain.
static bool isVmKind(WaitCounterKind k) { return k == WaitCounterKind::Vm; }
static bool isLgkmKind(WaitCounterKind k) {
  return k == WaitCounterKind::Ds || k == WaitCounterKind::Km ||
         k == WaitCounterKind::ScalarRead || k == WaitCounterKind::Lgkm;
}

WaitCnt WaitAnalysisCdna::countsFromWaitOp(WaitCntOpInterface waitOp) const {
  WaitCntCdna3 c;
  uint16_t vm = waitOp.getCounterValue(WaitCounterKind::Vm);
  if (vm != WaitCntOpInterface::kNoWaitCount)
    c.updateCount(WaitCounterKind::Vm, vm);
  uint16_t lgkm = waitOp.getCounterValue(WaitCounterKind::Lgkm);
  if (lgkm != WaitCntOpInterface::kNoWaitCount)
    c.updateCount(WaitCounterKind::Lgkm, lgkm);
  return c;
}

ChangeResult
WaitAnalysisCdna::advanceAndAdd(WaitState &state,
                                ArrayRef<TokenState> produced) const {
  // A produced vmem token advances every reaching vmem token (shared vmcnt); a
  // produced lgkm token advances every reaching lgkm token (shared lgkmcnt).
  bool bumpVm = false, bumpLgkm = false;
  for (const TokenState &t : produced) {
    bumpVm |= isVmKind(t.getKind());
    bumpLgkm |= isLgkmKind(t.getKind());
  }
  for (TokenState &t : state.reachingTokens) {
    if (bumpVm && isVmKind(t.getKind()))
      ++t;
    if (bumpLgkm && isLgkmKind(t.getKind()))
      ++t;
  }
  return merge(state.reachingTokens, produced) ? ChangeResult::Change
                                               : ChangeResult::NoChange;
}

ChangeResult WaitAnalysisCdna::transferWait(
    const WaitState &before, ValueRange deps, WaitCnt counts, WaitState &after,
    llvm::function_ref<TokenState(Value)> getState) const {
  std::optional<WaitOpInfo> oldInfo = after.waitOpInfo;
  if (after.waitOpInfo.has_value()) {
    after.waitOpInfo->counts = counts;
    after.waitOpInfo->waitedTokens.clear();
    after.waitOpInfo->impliedTokens.clear();
  } else {
    after.waitOpInfo = WaitOpInfo(counts);
  }
  WaitCntCdna3 &c = std::get<WaitCntCdna3>(after.waitOpInfo->counts);

  SmallVector<TokenState> waited =
      findWaitedTokens(before.reachingTokens, deps, getState);
  bool waitsLgkm = llvm::any_of(
      waited, [](const TokenState &t) { return isLgkmKind(t.getKind()); });
  bool outOfOrderReaching =
      llvm::any_of(before.reachingTokens, [&](const TokenState &t) {
        return drainBehavior(t.getKind()) == DrainBehavior::OutOfOrder;
      });

  // An out-of-order kind (scalar memory) outstanding on lgkmcnt makes a partial
  // lgkm wait unsound, so drain lgkmcnt fully. A pure in-order (ds) lgkm wait
  // keeps its partial count.
  if (waitsLgkm && outOfOrderReaching)
    c.updateCount(WaitCounterKind::Lgkm, 0);

  for (const TokenState &t : waited)
    c.updateCount(t.getKind(), t.getPosition());

  auto countOf = [&](const TokenState &t) { return c.getCount(t.getKind()); };
  SmallVector<TokenState> next = resolveReaching(
      before.reachingTokens, countOf, waited, after.waitOpInfo->impliedTokens);
  after.waitOpInfo->waitedTokens = std::move(waited);

  bool changed = !(oldInfo == after.waitOpInfo);
  if (after.reachingTokens != next) {
    changed = true;
    after.reachingTokens = std::move(next);
  }
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

//===----------------------------------------------------------------------===//
// WaitAnalysisGfx1250
//===----------------------------------------------------------------------===//

WaitCnt WaitAnalysisGfx1250::countsFromWaitOp(WaitCntOpInterface waitOp) const {
  WaitCntGfx1250 c;
  for (WaitCounterKind k : WaitCntGfx1250::kKinds) {
    uint16_t v = waitOp.getCounterValue(k);
    if (v != WaitCntOpInterface::kNoWaitCount)
      c.updateCount(k, v);
  }
  return c;
}

ChangeResult
WaitAnalysisGfx1250::advanceAndAdd(WaitState &state,
                                   ArrayRef<TokenState> produced) const {
  // Each counter is independent: a produced token advances only reaching tokens
  // of the same kind.
  DenseSet<WaitCounterKind> bumped;
  for (const TokenState &t : produced)
    bumped.insert(t.getKind());
  for (TokenState &t : state.reachingTokens)
    if (bumped.contains(t.getKind()))
      ++t;
  return merge(state.reachingTokens, produced) ? ChangeResult::Change
                                               : ChangeResult::NoChange;
}

ChangeResult WaitAnalysisGfx1250::transferWait(
    const WaitState &before, ValueRange deps, WaitCnt counts, WaitState &after,
    llvm::function_ref<TokenState(Value)> getState) const {
  // Snapshot the prior waitOpInfo for precise (content, not size) change
  // detection; the reaching set is the other, independent change dimension.
  std::optional<WaitOpInfo> oldInfo = after.waitOpInfo;
  if (after.waitOpInfo.has_value()) {
    after.waitOpInfo->counts = counts;
    after.waitOpInfo->waitedTokens.clear();
    after.waitOpInfo->impliedTokens.clear();
  } else {
    after.waitOpInfo = WaitOpInfo(counts);
  }
  WaitCntGfx1250 &c = std::get<WaitCntGfx1250>(after.waitOpInfo->counts);

  SmallVector<TokenState> waited =
      findWaitedTokens(before.reachingTokens, deps, getState);
  bool waitsKm = llvm::any_of(waited, [&](const TokenState &t) {
    return drainBehavior(t.getKind()) == DrainBehavior::OutOfOrder;
  });
  bool outOfOrderReaching =
      llvm::any_of(before.reachingTokens, [&](const TokenState &t) {
        return drainBehavior(t.getKind()) == DrainBehavior::OutOfOrder;
      });
  // An out-of-order kind (scalar_read) cannot be partially waited; drain its
  // counter fully.
  if (waitsKm && outOfOrderReaching)
    c.updateCount(WaitCounterKind::ScalarRead, 0);
  for (const TokenState &t : waited)
    c.updateCount(t.getKind(), t.getPosition());

  auto countOf = [&](const TokenState &t) { return c.getCount(t.getKind()); };
  SmallVector<TokenState> next = resolveReaching(
      before.reachingTokens, countOf, waited, after.waitOpInfo->impliedTokens);
  after.waitOpInfo->waitedTokens = std::move(waited);

  bool changed = !(oldInfo == after.waitOpInfo);
  if (after.reachingTokens != next) {
    changed = true;
    after.reachingTokens = std::move(next);
  }
  return changed ? ChangeResult::Change : ChangeResult::NoChange;
}

WaitAnalysisBase &mlir::aster::amdgcn::loadWaitAnalysis(DataFlowSolver &solver,
                                                        DominanceInfo &domInfo,
                                                        ISAVersion isaVersion) {
  switch (getWaitCounterModel(isaVersion)) {
  // CDNA3/4 have the same two-counter model.
  case WaitCounterModel::CDNA3:
  case WaitCounterModel::CDNA4:
    return *solver.load<WaitAnalysisCdna>(domInfo);
  case WaitCounterModel::GFX12_50:
    return *solver.load<WaitAnalysisGfx1250>(domInfo);
  }
  llvm_unreachable("unhandled wait counter model");
}

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
  // Keep the dominant (min) position per escaped token kind. Tokens may carry
  // any model's kinds (CDNA: vm/ds/km; gfx1250: load/store/ds/km/tensor), so
  // key by kind directly rather than a fixed slot range.
  DenseMap<WaitCounterKind, int32_t> minPos;
  for (TokenState &tok : escapedTokens) {
    // mapControlFlowOperands nulls out matched tokens with TokenState().
    // Skip these sentinels (id == -1 is not a valid token ID).
    if (tok.getID() == -1)
      continue;
    auto [it, inserted] = minPos.try_emplace(tok.getKind(), tok.getPosition());
    if (!inserted)
      it->second =
          std::min(it->second, static_cast<int32_t>(tok.getPosition()));
  }
  escapedTokens.clear();
  for (auto [kind, pos] : minPos)
    escapedTokens.push_back(TokenState::unknown(kind, pos));
  // Sort restores a deterministic order (DenseMap iteration is unordered).
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

TokenState WaitAnalysisBase::getState(Value token, TokenState::ID position) {
  std::optional<WaitCounterKind> kind = this->getCounterKind(token);
  assert(kind && "expected a dependency token");
  return TokenState(token, getID(token), *kind, position);
}

bool WaitAnalysisBase::mapControlFlowOperands(
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
    // Linear scan, not lower_bound: entries are nulled in place below, so the
    // set is not kept ordered, and correctness must not depend on the
    // counter-kind enum ordering.
    auto esc = llvm::find_if(escapedTokens,
                             [&](const TokenState &s) { return s == *it; });
    if (esc != escapedTokens.end()) {
      LDBG() << "  Removing escaped token: " << *esc;
      *esc = TokenState();
    }

    scratch.push_back(
        TokenState(value, getID(value), it->getKind(), it->getPosition()));
  }
  llvm::sort(scratch);
  return merge(results, scratch);
}

LogicalResult WaitAnalysisBase::visitOperation(Operation *op,
                                               const WaitState &before,
                                               WaitState *after) {
  DUMP_STATE_HELPER("op", OpWithFlags(op, OpPrintingFlags().skipRegions()), {});

  // A wait op (amdgcn.wait or amdgcn.wait_gfx1250) is handled by the model's
  // transferWait. Both ops' only operands are their token dependencies.
  if (auto waitOp = dyn_cast<WaitCntOpInterface>(op)) {
    auto getSt = [&](Value token) { return this->getState(token, 0); };
    propagateIfChanged(after,
                       transferWait(before, op->getOperands(),
                                    countsFromWaitOp(waitOp), *after, getSt));
    return success();
  }

  // Handle other operations.
  ChangeResult changed = after->join(before);
  SmallVector<TokenState> producedTokens;

  // Collect produced tokens.
  for (OpResult result : op->getResults()) {
    if (!this->getCounterKind(result))
      continue;
    producedTokens.push_back(getState(result, 0));
  }

  // Add produced tokens to the reaching set.
  if (!producedTokens.empty()) {
    llvm::sort(producedTokens);
    producedTokens.erase(llvm::unique(producedTokens), producedTokens.end());
    changed = advanceAndAdd(*after, producedTokens) | changed;
  }
  propagateIfChanged(after, changed);
  return success();
}

void WaitAnalysisBase::visitBlockTransfer(Block *block, ProgramPoint *point,
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

void WaitAnalysisBase::visitRegionBranchControlFlowTransfer(
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

void WaitAnalysisBase::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const WaitState &before, WaitState *after) {
  DUMP_STATE_HELPER("call op",
                    OpWithFlags(call, OpPrintingFlags().skipRegions()), {});
  assert(false && "we don't support inter-procedural analysis");
}

void WaitAnalysisBase::setToEntryState(WaitState *lattice) {
  auto fingerprint = getStateFingerprint(*lattice);
  lattice->reachingTokens.clear();
  lattice->waitOpInfo.reset();
  auto newFingerprint = getStateFingerprint(*lattice);
  propagateIfChanged(lattice, fingerprint == newFingerprint
                                  ? ChangeResult::NoChange
                                  : ChangeResult::Change);
}

#undef DUMP_STATE_HELPER
