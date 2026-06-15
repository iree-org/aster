//===- MemoryDependenceAnalysis.cpp - Memory dependence analysis ---------===//
//
// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/MemoryDependenceAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Analysis/SliceWalk.h"
#include "mlir/Dialect/Transform/Utils/DiagnosedSilenceableFailure.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <utility>

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;
using namespace mlir::aster::lsir;

const SmallVector<MemDepEdge> MemoryDependenceAnalysis::emptyDeps = {};

/// Resolve the value that identifies the resource instance `op`'s access on
/// `resource` derives from, by tracing back def-use chains to RootOps and
/// through bbargs.
/// From the point of multi-buffer LDS, 3 main cases of interest are supported,
/// driven by 2 key considerations.
///
/// The key considerations are:
///   1. ds_write / ds_read memory patterns are subject to layout transfers and
///      swizzles that make single-op level analysis useless: the minimal
///      aliasing granularity is the entire buffer. To avoid over-conservative
///      analysis, buffers should be as small as possible to support a full
///      wave GR -> DS_W -> DS_R with good swizzle and layout transfers.
///   2. Unrolling by an amount that is not a multiple of the buffer count
///      results in rotating patterns that are handled conservatively. This is
///      acceptable because rotation without a matching unroll is a perf
///      non-starter due to register clobber and all its implications w/ static
///      register names.
///
/// The 3 main cases are:
///   1. Single buffer (with or without unrolling): every path reaches the one
///      alloc_lds and all accesses alias.
///   2. Multi-buffer + unrolling by a multiple of the buffer count: no yield
///      rotation pattern, alloc_lds has a precise 1-1 disambiguation.
///   3. Multi-buffer not unrolled by a multiple, iter_args ROTATE and tracing
///      one block arg reaches several distinct alloc_lds. We treat this
///      conservatively. This is acceptable by key consideration 2.
///
///      Example:
///        cf.br ^body(%off1, %off0 : i32, i32)
///      ^body(%cur: i32, %prev: i32):              // %cur, %prev both -> null
///        ds_write ... addr (to_reg %cur)          //   (each reaches off0+off1
///        ds_read  ... addr (to_reg %prev)         //    via the rotated yield)
///        cf.cond_br %c, ^body(%prev, %cur), ^exit // rotation
///
/// Return failure if zero or > 1 instances reach, treat it conservatively.
template <typename... RootOps>
static FailureOr<Value> resolveInstance(Operation *op) {
  Value data;
  if (auto store = dyn_cast<StoreOpInterface>(op))
    data = store.getDataOperand().getValue();

  SmallVector<Value> worklist;
  for (Value v : op->getOperands())
    if (v != data)
      worklist.push_back(v);

  DenseSet<Value> seen;
  Value root;
  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    if (!seen.insert(v).second)
      continue;
    Operation *def = v.getDefiningOp();
    if (def && (isa<RootOps>(def) || ...)) {
      if (root && root != v)
        return failure(); // several distinct instances reach -> unresolved
      root = v;
      continue; // do not trace past a resource root
    }
    if (def) {
      worklist.append(def->getOperands().begin(), def->getOperands().end());
      continue;
    }
    // Block arguments trace through the predecessor branch. Rotation patterns
    // will trigger aliases that we do not try to disambiguate for now.
    if (auto preds = getControlFlowPredecessors(v))
      worklist.append(preds->begin(), preds->end());
  }
  if (!root)
    return failure();
  return root;
}

namespace {
using AliasClassKey = std::pair<SideEffects::Resource *, Value>;
} // namespace

bool MemoryDependenceAnalysis::mayAlias(const AccessSite &a,
                                        const AccessSite &b) const {
  if (a.resource != b.resource)
    return false;
  if (auto unknown = unknownAliasClass.find(a.resource);
      unknown != unknownAliasClass.end() &&
      (a.aliasClass == unknown->second || b.aliasClass == unknown->second))
    return true;
  return a.aliasClass == b.aliasClass;
}

void MemoryDependenceAnalysis::buildAccessSites(Operation *root) {
  DenseMap<AliasClassKey, int64_t> resolvedAliasClass;

  root->walk([&](Operation *op) {
    auto iface = dyn_cast<MemoryEffectOpInterface>(op);
    if (!iface)
      return;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    iface.getEffects(effects);
    for (const auto &e : effects) {
      SideEffects::Resource *r = e.getResource();
      if (!llvm::is_contained(resources, r))
        continue;
      bool isWrite = isa<MemoryEffects::Write>(e.getEffect());
      if (!isWrite && !isa<MemoryEffects::Read>(e.getEffect()))
        continue;

      FailureOr<Value> instance =
          resolveInstance<AllocLDSOp, AssumeNoaliasOp>(op);
      int64_t aliasClass;
      if (succeeded(instance)) {
        aliasClass =
            resolvedAliasClass.try_emplace({r, *instance}, nextAliasClass++)
                .first->second;
      } else {
        aliasClass =
            unknownAliasClass.try_emplace(r, nextAliasClass++).first->second;
      }

      int64_t siteIdx = accessSites.size();
      accessSites.push_back({op, isWrite, r, aliasClass});
      opAccessIndices[op].push_back(siteIdx);
    }
  });
}

void MemoryDependenceAnalysis::buildDependences(Operation *root) {
  root->walk([&](Operation *op) {
    auto it = opAccessIndices.find(op);
    if (it == opAccessIndices.end())
      return;

    struct Reachability {
      const MemoryDependenceAnalysis &analysis;
      ArrayRef<AccessSite> accessSites;
      const DenseMap<Operation *, SmallVector<int64_t, 2>> &opAccessIndices;
      ArrayRef<int64_t> queryIndices;
      DenseSet<Block *> seenBlocks;
      DenseSet<std::pair<Operation *, int>> seenEdges;
      SmallVector<MemDepEdge> deps;

      void collect(Block *block, Operation *onlyBefore) {
        for (Operation &producer : *block) {
          if (&producer == onlyBefore)
            break;
          auto prodIt = opAccessIndices.find(&producer);
          if (prodIt == opAccessIndices.end())
            continue;
          for (int64_t pi : prodIt->second)
            for (int64_t qi : queryIndices) {
              const AccessSite &p = accessSites[pi];
              const AccessSite &a = accessSites[qi];
              if (!analysis.mayAlias(p, a) || (!p.isWrite && !a.isWrite))
                continue;
              DepKind kind = p.isWrite
                                 ? (a.isWrite ? DepKind::WAW : DepKind::RAW)
                                 : DepKind::WAR;
              if (seenEdges.insert({p.op, static_cast<int>(kind)}).second)
                deps.push_back({p.op, kind, a.resource});
            }
        }
      }

      void reachesEntryOf(Block *block) {
        for (Block *pred : block->getPredecessors()) {
          if (!seenBlocks.insert(pred).second)
            continue;
          collect(pred, /*onlyBefore=*/nullptr);
          reachesEntryOf(pred);
        }
      }
    };

    Reachability walk{*this, accessSites, opAccessIndices, it->second, {},
                      {},    {}};
    walk.collect(op->getBlock(), /*onlyBefore=*/op);
    walk.reachesEntryOf(op->getBlock());

    if (!walk.deps.empty())
      dependences[op] = std::move(walk.deps);
  });
}

MemoryDependenceAnalysis::MemoryDependenceAnalysis(
    ArrayRef<SideEffects::Resource *> resources)
    : resources(resources.begin(), resources.end()) {}

FailureOr<MemoryDependenceAnalysis>
MemoryDependenceAnalysis::create(Operation *root,
                                 ArrayRef<SideEffects::Resource *> resources) {
  // Ensure flat CFG normal form. Ideally this should be done by the caller.
  if (failed(NoScfOpsAttr::get(root->getContext())
                 .checkOperation(root)
                 .checkAndReport()))
    return failure();
  MemoryDependenceAnalysis analysis(resources);
  analysis.buildAccessSites(root);
  analysis.buildDependences(root);
  return analysis;
}

ArrayRef<MemDepEdge>
MemoryDependenceAnalysis::getDependences(Operation *op) const {
  auto it = dependences.find(op);
  if (it == dependences.end())
    return emptyDeps;
  return it->second;
}
