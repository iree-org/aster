//===- RegisterAlloc.cpp ------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Analysis/InterferenceAnalysis.h"
#include "aster/Analysis/RangeAnalysis.h"
#include "aster/Analysis/DPSAliasAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Interfaces/ResourceInterfaces.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/InterleavedRange.h"
#include "llvm/Support/LogicalResult.h"

#define DEBUG_TYPE "register-allocation"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_REGISTERALLOC
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
//===----------------------------------------------------------------------===//
// RegisterAlloc pass
//===----------------------------------------------------------------------===//
struct RegisterAlloc : public amdgcn::impl::RegisterAllocBase<RegisterAlloc> {
public:
  using Base::Base;
  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// Register manager
//===----------------------------------------------------------------------===//
/// Register resource manager.
struct RegisterManager {
  using Key = Resource *;
  RegisterManager(MLIRContext *context, int numSgpr, int numVgpr, int numAgpr);
  /// Allocate a range of registers.
  FailureOr<RegisterRange> alloc(Key kind, int numRegs, int alignment = 1);
  /// Release a range of registers.
  void release(Key kind, RegisterRange range);
  /// Take ownership of a range of registers.
  void takeRegisters(Key kind, ArrayRef<RegisterRange> ranges);

private:
  using RegSet = std::set<Register>;
  /// Allocate a range of registers.
  FailureOr<RegisterRange> alloc(RegSet &regs, int numRegs, int alignment);
  /// Release a range of registers.
  void release(RegSet &regs, RegisterRange range);
  /// Take ownership of a range of registers.
  void takeRegisters(RegSet &regs, ArrayRef<RegisterRange> ranges);
  DenseMap<Key, RegSet> regMap;
};

//===----------------------------------------------------------------------===//
// Register allocation driver
//===----------------------------------------------------------------------===//
struct RegAlloc {
  RegAlloc(Operation *topOp, InterferenceAnalysis &graph,
           RangeAnalysis &rangeAnalysis, int numSGPR, int numVGPR, int numAGPR)
      : topOp(topOp), graph(graph), rangeAnalysis(rangeAnalysis),
        registers(topOp->getContext(), numSGPR, numVGPR, numAGPR) {}
  /// Run the register allocation.
  LogicalResult run(RewriterBase &rewriter);

private:
  LogicalResult
  allocateVariable(EqClassID eqClassId, AllocaOp alloca, RegisterTypeInterface &cTy,
                   SmallVectorImpl<RegisterTypeInterface> &constraints);
  void collectConstraints(EqClassID eqClassId,
                          SmallVectorImpl<RegisterTypeInterface> &constraints);
  LogicalResult transform(RewriterBase &rewriter, MutableArrayRef<AllocaOp> aOp,
                          ArrayRef<EqClassID> eqClassIds);
  Operation *topOp;
  InterferenceAnalysis &graph;
  RangeAnalysis &rangeAnalysis;
  RegisterManager registers;
  /// The register coloring.
  SmallVector<RegisterTypeInterface> coloring;
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

struct SplitRegisterRangeOpPattern
    : public OpRewritePattern<SplitRegisterRangeOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(SplitRegisterRangeOp op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// RegisterManager
//===----------------------------------------------------------------------===//

RegisterManager::RegisterManager(MLIRContext *context, int numSgpr, int numVgpr,
                                 int numAgpr) {
  {
    RegSet &regs = regMap[SGPRResource::get()];
    for (int i = 0; i < numSgpr; ++i)
      regs.insert(Register(i));
  }
  {
    RegSet &regs = regMap[VGPRResource::get()];
    for (int i = 0; i < numVgpr; ++i)
      regs.insert(Register(i));
  }
  {
    RegSet &regs = regMap[AGPRResource::get()];
    for (int i = 0; i < numAgpr; ++i)
      regs.insert(Register(i));
  }
}

FailureOr<RegisterRange> RegisterManager::alloc(Key kind, int numRegs,
                                                int alignment) {
  return alloc(regMap[kind], numRegs, alignment);
}

void RegisterManager::release(Key kind, RegisterRange range) {
  release(regMap[kind], range);
}

void RegisterManager::takeRegisters(Key kind, ArrayRef<RegisterRange> ranges) {
  takeRegisters(regMap[kind], ranges);
}

FailureOr<RegisterRange> RegisterManager::alloc(RegSet &regs, int numRegs,
                                                int alignment) {
  if (numRegs <= 0 || alignment < 0)
    return failure();

  RegSet::iterator startIt = regs.begin();
  while (startIt != regs.end()) {
    int startReg = startIt->getRegister();

    // Check alignment and skip if we can't satisfy it
    if (alignment > 0 && startReg % alignment != 0) {
      ++startIt;
      continue;
    }

    RegSet::iterator endIt = startIt;
    ++endIt;
    int regCount = 1;
    while (regCount < numRegs && endIt != regs.end()) {
      if (endIt->getRegister() != startReg + regCount)
        break;
      ++regCount;
      ++endIt;
    }
    if (regCount == numRegs) {
      regs.erase(startIt, endIt);
      return RegisterRange(Register(startReg), numRegs, alignment);
    }
    startIt = endIt;
  }
  return failure();
}

void RegisterManager::release(RegSet &regs, RegisterRange range) {
  for (int i = 0; i < range.size(); ++i) {
    LDBG() << "Releasing register " << (range.begin().getRegister() + i);
    regs.insert(Register(range.begin().getRegister() + i));
  }
}

void RegisterManager::takeRegisters(RegSet &regs,
                                    ArrayRef<RegisterRange> ranges) {
  for (const RegisterRange &range : ranges) {
    for (int i = 0; i < range.size(); ++i) {
      LDBG() << "Taking register " << (range.begin().getRegister() + i);
      regs.erase(Register(range.begin().getRegister() + i));
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

LogicalResult
InstRewritePattern::matchAndRewrite(InstOpInterface op,
                                    PatternRewriter &rewriter) const {
  bool mutatedIns = false;
  bool mutatedOuts = false;
  // Helper to handle an operand.
  auto handleOperand = [&](Value operand) -> Value {
    auto cOp = dyn_cast_or_null<DeallocCastOp>(operand.getDefiningOp());
    if (!cOp) {
      return nullptr;
    }
    return cOp.getInput();
  };

  // Check if any operand or result needs to be updated.
  for (OpOperand &operand : op.getInstOutsMutable()) {
    Value nV = handleOperand(operand.get());
    mutatedOuts |= (nV != nullptr);
    if (nV)
      operand.set(nV);
  }
  for (OpOperand &operand : op.getInstInsMutable()) {
    Value nV = handleOperand(operand.get());
    mutatedIns |= (nV != nullptr);
    if (nV)
      operand.set(nV);
  }

  // Early exit if nothing changed.
  if (!mutatedIns && !mutatedOuts)
    return failure();

  // Create the new instruction.
  auto newInst = cast<InstOpInterface>(rewriter.clone(*op.getOperation()));

  if (!mutatedOuts) {
    rewriter.replaceOp(op, newInst->getResults());
    return success();
  }

  // Get the updated results.
  SmallVector<Value> newRes;
  ResultRange results = newInst->getResults();
  ResultRange instResults = newInst.getInstResults();
  ValueRange outs = newInst.getInstOuts();
  int64_t rPos = 0;
  int64_t rSz = results.size();
  int64_t oPos = 0;
  int64_t oSz = outs.size();
  while (rPos < rSz) {
    OpResult res = rPos >= rSz ? nullptr : results[rPos++];
    OpResult out = oPos >= oSz ? nullptr : instResults[oPos];

    // Add non-inst results as is.
    if (res != out) {
      newRes.push_back(res);
      continue;
    }

    // Handle inst results.
    Value outVal = outs[oPos++];

    // If the types match, add the result as is.
    if (out.getType() == outVal.getType()) {
      newRes.push_back(res);
      continue;
    }

    // Otherwise, create a cast to the expected type.
    out.setType(outVal.getType());
    newRes.push_back(DeallocCastOp::create(rewriter, out.getLoc(), out));
  }
  rewriter.replaceOp(op, newRes);
  return success();
}

LogicalResult
MakeRegisterRangeOpPattern::matchAndRewrite(MakeRegisterRangeOp op,
                                            PatternRewriter &rewriter) const {
  SmallVector<Value> ins;
  for (Value v : op.getInputs()) {
    auto cOp = dyn_cast_or_null<DeallocCastOp>(v.getDefiningOp());
    if (!cOp)
      return failure();
    ins.push_back(cOp.getInput());
  }
  auto newOp = MakeRegisterRangeOp::create(rewriter, op.getLoc(), ins);
  rewriter.replaceOpWithNewOp<DeallocCastOp>(op, op.getType(), newOp);
  return success();
}

LogicalResult
RegInterferenceOpPattern::matchAndRewrite(RegInterferenceOp op,
                                          PatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult
SplitRegisterRangeOpPattern::matchAndRewrite(SplitRegisterRangeOp op,
                                             PatternRewriter &rewriter) const {
  Value in;
  if (auto cOp =
          dyn_cast_or_null<DeallocCastOp>(op.getInput().getDefiningOp())) {
    in = cOp.getInput();
  } else {
    return failure();
  }
  auto newOp = SplitRegisterRangeOp::create(rewriter, op.getLoc(), in);
  SmallVector<Value> outs;
  for (Value v : op.getResults()) {
    outs.push_back(DeallocCastOp::create(rewriter, v.getLoc(),
                                         newOp.getResult(outs.size())));
  }
  rewriter.replaceOp(op, outs);
  return success();
}

//===----------------------------------------------------------------------===//
// RegAlloc
//===----------------------------------------------------------------------===//

static RegisterTypeInterface getRegisterType(RegisterTypeInterface base,
                                             Register reg) {
  return base.cloneRegisterType(reg);
}

void RegAlloc::collectConstraints(
    EqClassID eqClassId, SmallVectorImpl<RegisterTypeInterface> &constraints) {
  constraints.clear();
  for (Graph::Edge edge : graph.edges(eqClassId)) {
    RegisterTypeInterface rTy = coloring[edge.second];
    if (rTy == nullptr)
      continue;
    constraints.push_back(rTy);
    registers.takeRegisters(rTy.getResource(), rTy.getAsRange());
  }
  LDBG() << "Register constraints: " << llvm::interleaved_array(constraints);
}

LogicalResult RegAlloc::allocateVariable(
    EqClassID eqClassId, AllocaOp alloca, RegisterTypeInterface &cTy,
    SmallVectorImpl<RegisterTypeInterface> &constraints) {
  FailureOr<RegisterRange> allocatedRange;
  RegisterTypeInterface rTy = alloca.getType();
  Resource *regKind = rTy.getResource();
  LDBG() << "Allocating equivalence class " << eqClassId << ": " << alloca;

  // Collect neighbor coloring constraints.
  collectConstraints(eqClassId, constraints);

  // Deallocate constraints on scope exit.
  auto deallocConstraints = llvm::make_scope_exit([&]() {
    for (RegisterTypeInterface ty : constraints)
      registers.release(ty.getResource(), ty.getAsRange());
    if (llvm::succeeded(allocatedRange))
      registers.release(regKind, *allocatedRange);
  });

  const RangeAllocation *allocation = rangeAnalysis.lookupAllocation(eqClassId);

  // We only need to allocate a single register in this case.
  if (!allocation) {
    RegisterRange reqs = rTy.getAsRange();
    assert(reqs.size() == 1 && "expected a range with a single element");
    allocatedRange =
        registers.alloc(rTy.getResource(), reqs.size(), reqs.alignment());
    if (failed(allocatedRange))
      return alloca.emitError() << "failed to allocate register";
    cTy = getRegisterType(rTy, allocatedRange->begin());
    return success();
  }

  // The register belongs to a range, allocate the entire range.
  ArrayRef<EqClassID> eqClassIds = allocation->getAllocatedEqClassIds();
  allocatedRange = registers.alloc(rTy.getResource(), eqClassIds.size(),
                                   allocation->getAlignment());
  if (failed(allocatedRange))
    return alloca.emitError() << "failed to allocate register range";

  Register begin = allocatedRange->begin();
  for (auto [i, eqClassId] : llvm::enumerate(eqClassIds))
    coloring[eqClassId] = getRegisterType(rTy, Register(begin.getRegister() + i));
  return success();
}

LogicalResult RegAlloc::run(RewriterBase &rewriter) {
  // Get a deterministic order by sorting the allocas by dominance:
  SmallVector<AllocaOp> allocaOps;
  {
    SetVector<Operation *> vars, unsortedVars;
    unsortedVars.insert_range(
        llvm::map_range(graph->getValues(), [](Value v) {
          return cast<AllocaOp>(v.getDefiningOp());
        }));
    vars = mlir::topologicalSort(unsortedVars);
    allocaOps = llvm::map_to_vector(
        vars.getArrayRef(), [](Operation *op) { return cast<AllocaOp>(op); });
  }
  // Initialize the coloring.
  coloring.assign(allocaOps.size(), nullptr);

  // Get the equivalence class IDs.
  SmallVector<EqClassID> eqClassIds = llvm::map_to_vector(
      allocaOps, [&](AllocaOp op) { return graph.getEqClassIds(op).front(); });
  assert(eqClassIds.size() == static_cast<size_t>(graph.sizeNodes()) &&
         "ill-formed analysis");

  // Pre-color non-relocatable registers and register ranges.
  for (auto [allocaOp, eqClassId] : llvm::zip(allocaOps, eqClassIds)) {
    RegisterTypeInterface rTy = allocaOp.getType();
    if (rTy.isRelocatable())
      continue;
    coloring[eqClassId] = rTy;
  }

  SmallVector<RegisterTypeInterface> localConstraints;
  // Color the graph.
  for (auto &&[allocaOp, eqClassId] : llvm::zip(allocaOps, eqClassIds)) {
    RegisterTypeInterface &cTy = coloring[eqClassId];
    if (cTy != nullptr)
      continue;
    if (failed(allocateVariable(eqClassId, allocaOp, cTy, localConstraints)))
      return failure();
  }
  LDBG_OS([&](raw_ostream &os) {
    os << "Register coloring:\n";
    llvm::interleave(
        coloring, os, [&](RegisterTypeInterface ty) { os << "  " << ty; },
        "\n");
  });
  return transform(rewriter, allocaOps, eqClassIds);
}

LogicalResult RegAlloc::transform(RewriterBase &rewriter,
                                  MutableArrayRef<AllocaOp> aOp,
                                  ArrayRef<EqClassID> eqClassIds) {
  for (auto [alloca, eqClassId] : llvm::zip(aOp, eqClassIds)) {
    RegisterTypeInterface coloredType = coloring[eqClassId];
    // Insert the new alloca before the original one
    rewriter.setInsertionPoint(alloca);

    // Create new alloca with the colored register type
    AllocaOp newAlloca =
        AllocaOp::create(rewriter, alloca.getLoc(), coloredType);

    // Insert unrealized cast from new alloca to original type
    auto cast =
        DeallocCastOp::create(rewriter, alloca.getLoc(), newAlloca.getResult());

    // Replace the alloca with the new one
    rewriter.replaceOp(alloca, cast);
  }
  // Configure and run the greedy pattern rewriter
  RewritePatternSet patterns(rewriter.getContext());
  patterns.add<InstRewritePattern, MakeRegisterRangeOpPattern,
               RegInterferenceOpPattern, SplitRegisterRangeOpPattern>(
      rewriter.getContext());
  if (failed(applyPatternsGreedily(topOp, std::move(patterns))))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// RegisterAlloc pass
//===----------------------------------------------------------------------===//

void RegisterAlloc::runOnOperation() {
  Operation *op = getOperation();
  if (failed(runVerifiersOnOp<IsAllocatableOpAttr>(op)))
    return signalPassFailure();
  SymbolTableCollection symbolTable;
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));
  FailureOr<InterferenceAnalysis> graph =
      InterferenceAnalysis::create(op, solver, symbolTable);
  if (failed(graph)) {
    op->emitError() << "Failed to create interference graph";
    return signalPassFailure();
  }
  RangeAnalysis range = RangeAnalysis::create(op, graph->getAnalysis());
  if (!range.isSatisfiable()) {
    op->emitError() << "Range constraints are not satisfiable";
    return signalPassFailure();
  }
  IRRewriter rewriter(op->getContext());
  RegAlloc regAlloc(op, *graph, range, 100, 256, 256);
  if (failed(regAlloc.run(rewriter)))
    return signalPassFailure();
}
