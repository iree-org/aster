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
#include "aster/Analysis/VariableAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNEnums.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
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
  RegisterManager(MLIRContext *context, int numSgpr, int numVgpr, int numAgpr);
  /// Allocate a range of registers.
  FailureOr<RegisterRange> alloc(Attribute kind, int numRegs,
                                 int alignment = 1);
  /// Release a range of registers.
  void release(Attribute kind, RegisterRange range);
  /// Take ownership of a range of registers.
  void takeRegisters(Attribute kind, ArrayRef<RegisterRange> ranges);

private:
  using RegSet = std::set<Register>;
  /// Allocate a range of registers.
  FailureOr<RegisterRange> alloc(RegSet &regs, int numRegs, int alignment);
  /// Release a range of registers.
  void release(RegSet &regs, RegisterRange range);
  /// Take ownership of a range of registers.
  void takeRegisters(RegSet &regs, ArrayRef<RegisterRange> ranges);
  DenseMap<Attribute, RegSet> regMap;
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
  allocateVariable(VariableID vid, AllocaOp alloca, RegisterTypeInterface &cTy,
                   SmallVectorImpl<RegisterTypeInterface> &constraints);
  void collectConstraints(VariableID vid,
                          SmallVectorImpl<RegisterTypeInterface> &constraints);
  LogicalResult transform(RewriterBase &rewriter, MutableArrayRef<AllocaOp> aOp,
                          ArrayRef<VariableID> vids);
  Operation *topOp;
  InterferenceAnalysis &graph;
  RangeAnalysis &rangeAnalysis;
  RegisterManager registers;
  /// The register coloring, it's indexed by variable ID.
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
    RegSet &regs = regMap[RegisterKindAttr::get(context, RegisterKind::SGPR)];
    for (int i = 0; i < numSgpr; ++i)
      regs.insert(Register(i));
  }
  {
    RegSet &regs = regMap[RegisterKindAttr::get(context, RegisterKind::VGPR)];
    for (int i = 0; i < numVgpr; ++i)
      regs.insert(Register(i));
  }
  {
    RegSet &regs = regMap[RegisterKindAttr::get(context, RegisterKind::AGPR)];
    for (int i = 0; i < numAgpr; ++i)
      regs.insert(Register(i));
  }
}

FailureOr<RegisterRange> RegisterManager::alloc(Attribute kind, int numRegs,
                                                int alignment) {
  return alloc(regMap[kind], numRegs, alignment);
}

void RegisterManager::release(Attribute kind, RegisterRange range) {
  release(regMap[kind], range);
}

void RegisterManager::takeRegisters(Attribute kind,
                                    ArrayRef<RegisterRange> ranges) {
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
  SmallVector<Value> operands;
  bool mutatedIns = false;
  bool mutatedOuts = false;
  // Helper to handle an operand.
  auto handleOperand = [&](Value operand) {
    auto cOp = dyn_cast_or_null<DeallocCastOp>(operand.getDefiningOp());
    if (!cOp) {
      operands.push_back(operand);
      return false;
    }
    operands.push_back(cOp.getInput());
    return true;
  };

  // Check if any operand or result needs to be updated.
  for (OpOperand &operand : op.getInstOutsMutable())
    mutatedOuts |= handleOperand(operand.get());
  for (OpOperand &operand : op.getInstInsMutable())
    mutatedIns |= handleOperand(operand.get());

  // Early exit if nothing changed.
  if (!mutatedIns && !mutatedOuts)
    return failure();

  // Create the new instruction.
  auto newInst = cast<InstOpInterface>(rewriter.clone(*op.getOperation()));
  assert(operands.size() == op->getNumOperands() &&
         "ill-formed operand rewrite");
  newInst->setOperands(operands);
  if (!mutatedOuts) {
    rewriter.replaceOp(op, newInst->getResults());
    return success();
  }
  operands.clear();
  for (auto &&[res, out] :
       llvm::zip(newInst->getResults(), newInst.getInstOuts())) {
    if (res.getType() == out.getType()) {
      operands.push_back(res);
      continue;
    }
    res.setType(out.getType());
    operands.push_back(DeallocCastOp::create(rewriter, res.getLoc(), res));
  }
  rewriter.replaceOp(op, operands);
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
    VariableID vid, SmallVectorImpl<RegisterTypeInterface> &constraints) {
  constraints.clear();
  for (Graph::Edge edge : graph.edges(vid)) {
    RegisterTypeInterface rTy = coloring[edge.second];
    if (rTy == nullptr)
      continue;
    constraints.push_back(rTy);
    registers.takeRegisters(rTy.getRegisterKind(), rTy.getAsRange());
  }
  LDBG() << "Register constraints: " << llvm::interleaved_array(constraints);
}

LogicalResult RegAlloc::allocateVariable(
    VariableID vid, AllocaOp alloca, RegisterTypeInterface &cTy,
    SmallVectorImpl<RegisterTypeInterface> &constraints) {
  FailureOr<RegisterRange> allocatedRange;
  RegisterTypeInterface rTy = alloca.getType();
  Attribute regKind = rTy.getRegisterKind();
  LDBG() << "Allocating variable " << vid << ": " << alloca;

  // Collect neighbor coloring constraints.
  collectConstraints(vid, constraints);

  // Deallocate constraints on scope exit.
  auto deallocConstraints = llvm::make_scope_exit([&]() {
    for (RegisterTypeInterface ty : constraints)
      registers.release(ty.getRegisterKind(), ty.getAsRange());
    if (llvm::succeeded(allocatedRange))
      registers.release(regKind, *allocatedRange);
  });

  const RangeAllocation *allocation = rangeAnalysis.lookupAllocation(vid);

  // We only need to allocate a single register in this case.
  if (!allocation) {
    RegisterRange reqs = rTy.getAsRange();
    assert(reqs.size() == 1 && "expected a range with a single element");
    allocatedRange =
        registers.alloc(rTy.getRegisterKind(), reqs.size(), reqs.alignment());
    if (failed(allocatedRange))
      return alloca.emitError() << "failed to allocate register";
    cTy = getRegisterType(rTy, allocatedRange->begin());
    return success();
  }

  // The register belongs to a range, allocate the entire range.
  ArrayRef<VariableID> vars = allocation->getAllocatedVariables();
  allocatedRange = registers.alloc(rTy.getRegisterKind(), vars.size(),
                                   allocation->getAlignment());
  if (failed(allocatedRange))
    return alloca.emitError() << "failed to allocate register range";

  Register begin = allocatedRange->begin();
  for (auto [i, var] : llvm::enumerate(vars))
    coloring[var] = getRegisterType(rTy, Register(begin.getRegister() + i));
  return success();
}

LogicalResult RegAlloc::run(RewriterBase &rewriter) {
  // Get a deterministic order by sorting the allocas by dominance:
  SmallVector<AllocaOp> variables;
  {
    SetVector<Operation *> vars, unsortedVars;
    unsortedVars.insert_range(
        llvm::map_range(graph->getVariables(), [](Value v) {
          return cast<AllocaOp>(v.getDefiningOp());
        }));
    vars = mlir::topologicalSort(unsortedVars);
    variables = llvm::map_to_vector(
        vars.getArrayRef(), [](Operation *op) { return cast<AllocaOp>(op); });
  }
  // Initialize the coloring.
  coloring.assign(variables.size(), nullptr);

  // Get the variable IDs.
  SmallVector<VariableID> varIds = llvm::map_to_vector(
      variables, [&](AllocaOp op) { return graph.getVariableIds(op).front(); });
  assert(varIds.size() == static_cast<size_t>(graph.sizeNodes()) &&
         "ill-formed analysis");

  // Pre-color non-relocatable variables.
  for (auto [var, vid] : llvm::zip(variables, varIds)) {
    RegisterTypeInterface rTy = var.getType();
    if (rTy.isRelocatable())
      continue;
    coloring[vid] = rTy;
  }

  SmallVector<RegisterTypeInterface> localConstraints;
  // Color the graph.
  for (auto &&[vid, allocOp] : llvm::zip(varIds, variables)) {
    RegisterTypeInterface &cTy = coloring[vid];
    if (cTy != nullptr)
      continue;
    if (failed(allocateVariable(vid, allocOp, cTy, localConstraints)))
      return failure();
  }
  LDBG_OS([&](raw_ostream &os) {
    os << "Register coloring:\n";
    llvm::interleave(
        coloring, os, [&](RegisterTypeInterface ty) { os << "  " << ty; },
        "\n");
  });
  return transform(rewriter, variables, varIds);
}

LogicalResult RegAlloc::transform(RewriterBase &rewriter,
                                  MutableArrayRef<AllocaOp> aOp,
                                  ArrayRef<VariableID> vids) {
  for (auto [alloca, vid] : llvm::zip(aOp, vids)) {
    RegisterTypeInterface coloredType = coloring[vid];
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
               SplitRegisterRangeOpPattern>(rewriter.getContext());
  if (failed(applyPatternsGreedily(topOp, std::move(patterns))))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// RegisterAlloc pass
//===----------------------------------------------------------------------===//

void RegisterAlloc::runOnOperation() {
  Operation *op = getOperation();
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
