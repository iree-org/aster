//===- RegisterColoring.cpp ----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/Analysis/RangeConstraintAnalysis.h"
#include "aster/Dialect/AMDGCN/Analysis/RegisterInterferenceGraph.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/Transforms/Passes.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "amdgcn-register-coloring"

namespace mlir::aster {
namespace amdgcn {
#define GEN_PASS_DEF_REGISTERCOLORING
#include "aster/Dialect/AMDGCN/Transforms/Passes.h.inc"
} // namespace amdgcn
} // namespace mlir::aster

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {
static constexpr std::string_view kCastOpTag = "__amdgcn_register_coloring__";
//===----------------------------------------------------------------------===//
// RegisterColoring pass
//===----------------------------------------------------------------------===//
struct RegisterColoring
    : public amdgcn::impl::RegisterColoringBase<RegisterColoring> {
public:
  using Base::Base;
  void runOnOperation() override;

  /// Run the transformation on the given function.
  LogicalResult run(FunctionOpInterface funcOp);
};

//===----------------------------------------------------------------------===//
// RegisterAllocator
//===----------------------------------------------------------------------===//
/// A memory allocation.
struct Allocation {
  int16_t begin;
  int16_t size;
  RegisterKind kind;

  Allocation(int16_t begin, int16_t size, RegisterKind kind)
      : begin(begin), size(size), kind(kind) {}

  Allocation(AMDGCNRegisterTypeInterface regTy, int64_t numRegs)
      : begin(regTy.getAsRange().begin().getRegister()), size(numRegs),
        kind(regTy.getRegisterKind()) {}

  Register getBegin() const { return Register(begin); }
  Register getEnd() const { return Register(begin + size); }
  RegisterRange getRange() const { return RegisterRange(getBegin(), size, 1); }
  int16_t end() const { return begin + size; }

  bool operator<(const Allocation &other) const {
    return std::make_tuple(kind, begin) <
           std::make_tuple(other.kind, other.begin);
  }
};

/// The allocation constraints.
struct AllocConstraints {
  /// Insert a given allocation.
  void insert(Allocation alloc);

  /// Allocate memory for a given node, returns failure if no suitable region
  /// could be found.
  FailureOr<Allocation> alloc(AMDGCNRegisterTypeInterface regTy,
                              int16_t numRegs, int16_t alignment);

  /// Clear all allocations.
  void clear();

  /// Print the allocation constraints.
  void print(raw_ostream &os) const;

private:
  /// TODO: Get these from the target machine.
  /// The number of SGPRs.
  const int32_t numSGPR = 102;
  /// The number of VGPRs.
  const int32_t numVGPR = 256;
  /// The number of AGPRs.
  const int32_t numAGPR = 256;
  /// The allocation constraints.
  std::set<Allocation> constraints;
};

/// A greedy allocator for registers based on the interference graph.
/// The allocator traverses nodes in breadth-first order and assigns registers.
struct RegisterAllocator {
  using NodeId = RegisterInterferenceGraph::NodeID;
  RegisterAllocator(RegisterInterferenceGraph &graph, MLIRContext *ctx)
      : graph(graph), rewriter(ctx) {}

  /// Run the allocator on all nodes, returns failure if an allocation request
  /// cannot be satisfied.
  LogicalResult run(FunctionOpInterface op);

private:
  /// Collect the allocation constraints for the given node.
  LogicalResult collectConstraints(NodeId nodeId, ArrayRef<Value> nodes);
  /// Allocate memory for a register node.
  LogicalResult alloc(NodeId nodeId, Value alloca);

  RegisterInterferenceGraph &graph;
  AllocConstraints constraints;
  IRRewriter rewriter;
  /// Insertion point, updated as allocations are inserted.
  OpBuilder::InsertPoint ip;
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

struct CopyOpPattern : public OpRewritePattern<lsir::CopyOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(lsir::CopyOp op,
                                PatternRewriter &rewriter) const override;
};

struct VOP1OpPattern : public OpRewritePattern<inst::VOP1Op> {
  using Base::Base;

  LogicalResult matchAndRewrite(inst::VOP1Op op,
                                PatternRewriter &rewriter) const override;
};

struct SOP1OpPattern : public OpRewritePattern<inst::SOP1Op> {
  using Base::Base;

  LogicalResult matchAndRewrite(inst::SOP1Op op,
                                PatternRewriter &rewriter) const override;
};
} // namespace

//===----------------------------------------------------------------------===//
// AllocConstraints
//===----------------------------------------------------------------------===//

void AllocConstraints::insert(Allocation alloc) { constraints.insert(alloc); }

FailureOr<Allocation> AllocConstraints::alloc(AMDGCNRegisterTypeInterface regTy,
                                              int16_t numRegs,
                                              int16_t alignment) {
  LDBG() << "  Allocating " << numRegs << " registers of kind "
         << regTy.getRegisterKind() << " with alignment " << alignment;
  int16_t start = 0;
  auto getStartAligned = [&](int64_t addr) {
    return ((addr + alignment - 1) / alignment) * alignment;
  };

  int16_t maxRegs = 0;
  switch (regTy.getRegisterKind()) {
  case RegisterKind::SGPR:
    maxRegs = numSGPR;
    break;
  case RegisterKind::VGPR:
    maxRegs = numVGPR;
    break;
  case RegisterKind::AGPR:
    maxRegs = numAGPR;
    break;
  default:
    maxRegs = 1;
  }

  auto lb = constraints.lower_bound({0, 1, regTy.getRegisterKind()});
  auto ub = constraints.upper_bound({maxRegs, 1, regTy.getRegisterKind()});

  for (const Allocation &alloc : llvm::make_range(lb, ub)) {
    // Check if we can fit before this allocation.
    if (start + numRegs <= alloc.begin) {
      Allocation result = {start, numRegs, alloc.kind};
      constraints.insert(result);
      return result;
    }
    start = getStartAligned(alloc.end());
  }

  // Check if we can fit at the end.
  if (start + numRegs <= maxRegs) {
    Allocation result = {start, numRegs, regTy.getRegisterKind()};
    constraints.insert(result);
    return result;
  }

  return failure();
}

void AllocConstraints::clear() { constraints.clear(); }

void AllocConstraints::print(raw_ostream &os) const {
  os << "{";
  llvm::interleaveComma(constraints, os, [&](const Allocation &alloc) {
    os << alloc.getRange() << " : " << stringifyRegisterKind(alloc.kind);
  });
  os << "}";
}

//===----------------------------------------------------------------------===//
// RegisterAllocator
//===----------------------------------------------------------------------===//

LogicalResult RegisterAllocator::collectConstraints(NodeId nodeId,
                                                    ArrayRef<Value> nodes) {
  LDBG() << " Collecting constraints for node[" << nodeId
         << "]: " << nodes[nodeId];
  for (auto [src, tgt] : graph.edges(nodeId)) {
    LDBG() << "  Inspecting neighbor[" << tgt << "]: " << nodes[tgt];
    Value tgtNode = nodes[tgt];
    AMDGCNRegisterTypeInterface regTy =
        dyn_cast<AMDGCNRegisterTypeInterface>(tgtNode.getType());
    // Skip if the node is not a register type.
    if (!regTy)
      continue;

    // Error if the node is a value register.
    if (regTy.hasValueSemantics())
      return emitError(tgtNode.getLoc()) << "found unexpected value register";

    // Skip if the node is not an allocated register.
    if (!regTy.hasAllocatedSemantics())
      continue;

    constraints.insert(Allocation(regTy, 1));
  }
  return success();
}

LogicalResult RegisterAllocator::alloc(NodeId nodeId, Value alloca) {
  auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(alloca.getType());
  int64_t numRegs = 1;
  int64_t alignment = 1;
  ValueRange allocas = alloca;
  // If the alloca has a range constraint, use the constraint to allocate the
  // register.
  auto [rangeId, constraint] = graph.getRangeInfo(nodeId);
  if (constraint) {
    numRegs = constraint->allocations.size();
    alignment = constraint->alignment;
    allocas = constraint->allocations;
  }

  OpBuilder::InsertionGuard guard(rewriter);

  // Try to allocate the registers.
  FailureOr<Allocation> alloc = constraints.alloc(regTy, numRegs, alignment);
  if (failed(alloc))
    return emitError(alloca.getLoc()) << "failed to allocate the registers";

  // Replace the alloca with the new alloca.
  for (auto [i, alloca] : llvm::enumerate(allocas)) {
    // Get the alloca and set the insertion point.
    auto allocaOp = cast<AllocaOp>(alloca.getDefiningOp());

    // Set the insertion point to the last saved position.
    assert(ip.isSet() && "insertion point is not set");
    rewriter.restoreInsertionPoint(ip);

    // Create the new alloca.
    Register reg = alloc->getBegin().getWithOffset(i);
    AllocaOp newAlloca = AllocaOp::create(rewriter, allocaOp.getLoc(),
                                          regTy.cloneRegisterType(reg));

    // Cast back to the original type.
    auto cOp = UnrealizedConversionCastOp::create(rewriter, allocaOp.getLoc(),
                                                  regTy, newAlloca.getResult());
    cOp->setAttr(kCastOpTag, rewriter.getUnitAttr());

    // Update the insertion point to the new alloca.
    // NOTE: This checkpoint has to happen after the creation of the cast
    // operation so that the iteration is never invalid.
    rewriter.setInsertionPointAfter(newAlloca.getOperation());
    ip = rewriter.saveInsertionPoint();

    // Update the graph and the IR.
    rewriter.replaceOp(allocaOp, cOp.getResult(0));
    NodeId nodeId = graph.getNodeId(alloca);
    assert(nodeId >= 0 && "node not found in graph");
    graph.getValues()[nodeId] = newAlloca.getResult();
  }
  return success();
}

static bool needsAllocation(Value node) {
  auto regTy = dyn_cast<AMDGCNRegisterTypeInterface>(node.getType());
  return regTy && !regTy.hasAllocatedSemantics();
}

LogicalResult RegisterAllocator::run(FunctionOpInterface op) {
  Region &body = op.getFunctionBody();
  if (body.empty())
    return success();

  // Set the insertion point to the start of the entry block. This is used to
  // insert the allocas at the correct position and in the order they are
  // allocated.
  Block *entryBlock = &body.front();
  rewriter.setInsertionPointToStart(entryBlock);
  ip = rewriter.saveInsertionPoint();

  llvm::DenseSet<NodeId> visited;
  ArrayRef<Value> nodes = graph.getValues();
  for (auto [i, node] : llvm::enumerate(nodes)) {
    // Skip already visited or allocated nodes.
    if (!visited.insert(i).second || !needsAllocation(node))
      continue;

    LDBG() << "Allocating node[" << i << "]: " << node;

    // Collect the neighbors constraints.
    constraints.clear();
    if (failed(collectConstraints(i, nodes)))
      return failure();

    LDBG_OS([&](raw_ostream &os) {
      os << " Initial constraints: ";
      constraints.print(os);
    });

    // Allocate the node.
    if (failed(alloc(i, node)))
      return failure();

    LDBG_OS([&](raw_ostream &os) {
      os << " Final constraints: ";
      constraints.print(os);
    });
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Rewrite patterns
//===----------------------------------------------------------------------===//

LogicalResult
InstRewritePattern::matchAndRewrite(InstOpInterface op,
                                    PatternRewriter &rewriter) const {
  bool mutatedIns = false;
  bool mutatedOuts = false;

  // Bail out if the instruction has results.
  if (!op.getInstResults().empty()) {
    return rewriter.notifyMatchFailure(
        op, "expected instruction with register semantics");
  }

  // Helper to handle an operand.
  auto handleOperand = [&](Value operand) -> Value {
    auto cOp =
        dyn_cast_or_null<UnrealizedConversionCastOp>(operand.getDefiningOp());
    if (!cOp || !cOp->getDiscardableAttr(kCastOpTag))
      return nullptr;
    return cOp.getInputs().front();
  };

  // Check if any operand or result needs to be updated.
  SmallVector<Value> newIns = llvm::to_vector(op.getInstIns());
  SmallVector<Value> newOuts = llvm::to_vector(op.getInstOuts());
  for (Value &operand : newOuts) {
    Value nV = handleOperand(operand);
    mutatedOuts |= (nV != nullptr);
    if (nV)
      operand = nV;
  }
  for (Value &operand : newIns) {
    Value nV = handleOperand(operand);
    mutatedIns |= (nV != nullptr);
    if (nV)
      operand = nV;
  }

  // Early exit if nothing changed.
  if (!mutatedIns && !mutatedOuts)
    return failure();

  // Create the new instruction.
  auto newInst = op.cloneInst(rewriter, newOuts, newIns);
  if (!newInst)
    return failure();

  // Replace the original instruction with the new results.
  rewriter.replaceOp(op, newInst->getResults());
  return success();
}

LogicalResult
MakeRegisterRangeOpPattern::matchAndRewrite(MakeRegisterRangeOp op,
                                            PatternRewriter &rewriter) const {
  SmallVector<Value> ins;
  for (Value v : op.getInputs()) {
    auto cOp = dyn_cast_or_null<UnrealizedConversionCastOp>(v.getDefiningOp());
    if (!cOp || !cOp->getDiscardableAttr(kCastOpTag))
      return failure();
    ins.push_back(cOp.getInputs().front());
  }
  auto newOp = MakeRegisterRangeOp::create(rewriter, op.getLoc(), ins);
  auto cOp = rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
      op, op.getType(), newOp.getResult());
  cOp->setDiscardableAttr(kCastOpTag, rewriter.getUnitAttr());
  return success();
}

LogicalResult
RegInterferenceOpPattern::matchAndRewrite(RegInterferenceOp op,
                                          PatternRewriter &rewriter) const {
  rewriter.eraseOp(op);
  return success();
}

LogicalResult CopyOpPattern::matchAndRewrite(lsir::CopyOp op,
                                             PatternRewriter &rewriter) const {
  // Try to canonicalize the operation.
  if (succeeded(op.canonicalize(op, rewriter)))
    return success();

  // If the result is used, bail out.
  if (op.getTargetRes() && !op.getTargetRes().use_empty())
    return failure();

  // Get the source allocas. Bail out if the allocas are missing or need
  // allocation.
  FailureOr<ValueRange> srcAlloc = getAllocasOrFailure(op.getSource());
  if (failed(srcAlloc) ||
      llvm::any_of(*srcAlloc, [](Value v) { return needsAllocation(v); }))
    return failure();

  // Get the target allocas. Bail out if the allocas are missing or need
  // allocation.
  FailureOr<ValueRange> tgtAlloc = getAllocasOrFailure(op.getTarget());
  if (failed(tgtAlloc) ||
      llvm::any_of(*tgtAlloc, [](Value v) { return needsAllocation(v); }))
    return failure();

  assert(srcAlloc->size() == tgtAlloc->size() &&
         "source and target allocas must have the same size");

  assert(srcAlloc->size() > 0 &&
         "source and target allocas must have at least one alloca");

  auto srcTy = dyn_cast<AMDGCNRegisterTypeInterface>(op.getSource().getType());
  auto tgtTy = dyn_cast<AMDGCNRegisterTypeInterface>(op.getTarget().getType());

  // Bail if the source or target is not an AMDGCN register type.
  if (!srcTy || !tgtTy)
    return failure();

  // Bail if the copy cannot be performed.
  if (srcTy.getRegisterKind() != RegisterKind::SGPR &&
      tgtTy.getRegisterKind() == RegisterKind::SGPR) {
    return rewriter.notifyMatchFailure(
        op, "cannot copy between non-sgpr type to an sgpr type");
  }
  if (!llvm::is_contained({RegisterKind::SGPR, RegisterKind::VGPR},
                          srcTy.getRegisterKind()) ||
      !llvm::is_contained({RegisterKind::SGPR, RegisterKind::VGPR},
                          tgtTy.getRegisterKind())) {
    return rewriter.notifyMatchFailure(
        op, "cannot copy if data type is not SGPR or VGPR");
  }

  auto copyReg = [&](Value src, Value tgt) {
    if (tgtTy.getRegisterKind() == RegisterKind::SGPR) {
      S_MOV_B32::create(rewriter, tgt.getLoc(), tgt, src);
      return;
    }
    V_MOV_B32_E32::create(rewriter, tgt.getLoc(), tgt, src);
  };

  // Create the copy operations.
  for (auto [src, tgt] : llvm::zip_equal(*srcAlloc, *tgtAlloc))
    copyReg(src, tgt);
  rewriter.eraseOp(op);
  return success();
}

LogicalResult VOP1OpPattern::matchAndRewrite(inst::VOP1Op op,
                                             PatternRewriter &rewriter) const {
  auto opCode = op.getOpcode();
  if (opCode != OpCode::V_MOV_B32_E32)
    return failure();
  RegisterTypeInterface dstTy = op.getVdst().getType();
  auto srcTy = llvm::dyn_cast<RegisterTypeInterface>(op.getSrc0().getType());
  if (!srcTy)
    return failure();
  if (!srcTy.hasAllocatedSemantics() || !dstTy.hasAllocatedSemantics())
    return failure();
  if (srcTy != dstTy)
    return rewriter.notifyMatchFailure(
        op, "source and destination types do not match");
  rewriter.eraseOp(op);
  return success();
}

LogicalResult SOP1OpPattern::matchAndRewrite(inst::SOP1Op op,
                                             PatternRewriter &rewriter) const {
  auto opCode = op.getOpcode();
  if (opCode != OpCode::S_MOV_B32)
    return failure();
  RegisterTypeInterface dstTy = op.getSdst().getType();
  auto srcTy = llvm::dyn_cast<RegisterTypeInterface>(op.getSrc0().getType());
  if (!srcTy)
    return failure();
  if (!srcTy.hasAllocatedSemantics() || !dstTy.hasAllocatedSemantics())
    return failure();
  if (srcTy != dstTy)
    return rewriter.notifyMatchFailure(
        op, "source and destination types do not match");
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// RegisterColoring pass
//===----------------------------------------------------------------------===//

LogicalResult RegisterColoring::run(FunctionOpInterface funcOp) {
  // Parse build mode option.
  RegisterInterferenceGraph::BuildMode buildMode;
  if (this->buildMode == "full") {
    buildMode = RegisterInterferenceGraph::BuildMode::Full;
  } else if (this->buildMode == "minimal") {
    buildMode = RegisterInterferenceGraph::BuildMode::Minimal;
  } else {
    return funcOp.emitError()
           << "build-mode must be \"full\" or \"minimal\", got \""
           << this->buildMode << "\"";
  }

  // Create the range constraint analysis.
  FailureOr<RangeConstraintAnalysis> rangeAnalysis =
      RangeConstraintAnalysis::create(funcOp);
  if (failed(rangeAnalysis))
    return funcOp.emitError() << "failed to run range constraint analysis";

  // Create the dataflow solver and load the liveness analysis.
  SymbolTableCollection symbolTable;
  DataFlowSolver solver(DataFlowConfig().setInterprocedural(false));

  // Create the interference graph.
  FailureOr<RegisterInterferenceGraph> graph =
      RegisterInterferenceGraph::create(funcOp, solver, symbolTable,
                                        *rangeAnalysis, buildMode);
  if (failed(graph)) {
    return funcOp.emitError() << "failed to create register interference graph";
  }

  // Create and run the register allocator.
  RegisterAllocator allocator(*graph, funcOp->getContext());
  if (failed(allocator.run(funcOp)))
    return funcOp.emitError() << "failed to run register allocator";

  RewritePatternSet patterns(&getContext());
  patterns.add<InstRewritePattern, MakeRegisterRangeOpPattern,
               RegInterferenceOpPattern, CopyOpPattern, SOP1OpPattern,
               VOP1OpPattern>(&getContext());
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  if (failed(applyPatternsGreedily(
          funcOp, frozenPatterns,
          GreedyRewriteConfig().setRegionSimplificationLevel(
              GreedySimplifyRegionLevel::Disabled)))) {
    return funcOp.emitError() << "failed to apply patterns";
  }
  return success();
}

void RegisterColoring::runOnOperation() {
  Operation *op = getOperation();
  WalkResult walkResult =
      op->walk<WalkOrder::PreOrder>([&](FunctionOpInterface funcOp) {
        if (failed(run(funcOp)))
          return WalkResult::interrupt();
        return WalkResult::skip();
      });
  if (walkResult.wasInterrupted())
    return signalPassFailure();
}
