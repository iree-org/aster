// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "water/Dialect/Wave/IR/WaveAttrs.h"
#include "water/Dialect/Wave/IR/WaveDialect.h"
#include "water/Dialect/Wave/IR/WaveInterfaces.h"
#include "water/Dialect/Wave/IR/WaveOps.h"
#include "water/Dialect/Wave/Transforms/Passes.h"
#include "water/Dialect/Wave/Transforms/Utils.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"

using namespace mlir;

#define DEBUG_TYPE "wave-infer-types"

namespace wave {
#define GEN_PASS_DEF_WATERWAVEINFERTYPESPASS
#include "water/Dialect/Wave/Transforms/Passes.h.inc"
} // namespace wave

namespace {

// Core lattice for type/shape inference of wave tensors. In addition to the
// bottom and top states, it can represent a concrete type which may be
// a fully specified tensor type (specific) or an underspecified type (any). The
// JOIN function is defined by the following table:
//
// JOIN         top       specific       any         bottom
// top          top       top            top         top
// specific     top       specific|top*  specific    specific
// any          top       specific       any         any
// bottom       top       specific       any         bottom
//   * if two specific shapes are equal, their JOIN is equal to them, otherwise
//     it is an inference conflict and the result of a JOIN is the top state.
class InferTypeLatticeStorage {
public:
  InferTypeLatticeStorage() : value(nullptr, kUninitializedState) {}
  InferTypeLatticeStorage(const InferTypeLatticeStorage &value) = default;
  InferTypeLatticeStorage(wave::WaveTensorType concreteValue)
      : value(concreteValue, kSpecificTypeState) {}

  InferTypeLatticeStorage &
  operator=(const InferTypeLatticeStorage &other) = default;

  bool operator==(const InferTypeLatticeStorage &other) const {
    return value == other.value;
  }

  bool operator!=(const InferTypeLatticeStorage &other) const {
    return !(*this == other);
  }

  bool isBottom() const { return value.getInt() == kUninitializedState; }
  bool isTop() const { return value.getInt() == kUndecidableState; }

  wave::WaveTensorType getConcreteValue() const {
    if (value.getInt() != kSpecificTypeState)
      return nullptr;
    return llvm::cast<wave::WaveTensorType>(value.getPointer());
  }

  static InferTypeLatticeStorage top() {
    InferTypeLatticeStorage result;
    result.value.setPointer(nullptr);
    result.value.setInt(kUndecidableState);
    return result;
  }

  static InferTypeLatticeStorage join(const InferTypeLatticeStorage &lhs,
                                      const InferTypeLatticeStorage &rhs) {
    if (lhs.value == rhs.value)
      return lhs;

    if (lhs.isTop() || rhs.isTop())
      return top();

    if (lhs.isBottom())
      return rhs;

    if (rhs.isBottom())
      return lhs;

    wave::WaveTensorType lhsType = lhs.getConcreteValue();
    wave::WaveTensorType rhsType = rhs.getConcreteValue();
    if (!lhsType.getFullySpecified())
      return rhs;
    if (!rhsType.getFullySpecified())
      return lhs;

    if (lhsType.getShape() == rhsType.getShape())
      return lhsType;

    return top();
  }

  static InferTypeLatticeStorage meet(const InferTypeLatticeStorage &lhs,
                                      const InferTypeLatticeStorage &rhs) {
    return join(lhs, rhs);
  }

  void unsafeSet(const InferTypeLatticeStorage &value) {
    this->value = value.value;
  }

  void print(llvm::raw_ostream &os) const {
    if (isBottom())
      os << "<bottom>";
    else if (isTop())
      os << "<top>";
    else
      os << getConcreteValue();
  }

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

private:
  llvm::PointerIntPair<Type, 2> value;

  const static unsigned kUninitializedState = 0;
  const static unsigned kSpecificTypeState = 1;
  const static unsigned kUndecidableState = 2;
};

class InferTypeLattice : public dataflow::Lattice<InferTypeLatticeStorage> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(InferTypeLattice);
  using Lattice::Lattice;
};

static std::optional<llvm::LogicalResult>
handleNonInterfaceOpInferType(Operation *op) {
  if (llvm::isa<wave::WaveInferTypeOpInterface>(op))
    return std::nullopt;

  if (!llvm::any_of(op->getOperandTypes(),
                    llvm::IsaPred<wave::WaveTensorType>) &&
      !llvm::any_of(op->getResultTypes(),
                    llvm::IsaPred<wave::WaveTensorType>)) {
    return success();
  }
  return op->emitError()
         << "cannot propagate types across an operation not implementing "
            "the wave infer type interface";
}

class InferTypeForwardAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<InferTypeLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    if (failed(AbstractSparseForwardDataFlowAnalysis::initialize(top)))
      return failure();

    top->walk([&](Operation *op) {
      if (auto iface = llvm::dyn_cast<wave::WaveInferTypeOpInterface>(op)) {
        initForResults(iface);
      } else if (auto iface = llvm::dyn_cast<FunctionOpInterface>(op)) {
        if (!iface.isDeclaration())
          initForBlockArguments(iface.getFunctionBody().front());
      } else if (auto iterate = llvm::dyn_cast<wave::IterateOp>(op)) {
        initForResults(op);
        initForBlockArguments(iterate.getBody().front());
      }
      return WalkResult::advance();
    });
    return success();
  }

  void setToEntryState(InferTypeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  LogicalResult
  visitOperation(Operation *op,
                 llvm::ArrayRef<const InferTypeLattice *> operands,
                 llvm::ArrayRef<InferTypeLattice *> results) override {
    std::optional<LogicalResult> res = handleNonInterfaceOpInferType(op);
    if (res)
      return *res;

    auto extractType = [](const InferTypeLattice *lattice) {
      return lattice->getValue().getConcreteValue();
    };
    llvm::SmallVector<wave::WaveTensorType> operandTypes =
        llvm::map_to_vector(operands, extractType);
    llvm::SmallVector<wave::WaveTensorType> resultTypes =
        llvm::map_to_vector(results, extractType);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<ChangeResult> result =
        llvm::cast<wave::WaveInferTypeOpInterface>(op).propagateForward(
            operandTypes, resultTypes, errs);
    if (failed(result)) {
      return op->emitError()
             << "failed to propagate type information forward: " << errs.str();
    }
    if (*result == ChangeResult::NoChange)
      return success();

    for (auto &&[result, lattice] : llvm::zip_equal(resultTypes, results)) {
      propagateIfChanged(lattice,
                         lattice->join(InferTypeLatticeStorage(result)));
    }
    return success();
  }

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      llvm::ArrayRef<InferTypeLattice *> lattices) /*override*/ {
    auto iterateOp = llvm::dyn_cast<wave::IterateOp>(op);
    if (!iterateOp)
      return;

    assert((successor.isParent() ||
            successor.getSuccessor()->getRegionNumber() == 0) &&
           "unexpected control flow");

    auto yieldOp =
        llvm::cast<wave::YieldOp>(iterateOp.getLoopBody()->getTerminator());
    if (successor.getSuccessor()) {
      for (auto &&[terminatorOperand, iterArg, lattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(),
               lattices.take_front(iterateOp.getIterArgs().size()))) {
        const InferTypeLattice *iterArgLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), iterArg);
        const InferTypeLattice *terminatorOperandLattice = getLatticeElementFor(
            getProgramPointBefore(iterateOp.getLoopBody()), terminatorOperand);
        ChangeResult changed = lattice->join(iterArgLattice->getValue());
        changed |= lattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(lattice, changed);
      }
    } else {
      for (auto &&[terminatorOperand, iterArg, resultLattice] : llvm::zip_equal(
               yieldOp.getOperands(), iterateOp.getIterArgs(), lattices)) {
        const InferTypeLattice *terminatorOperandLattice = getLatticeElementFor(
            getProgramPointAfter(iterateOp), terminatorOperand);
        const InferTypeLattice *iterArgLattice =
            getLatticeElementFor(getProgramPointAfter(iterateOp), iterArg);
        ChangeResult changed = resultLattice->join(iterArgLattice->getValue());
        changed |= resultLattice->join(terminatorOperandLattice->getValue());
        propagateIfChanged(resultLattice, changed);
      }
    }
  }

private:
  InferTypeLattice *initForValue(Value value) {
    auto tensorType = llvm::dyn_cast<wave::WaveTensorType>(value.getType());
    if (!tensorType)
      return nullptr;
    InferTypeLattice *lattice = getLatticeElement(value);
    lattice->getValue().unsafeSet(InferTypeLatticeStorage(tensorType));
    propagateIfChanged(lattice, ChangeResult::Change);
    return lattice;
  }

  void initForResults(Operation *op) {
    for (Value result : op->getResults())
      initForValue(result);
  }

  void initForBlockArguments(Block &block) {
    for (Value arg : block.getArguments())
      initForValue(arg);
  }
};

class InferTypeBackwardAnalysis
    : public dataflow::SparseBackwardDataFlowAnalysis<InferTypeLattice> {
public:
  using SparseBackwardDataFlowAnalysis::SparseBackwardDataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    if (getSolverConfig().isInterprocedural())
      return top->emitError() << "interprocedural analysis not supported";

    if (failed(SparseBackwardDataFlowAnalysis::initialize(top)))
      return failure();

    top->walk([this](Operation *op) {
      if (!op->hasTrait<OpTrait::ReturnLike>())
        return;
      if (!llvm::isa<FunctionOpInterface>(op->getParentOp()))
        return;
      for (Value operand : op->getOperands()) {
        auto tensorType =
            llvm::dyn_cast<wave::WaveTensorType>(operand.getType());
        if (!tensorType)
          continue;
        InferTypeLattice *lattice = getLatticeElement(operand);
        lattice->getValue().unsafeSet(InferTypeLatticeStorage(tensorType));
      }
    });
    return success();
  }

  void setToExitState(InferTypeLattice *lattice) override {
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  LogicalResult
  visitOperation(Operation *op, llvm::ArrayRef<InferTypeLattice *> operands,
                 llvm::ArrayRef<const InferTypeLattice *> results) override {
    std::optional<LogicalResult> res = handleNonInterfaceOpInferType(op);
    if (res)
      return *res;

    auto extractType = [](const InferTypeLattice *lattice) {
      return lattice->getValue().getConcreteValue();
    };
    llvm::SmallVector<wave::WaveTensorType> operandTypes =
        llvm::map_to_vector(operands, extractType);
    llvm::SmallVector<wave::WaveTensorType> resultTypes =
        llvm::map_to_vector(results, extractType);

    std::string errorMessage;
    llvm::raw_string_ostream errs(errorMessage);
    llvm::FailureOr<ChangeResult> result =
        llvm::cast<wave::WaveInferTypeOpInterface>(op).propagateBackward(
            operandTypes, resultTypes, errs);
    if (failed(result)) {
      return op->emitError()
             << "failed to propagate type information backward: " << errs.str();
    }
    if (*result == ChangeResult::NoChange)
      return success();

    for (auto &&[operand, lattice] : llvm::zip_equal(operandTypes, operands)) {
      propagateIfChanged(lattice,
                         lattice->join(InferTypeLatticeStorage(operand)));
    }
    return success();
  }

  void visitBranchOperand(OpOperand &opOperand) override {
    auto tensorType =
        llvm::dyn_cast<wave::WaveTensorType>(opOperand.get().getType());
    if (!tensorType)
      return;

    if (auto iterateOp =
            llvm::dyn_cast<wave::IterateOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      Value blockArgument = iterateOp.getLoopBody()->getArgument(position);
      const InferTypeLattice *blockArgLattice =
          getLatticeElement(blockArgument);
      InferTypeLattice *lattice = getLatticeElement(opOperand.get());
      addDependency(const_cast<InferTypeLattice *>(blockArgLattice),
                    getProgramPointAfter(iterateOp));
      propagateIfChanged(lattice, lattice->join(blockArgLattice->getValue()));
      return;
    }

    if (auto yieldOp = llvm::dyn_cast<wave::YieldOp>(opOperand.getOwner())) {
      unsigned position = opOperand.getOperandNumber();
      Value result = yieldOp->getParentOp()->getResult(position);
      const InferTypeLattice *resultLattice = getLatticeElement(result);
      InferTypeLattice *lattice = getLatticeElement(opOperand.get());
      addDependency(const_cast<InferTypeLattice *>(resultLattice),
                    getProgramPointAfter(yieldOp));
      propagateIfChanged(lattice, lattice->join(resultLattice->getValue()));
      return;
    }

    InferTypeLattice *lattice = getLatticeElement(opOperand.get());
    propagateIfChanged(lattice, lattice->join(InferTypeLatticeStorage::top()));
  }

  void visitCallOperand(OpOperand &opOperand) override {
    auto tensorType =
        llvm::dyn_cast<wave::WaveTensorType>(opOperand.get().getType());
    if (!tensorType)
      return;
    assert(tensorType.getFullySpecified() &&
           "expected fully-specified types at the call boundary");
    InferTypeLattice *lattice = getLatticeElement(opOperand.get());
    propagateIfChanged(lattice,
                       lattice->join(InferTypeLatticeStorage(tensorType)));
  }

  void
  visitNonControlFlowArguments(RegionSuccessor & /*successor*/,
                               ArrayRef<BlockArgument> /*arguments*/) override {
  }
};
} // namespace

// Run the dataflow analyses and capture whether some diagnostics were emitted.
// Only emit a generic diagnostic if no more specific diagnostic was emitted.
static llvm::LogicalResult
runSolverAndCaptureErrors(DataFlowSolver &solver, Operation *root, bool force) {
  bool emittedError = false;
  DiagnosticEngine::HandlerID handlerID =
      root->getContext()->getDiagEngine().registerHandler(
          [&](Diagnostic &diag) {
            if (diag.getSeverity() == DiagnosticSeverity::Error)
              emittedError = true;
            return failure();
          });
  if (failed(solver.initializeAndRun(root))) {
    if (!emittedError)
      root->emitError() << "dataflow analysis failed";
    if (!force)
      return llvm::failure();
  }
  root->getContext()->getDiagEngine().eraseHandler(handlerID);
  return llvm::success();
}

static llvm::LogicalResult
updateValueTypes(Operation *root,
                 llvm::function_ref<llvm::LogicalResult(Value, llvm::StringRef)>
                     updateType) {
  WalkResult walkResult = root->walk([&](Operation *op) {
    for (OpResult res : op->getResults()) {
      if (failed(updateType(res, "result #" +
                                     std::to_string(res.getResultNumber()))))
        return WalkResult::interrupt();
    }

    for (Region &region : op->getRegions()) {
      for (auto &&[blockNumber, block] : llvm::enumerate(region)) {
        for (BlockArgument arg : block.getArguments()) {
          auto fmt = llvm::formatv("argument #{0} of block #{1} in region #{2}",
                                   arg.getArgNumber(), blockNumber,
                                   region.getRegionNumber());
          if (failed(updateType(arg, fmt.str())))
            return WalkResult::interrupt();
        }
      }
    }

    return WalkResult::advance();
  });

  return llvm::failure(walkResult.wasInterrupted());
}

namespace {
class InferTypes : public wave::impl::WaterWaveInferTypesPassBase<InferTypes> {
public:
  using WaterWaveInferTypesPassBase::WaterWaveInferTypesPassBase;

  void runOnOperation() override {
    if (llvm::failed(verifyWaterNormalFormPassPrecondition(
            wave::WaveWaterNormalForm::FunctionBoundarySpecified,
            getOperation(), getArgument())))
      return signalPassFailure();

    SymbolTableCollection symbolTable;
    DataFlowConfig dataFlowConfig;
    dataFlowConfig.setInterprocedural(false);
    DataFlowSolver solver(dataFlowConfig);
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<InferTypeForwardAnalysis>();
    solver.load<InferTypeBackwardAnalysis>(symbolTable);
    Operation *root = getOperation();

    if (llvm::failed(runSolverAndCaptureErrors(solver, root, force)))
      return signalPassFailure();

    auto updateType = [&](Value value, llvm::StringRef description) {
      if (!llvm::isa<wave::WaveTensorType>(value.getType()))
        return success();

      auto *lattice = solver.lookupState<InferTypeLattice>(value);
      if (!lattice || lattice->getValue().isBottom()) {
        emitError(value.getLoc()) << "couldn't infer type for " << description;
        return failure(!force);
      }
      if (lattice->getValue().isTop()) {
        emitError(value.getLoc())
            << "type conflict was detected for " << description;
        return failure(!force);
      }

      value.setType(lattice->getValue().getConcreteValue());
      return success();
    };

    if (llvm::failed(updateValueTypes(getOperation(), updateType)))
      return signalPassFailure();

    WalkResult walkResult =
        getOperation()->walk([&](wave::WaveInferTypeOpInterface iface) {
          if (failed(iface.finalizeTypeInference()))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return signalPassFailure();

    if (!partial) {
      llvm::LogicalResult result = setWaterNormalFormPassPostcondition(
          wave::WaveWaterNormalForm::AllTypesSpecified, getOperation());
      if (llvm::failed(result) && !force)
        return signalPassFailure();
    }
  }
};
} // namespace
