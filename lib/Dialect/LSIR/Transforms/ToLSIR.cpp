//===- ConvertToLSIR.cpp --------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/LSIR/Transforms/Passes.h"

#include "aster/Analysis/RegisterConstraints.h"
#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Dialect/LSIR/Transforms/ToLSIR.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/TypeSize.h"
#include <optional>

namespace mlir {
namespace aster {
namespace lsir {
#define GEN_PASS_DEF_CONVERTTOLSIR
#include "aster/Dialect/LSIR/Transforms/Passes.h.inc"
} // namespace lsir
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::lsir;

namespace {
//===----------------------------------------------------------------------===//
// ConvertToLSIR pass
//===----------------------------------------------------------------------===//
struct ConvertToLSIR : public lsir::impl::ConvertToLSIRBase<ConvertToLSIR> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToLSIRState
//===----------------------------------------------------------------------===//

struct ConvertToLSIRState::Impl {
  Impl(Operation *op) : rootOp(op), solver(), dataLayoutAnalysis(op) {}
  /// Initialize the analyses.
  LogicalResult initialize();
  /// Get the size of the given type according to the data layout.
  llvm::TypeSize getTypeSize(Type type) const;

  /// The root operation for instruction selection.
  Operation *rootOp;
  /// The data flow solver for analyses.
  DataFlowSolver solver;
  /// The data layout analysis.
  DataLayoutAnalysis dataLayoutAnalysis;
  const DataLayout *dataLayout;
  /// The register constraints analysis.
  std::optional<RegisterConstraints> registerConstraints;
};

LogicalResult ConvertToLSIRState::Impl ::initialize() {
  // Load the necessary analyses.
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<aster::dataflow::ThreadUniformAnalysis>();

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(rootOp)))
    return failure();

  dataLayout = &dataLayoutAnalysis.getAtOrAbove(rootOp);
  auto regConstraints = RegisterConstraints::create(rootOp);
  if (failed(regConstraints))
    return failure();
  registerConstraints = std::move(*regConstraints);
  return success();
}

llvm::TypeSize ConvertToLSIRState::Impl::getTypeSize(Type type) const {
  return dataLayout->getTypeSize(type);
}

Attribute ConvertToLSIRState::getRegisterConstraint(Value value) const {
  return impl->registerConstraints->getConstraint(value);
}

ConvertToLSIRState::~ConvertToLSIRState() = default;

FailureOr<ConvertToLSIRState> ConvertToLSIRState::create(Operation *op) {
  if (!op)
    return failure();
  ConvertToLSIRState state(op->getContext());
  state.impl = std::make_unique<Impl>(op);
  if (failed(state.impl->initialize()))
    return failure();
  return FailureOr<ConvertToLSIRState>(std::move(state));
}

bool ConvertToLSIRState::isThreadUniform(Value value) const {
  auto cOp =
      dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  while (cOp && cOp.getNumOperands() == 1) {
    value = cOp.getOperand(0);
    cOp =
        dyn_cast_if_present<UnrealizedConversionCastOp>(value.getDefiningOp());
  }
  auto const *lattice =
      impl->solver.lookupState<dataflow::ThreadUniformLattice>(value);
  return lattice && lattice->getValue().isUniform();
}

int64_t ConvertToLSIRState::getTypeSize(Type type) const {
  return impl->getTypeSize(type).getFixedValue();
}

int64_t ConvertToLSIRState::getTypeSizeInBits(Type type) const {
  return impl->dataLayout->getTypeSizeInBits(type).getFixedValue();
}

//===----------------------------------------------------------------------===//
// ConvertToLSIRConverter
//===----------------------------------------------------------------------===//

ToLSIRConverter::ToLSIRConverter(ConvertToLSIRState &state) : state(&state) {
  // Add generic source and target materializations.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (isa<RegisterTypeInterface>(resultType) && inputs.size() == 1 &&
        isa<RegisterTypeInterface>(inputs[0].getType())) {
      return lsir::RegCastOp::create(builder, loc, resultType, inputs[0])
          .getResult();
    }
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) -> Value {
    if (isa<RegisterTypeInterface>(resultType) && inputs.size() == 1 &&
        isa<RegisterTypeInterface>(inputs[0].getType())) {
      return lsir::RegCastOp::create(builder, loc, resultType, inputs[0])
          .getResult();
    }
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
}

//===----------------------------------------------------------------------===//
// ConvertToLSIRPatternBase
//===----------------------------------------------------------------------===//

Value ToLSIRPatternBase::createAlloca(RewriterBase &rewriter, Location loc,
                                      Type regTy) const {
  assert(isa<RegisterTypeInterface>(regTy) &&
         "Expected a register type for alloca");
  return AllocaOp::create(rewriter, loc, regTy);
}

//===----------------------------------------------------------------------===//
// ConvertToLSIR pass
//===----------------------------------------------------------------------===//

void ConvertToLSIR::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  FailureOr<ConvertToLSIRState> state = ConvertToLSIRState::create(op);
  if (failed(state))
    return signalPassFailure();
  ToLSIRConverter converter(*state);
  ConversionTarget target(getContext());
  lsir::populateToLSIRPatterns(converter, patterns, target);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          op, target, FrozenRewritePatternSet(std::move(patterns)), config)))
    return signalPassFailure();
  SmallVector<UnrealizedConversionCastOp> ops;
  getOperation()->walk(
      [&](UnrealizedConversionCastOp castOp) { ops.push_back(castOp); });
  reconcileUnrealizedCasts(ops);
}
