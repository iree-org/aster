//===- ConvertToPIR.cpp -----------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/PIR/Transforms/Passes.h"

#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/PIR/IR/PIRDialect.h"
#include "aster/Dialect/PIR/IR/PIROps.h"
#include "aster/Dialect/PIR/Transforms/ToPIR.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/TypeSize.h"

namespace mlir {
namespace aster {
namespace pir {
#define GEN_PASS_DEF_CONVERTTOPIR
#include "aster/Dialect/PIR/Transforms/Passes.h.inc"
} // namespace pir
} // namespace aster
} // namespace mlir

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::pir;

namespace {
//===----------------------------------------------------------------------===//
// ConvertToPIR pass
//===----------------------------------------------------------------------===//
struct ConvertToPIR : public pir::impl::ConvertToPIRBase<ConvertToPIR> {
public:
  using Base::Base;
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// ConvertToPIRState
//===----------------------------------------------------------------------===//

struct ConvertToPIRState::Impl {
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
};

LogicalResult ConvertToPIRState::Impl ::initialize() {
  // Load the necessary analyses.
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<aster::dataflow::ThreadUniformAnalysis>();

  // Initialize and run the solver.
  if (failed(solver.initializeAndRun(rootOp)))
    return failure();

  dataLayout = &dataLayoutAnalysis.getAtOrAbove(rootOp);

  return success();
}

llvm::TypeSize ConvertToPIRState::Impl::getTypeSize(Type type) const {
  return dataLayout->getTypeSize(type);
}

ConvertToPIRState::~ConvertToPIRState() = default;

FailureOr<ConvertToPIRState> ConvertToPIRState::create(Operation *op) {
  if (!op)
    return failure();
  ConvertToPIRState state(op->getContext());
  state.impl = std::make_unique<Impl>(op);
  if (failed(state.impl->initialize()))
    return failure();
  return FailureOr<ConvertToPIRState>(std::move(state));
}

bool ConvertToPIRState::isThreadUniform(Value value) const {
  auto const *lattice =
      impl->solver.lookupState<dataflow::ThreadUniformLattice>(value);
  return lattice && lattice->getValue().isUniform();
}

int64_t ConvertToPIRState::getTypeSize(Type type) const {
  return impl->getTypeSize(type).getFixedValue();
}

int64_t ConvertToPIRState::getTypeSizeInBits(Type type) const {
  return impl->dataLayout->getTypeSizeInBits(type).getFixedValue();
}

//===----------------------------------------------------------------------===//
// ConvertToPIRConverter
//===----------------------------------------------------------------------===//

ToPIRConverter::ToPIRConverter(ConvertToPIRState &state) : state(&state) {
  indexType =
      state.getIntegerType(state.getTypeSizeInBits(state.getIndexType()));
  Type iTy = indexType;
  addConversion([&](Type type) { return type; });
  addConversion([iTy](IndexType type) { return iTy; });
  // Add generic source and target materializations.
  addSourceMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
  addTargetMaterialization([&](OpBuilder &builder, Type resultType,
                               ValueRange inputs, Location loc) {
    return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs)
        .getResult(0);
  });
}

//===----------------------------------------------------------------------===//
// ConvertToPIRPatternBase
//===----------------------------------------------------------------------===//

Value ToPIRPatternBase::createAlloca(RewriterBase &rewriter, Location loc,
                                     Type regTy) const {
  assert(isa<RegisterTypeInterface>(regTy) &&
         "Expected a register type for alloca");
  return AllocaOp::create(rewriter, loc, regTy);
}

//===----------------------------------------------------------------------===//
// ConvertToPIR pass
//===----------------------------------------------------------------------===//

void ConvertToPIR::runOnOperation() {
  Operation *op = getOperation();
  RewritePatternSet patterns(&getContext());
  FailureOr<ConvertToPIRState> state = ConvertToPIRState::create(op);
  if (failed(state))
    return signalPassFailure();
  ToPIRConverter converter(*state);
  ConversionTarget target(getContext());
  // Populate instruction selection patterns. TODO: Use a dialect interface.
  pir::populateToPIRPatterns(converter, patterns, target);
  ConversionConfig config;
  config.allowPatternRollback = false;
  if (failed(applyPartialConversion(
          op, target, FrozenRewritePatternSet(std::move(patterns)), config)))
    return signalPassFailure();
}
