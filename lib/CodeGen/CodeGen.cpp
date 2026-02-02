//===- CodeGen.cpp --------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/CodeGen/CodeGen.h"

#include "aster/Analysis/RegisterConstraints.h"
#include "aster/Analysis/ThreadUniformAnalysis.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/TypeSize.h"

using namespace mlir;
using namespace mlir::aster;

//===----------------------------------------------------------------------===//
// ConvertCodeGenState
//===----------------------------------------------------------------------===//

struct ConvertCodeGenState::Impl {
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

LogicalResult ConvertCodeGenState::Impl::initialize() {
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

llvm::TypeSize ConvertCodeGenState::Impl::getTypeSize(Type type) const {
  return dataLayout->getTypeSize(type);
}

Attribute ConvertCodeGenState::getRegisterConstraint(Value value) const {
  return impl->registerConstraints->getConstraint(value);
}

ConvertCodeGenState::~ConvertCodeGenState() = default;

FailureOr<ConvertCodeGenState> ConvertCodeGenState::create(Operation *op) {
  if (!op)
    return failure();
  ConvertCodeGenState state(op->getContext());
  state.impl = std::make_unique<Impl>(op);
  if (failed(state.impl->initialize()))
    return failure();
  return FailureOr<ConvertCodeGenState>(std::move(state));
}

bool ConvertCodeGenState::isThreadUniform(Value value) const {
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

int64_t ConvertCodeGenState::getTypeSize(Type type) const {
  return impl->getTypeSize(type).getFixedValue();
}

int64_t ConvertCodeGenState::getTypeSizeInBits(Type type) const {
  return impl->dataLayout->getTypeSizeInBits(type).getFixedValue();
}

//===----------------------------------------------------------------------===//
// ConvertCodeGenConverter
//===----------------------------------------------------------------------===//

CodeGenConverter::CodeGenConverter(ConvertCodeGenState &state) : state(&state) {
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
// ConvertCodeGenPatternBase
//===----------------------------------------------------------------------===//

Value CodeGenPatternBase::createAlloca(RewriterBase &rewriter, Location loc,
                                       Type regTy) const {
  assert(isa<RegisterTypeInterface>(regTy) &&
         "Expected a register type for alloca");
  return lsir::AllocaOp::create(rewriter, loc, regTy);
}
