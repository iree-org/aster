//===- InstOpInterface.cpp - InstOp interface -------------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/Interfaces/InstOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::aster;

LogicalResult mlir::aster::detail::verifyInstImpl(InstOpInterface op) {
  if (!op.isDPSInstruction())
    return success();
  ValueRange outOperands = op.getInstOuts();
  ValueRange results = op.getInstResults();
  if (outOperands.size() != results.size()) {
    return op.emitOpError()
           << "number of output operands (" << outOperands.size()
           << ") does not match number of results (" << results.size() << ")";
  }
  if (TypeRange(op.getInstOuts()) != TypeRange(op.getInstResults())) {
    return op.emitOpError()
           << "types of output operands do not match types of results";
  }
  return success();
}

bool mlir::aster::detail::hasPureValueSemanticsImpl(InstOpInterface op) {
  /// Lambda to check if a type has value semantics.
  auto hasValueSemantics = +[](Type type) {
    auto regType = dyn_cast<RegisterTypeInterface>(type);
    return regType && regType.hasValueSemantics();
  };
  return llvm::all_of(TypeRange(op.getInstOuts()), hasValueSemantics) &&
         llvm::all_of(TypeRange(op.getInstIns()), hasValueSemantics);
}

Speculation::Speculatability
mlir::aster::detail::getInstSpeculatabilityImpl(InstOpInterface op) {
  // If the operation has pure value semantics, the op is Pure.
  if (op.hasPureValueSemantics())
    return Speculation::Speculatability::Speculatable;
  return Speculation::Speculatability::NotSpeculatable;
}

void mlir::aster::detail::getInstEffectsImpl(
    InstOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // If the operation has pure value semantics, the op is Pure.
  if (op.hasPureValueSemantics())
    return;

  // Helper to add effects for a register type with specific resources
  auto addEffectsForRegister = [&](Type type, MemoryEffects::Effect *effect) {
    auto regType = dyn_cast<RegisterTypeInterface>(type);

    // Skip if the type is not a register type or has value semantics.
    if (!regType || regType.hasValueSemantics())
      return;

    // Add the effect for the resource.
    if (SideEffects::Resource *resource = regType.getResource())
      effects.emplace_back(effect, resource);
  };

  // Add write effects for outputs
  for (OpResult res : op.getInstResults())
    addEffectsForRegister(res.getType(), MemoryEffects::Write::get());

  // Add read effects for inputs
  for (OpOperand &in : op.getInstInsMutable())
    addEffectsForRegister(in.get().getType(), MemoryEffects::Read::get());
}

#include "aster/Interfaces/InstOpInterface.cpp.inc"
