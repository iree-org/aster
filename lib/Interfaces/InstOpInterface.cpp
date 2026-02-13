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
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"

using namespace mlir;
using namespace mlir::aster;

LogicalResult mlir::aster::detail::verifyInstImpl(InstOpInterface op) {
  ValueRange outOperands = op.getInstOuts();
  ValueRange results = op.getInstResults();
  int64_t resPos = 0, numRes = 0, numOuts = 0, resSize = results.size();
  for (Value value : outOperands) {
    auto regType = dyn_cast<RegisterTypeInterface>(value.getType());
    if (!regType) {
      return op.emitError()
             << "output operand has unexpected type: " << value.getType();
    }
    if (!regType.hasValueSemantics())
      continue;
    ++numOuts;
    if (resPos >= resSize)
      continue;
    Type result = results[resPos++].getType();
    if (result != value.getType()) {
      return op.emitError() << "expected result type to be " << value.getType()
                            << " but got " << result;
    }
    ++numRes;
  }
  if (numRes != resSize) {
    return op.emitError() << "expected " << resSize << " results but got "
                          << numRes;
  }
  if (numOuts != numRes) {
    return op.emitError() << "expected " << numOuts << " results but got "
                          << numRes;
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

template <typename IRElement>
void addEffectsForRegister(
    IRElement element, Type type, MemoryEffects::Effect *effect,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto regType = dyn_cast<RegisterTypeInterface>(type);
  if (!regType || regType.hasValueSemantics())
    return;
  // Add the effect for the resource.
  if (SideEffects::Resource *resource = regType.getResource())
    effects.emplace_back(effect, element, resource);
}

static llvm::MutableArrayRef<OpOperand> getAsOpOperands(OperandRange operands) {
  if (operands.empty())
    return {};
  return llvm::MutableArrayRef<OpOperand>(operands.getBase(), operands.size());
}

void mlir::aster::detail::getInstEffectsImpl(
    InstOpInterface op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // If the operation has pure value semantics, the op is Pure.
  if (op.hasPureValueSemantics())
    return;

  // Add write effects for outputs
  for (OpResult res : op.getInstResults())
    addEffectsForRegister(res, res.getType(), MemoryEffects::Write::get(),
                          effects);
  for (OpOperand &out : getAsOpOperands(op.getInstOuts()))
    addEffectsForRegister(&out, out.get().getType(),
                          MemoryEffects::Write::get(), effects);

  // Add read effects for inputs
  for (OpOperand &in : getAsOpOperands(op.getInstIns()))
    addEffectsForRegister(&in, in.get().getType(), MemoryEffects::Read::get(),
                          effects);
}

#include "aster/Interfaces/InstOpInterface.cpp.inc"
