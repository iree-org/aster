//===- AMDGCNVerifiers.cpp - AMDGCN verifier implementations ----*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AMDGCN verifier functions.
//
//===----------------------------------------------------------------------===//

#include "aster/Dialect/AMDGCN/IR/AMDGCNVerifiers.h"

#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/Utils.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "aster/Interfaces/RegisterType.h"
#include "aster/Interfaces/VerifierAttr.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

//===----------------------------------------------------------------------===//
// verifyIsInKernelImpl
//===----------------------------------------------------------------------===//

LogicalResult amdgcn::verifyIsInKernelImpl(Operation *op,
                                           const VerifierState &state) {
  if (isa<KernelOp>(op))
    return success();
  // Walk up the parent chain to find a kernel operation.
  Operation *parent = op->getParentOfType<KernelOp>();
  if (parent)
    return success();
  return op->emitError("must be inside an 'amdgcn.kernel' operation");
}

//===----------------------------------------------------------------------===//
// verifyIsInModuleImpl
//===----------------------------------------------------------------------===//

LogicalResult amdgcn::verifyIsInModuleImpl(Operation *op,
                                           const VerifierState &state) {
  if (isa<amdgcn::ModuleOp>(op))
    return success();
  // Walk up the parent chain to find an AMDGCN module operation.
  Operation *parent = op->getParentOfType<amdgcn::ModuleOp>();
  if (parent)
    return success();
  return op->emitError("must be inside an 'amdgcn.module' operation");
}

//===----------------------------------------------------------------------===//
// verifyIsAllocatableOperationImpl
//===----------------------------------------------------------------------===//

/// Check if an operand is valid.
static LogicalResult checkOperand(Operation *op, Type type, int32_t pos,
                                  const VerifierState &state, bool isOut,
                                  bool allowUnallocated) {
  StringRef direction = isOut ? "output" : "input";
  if (type.isFloat() || type.isSignlessInteger(32)) {
    return success();
  }
  auto regTy = dyn_cast<RegisterTypeInterface>(type);
  if (!regTy || !isAMDReg(regTy)) {
    return (op->emitError(direction + " operand ")
            << pos << " has unexpected type: " << type)
               .attachNote(state.getLoc())
           << "is invalid";
  }
  if (!allowUnallocated && regTy.isRelocatable()) {
    return (op->emitError(direction + " operand ")
            << pos << " is unallocated register type: " << type)
               .attachNote(state.getLoc())
           << "is invalid";
  }
  return success();
}

/// Check if a value type is valid.
static LogicalResult checkValue(Operation *op, Type type, int32_t pos,
                                const VerifierState &state, bool isOut,
                                bool allowUnallocated) {
  StringRef direction = isOut ? "result" : "operand";
  auto regTy = dyn_cast<RegisterTypeInterface>(type);
  if (!regTy || !isAMDReg(regTy)) {
    return (op->emitError(direction) << pos << " has unexpected type: " << type)
               .attachNote(state.getLoc())
           << "is invalid";
  }
  if (!allowUnallocated && regTy.isRelocatable()) {
    return (op->emitError(direction)
            << pos << " is unallocated register type: " << type)
               .attachNote(state.getLoc())
           << "is invalid";
  }
  return success();
}

LogicalResult amdgcn::verifyIsAllocatableOpImpl(Operation *op,
                                                const VerifierState &state) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();
  if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op)) {
    for (auto [index, value] : llvm::enumerate(instOp.getInstOuts())) {
      if (failed(checkOperand(op, value.getType(), index, state,
                              /*isOut=*/true, /*allowUnallocated=*/true)))
        return failure();
    }
    for (auto [index, value] : llvm::enumerate(instOp.getInstIns())) {
      if (failed(checkOperand(op, value.getType(), index, state,
                              /*isOut=*/false, /*allowUnallocated=*/true)))
        return failure();
    }
    return success();
  }
  if (isa<KernelOp, aster::ModuleOpInterface, AllocaOp, MakeRegisterRangeOp,
          RegInterferenceOp, SplitRegisterRangeOp>(op))
    return success();
  if (state.getStrictness() != VerifierStrictness::Lax &&
      isa<ThreadIdOp, BlockIdOp, GridDimOp, BlockDimOp>(op)) {
    op->emitWarning("while valid, this operation should have been handled "
                    "before invoking the register allocator");
    return success();
  }
  return op->emitOpError() << "encountered unexpected operation";
}

//===----------------------------------------------------------------------===//
// verifyIsAllocatedOpImpl
//===----------------------------------------------------------------------===//

LogicalResult amdgcn::verifyIsAllocatedOpImpl(Operation *op,
                                              const VerifierState &state) {
  if (op->hasTrait<OpTrait::ConstantLike>())
    return success();
  if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op)) {
    for (auto [index, value] : llvm::enumerate(instOp.getInstOuts())) {
      if (failed(checkOperand(op, value.getType(), index, state,
                              /*isOut=*/true, /*allowUnallocated=*/false)))
        return failure();
    }
    for (auto [index, value] : llvm::enumerate(instOp.getInstIns())) {
      if (failed(checkOperand(op, value.getType(), index, state,
                              /*isOut=*/false, /*allowUnallocated=*/false)))
        return failure();
    }
    return success();
  }
  if (isa<KernelOp, aster::ModuleOpInterface>(op))
    return success();
  if (isa<AllocaOp, MakeRegisterRangeOp, SplitRegisterRangeOp>(op)) {
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (failed(checkValue(op, result.getType(), index, state,
                            /*isOut=*/true, /*allowUnallocated=*/false)))
        return failure();
    }
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (failed(checkValue(op, operand.getType(), index, state,
                            /*isOut=*/false, /*allowUnallocated=*/false)))
        return failure();
    }
    return success();
  }
  return op->emitOpError() << "encountered unexpected operation";
}

//===----------------------------------------------------------------------===//
// IsTranslatableOpAttr
//===----------------------------------------------------------------------===//

void IsTranslatableOpAttr::getDependentVerifiers(
    SmallVectorImpl<VerifierAttrInterface> &verifiers) const {
  verifiers.push_back(IsInModuleAttr::get(getContext()));
  verifiers.push_back(IsAllocatedOpAttr::get(getContext()));
  verifiers.push_back(VerifyConstantsAttr::get(getContext()));
}

//===----------------------------------------------------------------------===//
// verifyConstantsImpl
//===----------------------------------------------------------------------===//

static LogicalResult checkMaybeConstOperand(AMDGCNInstOpInterface op,
                                            Value value, int32_t pos,
                                            const VerifierState &state) {
  OperandKind kind = getOperandKind(value.getType());
  if (kind != OperandKind::IntImm && kind != OperandKind::FPImm) {
    return success();
  }
  if (value.getType().isSignlessInteger(64) || value.getType().isF64()) {
    return (op->emitError()
            << "constant operand " << pos
            << " has unsupported 64-bit type: " << value.getType())
               .attachNote(state.getLoc())
           << "is invalid";
  }
  if (isa<inst::VOP1Op, inst::SOP1Op, inst::SOP2Op, LoadOp, StoreOp,
          inst::SOPPOp>(op)) {
    return success();
  }
  // TODO: implement actual checks for other instructions.
  (void)(op->emitWarning() << "constant operand " << pos
                           << " may not be supported");
  return success();
}

LogicalResult amdgcn::verifyConstantsImpl(Operation *op,
                                          const VerifierState &state) {
  if (auto instOp = dyn_cast<AMDGCNInstOpInterface>(op)) {
    for (auto [index, value] : llvm::enumerate(instOp.getInstIns())) {
      // Always succeed add the moment. TODO: implement actual checks.
      if (failed(checkMaybeConstOperand(instOp, value, index, state))) {
        return failure();
      }
    }
    return success();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// verifyISAsSupportImpl
//===----------------------------------------------------------------------===//

LogicalResult
amdgcn::verifyISAsSupportImpl(Region &region, ArrayRef<ISAVersion> isas,
                              function_ref<InFlightDiagnostic()> emitError) {
  // If no ISAs specified, verify no isa-specific AMDGCN instructions.
  if (isas.empty()) {
    LogicalResult result = success();
    region.walk([&](Operation *op) {
      if (isa<AMDGCNInstOpInterface>(op)) {
        result =
            op->emitError("target-specific AMDGCN instruction not allowed in "
                          "target-agnostic context (no ISA specified)");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result;
  }

  // Validate that all AMDGCNInstOpInterface operations are valid for ALL
  // the listed ISA targets.
  LogicalResult result = success();
  region.walk([&](AMDGCNInstOpInterface instOp) {
    OpCode opcode = instOp.getOpcodeAttr().getValue();
    if (!isOpcodeValidForAllIsas(opcode, isas, region.getContext())) {
      result = instOp->emitError("instruction '")
               << stringifyOpCode(opcode)
               << "' is not valid for all specified ISA targets";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result;
}
