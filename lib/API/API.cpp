//===- API.cpp ------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aster/API/API.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Target/ASM/TranslateModule.h"
#include "aster/Target/Binary/CompileAsm.h"
#include "mlir/CAPI/IR.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;

namespace mlir::aster::amdgcn {
bool isRegisterType(MlirType type, RegisterKind kind, bool isRange) {
  Type mlirType = unwrap(type);
  auto regType = dyn_cast<AMDGCNRegisterTypeInterface>(mlirType);
  if (!regType)
    return false;
  if (getRegisterKind(regType) != kind)
    return false;
  return regType.isRegisterRange() == isRange;
}

MlirType getRegisterType(MlirContext context, RegisterKind kind, Register reg) {
  MLIRContext *ctx = unwrap(context);
  switch (kind) {
  case RegisterKind::AGPR:
    return wrap(AGPRType::get(ctx, reg));
  case RegisterKind::SGPR:
    return wrap(SGPRType::get(ctx, reg));
  case RegisterKind::VGPR:
    return wrap(VGPRType::get(ctx, reg));
  default:
    llvm_unreachable("nyi register kind");
  }
}

MlirType getRegisterRangeType(MlirContext context, RegisterKind kind,
                              RegisterRange range) {
  MLIRContext *ctx = unwrap(context);
  switch (kind) {
  case RegisterKind::AGPR:
    return wrap(AGPRType::get(ctx, range));
  case RegisterKind::SGPR:
    return wrap(SGPRType::get(ctx, range));
  case RegisterKind::VGPR:
    return wrap(VGPRType::get(ctx, range));
  default:
    llvm_unreachable("nyi register kind");
  }
}

std::optional<RegisterRange> getRegisterRange(MlirType type) {
  Type mlirType = unwrap(type);
  if (auto regType = dyn_cast<RegisterTypeInterface>(mlirType))
    return regType.getAsRange();

  return std::nullopt;
}

std::optional<std::string> translateMlirModule(MlirOperation moduleOp,
                                               bool debugPrint) {
  Operation *op = unwrap(moduleOp);
  auto module = dyn_cast<amdgcn::ModuleOp>(op);
  if (!module) {
    op->emitError("expected an amdgcn.module operation");
    return std::nullopt;
  }
  std::string result;
  llvm::raw_string_ostream os(result);
  if (failed(target::translateModule(module, os, debugPrint)))
    return std::nullopt;
  return result;
}

bool hasAMDGPUTarget() { return target::hasAMDGPUTarget(); }

bool compileAsm(MlirLocation loc, const std::string &asmCode,
                std::vector<char> &binary, std::string_view chip,
                std::string_view features, std::string_view triple,
                std::optional<std::string_view> path, bool isLLDPath) {
  Location location = unwrap(loc);
  llvm::SmallVector<char> tempBinary;

  // Call the target::compileAsm function
  if (failed(target::compileAsm(location, asmCode, tempBinary, chip, features,
                                triple))) {
    return false;
  }

  // Call the target::linkBinary function
  std::optional<StringRef> pathRef =
      path.has_value() ? StringRef(*path) : StringRef();
  if (failed(target::linkBinary(location, tempBinary, pathRef, isLLDPath)))
    return false;

  // Copy the result to the output vector
  binary.assign(tempBinary.begin(), tempBinary.end());
  return true;
}
} // namespace mlir::aster::amdgcn
