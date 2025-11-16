//===- Init.h -------------------------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_API_API_H
#define ASTER_API_API_H

#include "aster/Interfaces/RegisterType.h"
#include "aster/Support/API.h"
#include "aster/Target/ASM/TranslateModule.h"
#include "mlir-c/IR.h"
#include <optional>
#include <vector>

namespace mlir::aster::amdgcn {
enum class RegisterKind : uint32_t;
/// Check if the given MLIR type is a register type.
ASTER_EXPORTED bool isRegisterType(MlirType type, RegisterKind kind,
                                   bool isRange);
/// Get the MLIR type for a specific register kind and optional register number.
ASTER_EXPORTED MlirType getRegisterType(MlirContext context, RegisterKind kind,
                                        Register reg);
/// Get the MLIR type for a specific register range.
ASTER_EXPORTED MlirType getRegisterRangeType(MlirContext context,
                                             RegisterKind kind,
                                             RegisterRange range);
/// Get the register range for a specific MLIR type.
ASTER_EXPORTED std::optional<RegisterRange> getRegisterRange(MlirType type);
/// Translate an MLIR module operation to AMDGPU assembly code.
ASTER_EXPORTED std::optional<std::string>
translateMlirModule(MlirOperation moduleOp, bool debugPrint = false);
/// Check if the AMDGPU target is available.
ASTER_EXPORTED bool hasAMDGPUTarget();
/// Compile AMDGPU assembly code to binary and link it.
ASTER_EXPORTED bool
compileAsm(MlirLocation loc, const std::string &asmCode,
           std::vector<char> &binary, std::string_view chip,
           std::string_view features,
           std::string_view triple = "amdgcn-amd-amdhsa",
           std::optional<std::string_view> path = std::nullopt,
           bool isLLDPath = false);
} // namespace mlir::aster::amdgcn

#endif // ASTER_API_API_H
