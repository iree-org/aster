
//===- CompileAsm.h - AMDGPU Assembly Compilation ---------------*- C++ -*-===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for compiling AMDGPU assembly to binary and
// linking binaries to create HSA code objects.
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_TARGET_BINARY_COMPILEASM_H
#define ASTER_TARGET_BINARY_COMPILEASM_H

#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include <optional>

namespace mlir::aster {
namespace amdgcn {
namespace target {
/// Check if the AMDGPU target is available in the current LLVM build.
bool hasAMDGPUTarget();

/// Compile AMDGPU assembly code to a binary object file.
///
/// This function takes assembly code and compiles it to a binary object file
/// using the LLVM MC infrastructure. The assembly is parsed and assembled into
/// an ELF object file format.
///
/// \param loc The source location for error reporting.
/// \param asm The assembly code to compile.
/// \param binary The output buffer to store the compiled binary object.
/// \param chip The target chip name (e.g., "gfx942").
/// \param features The target features string (e.g., "+sramecc,+xnack").
/// \param triple The target triple (e.g., "amdgcn-amd-amdhsa").
/// \return LogicalResult indicating success or failure.
LogicalResult compileAsm(Location loc, StringRef asmCode,
                         SmallVectorImpl<char> &binary, StringRef chip,
                         StringRef features,
                         StringRef triple = "amdgcn-amd-amdhsa");

/// Link a binary object file to create an HSA code object.
///
/// This function takes a binary object file and uses the LLD linker to create
/// an HSA code object (.hsaco file). The linker is invoked as a shared object
/// linker to produce the final executable code object.
///
/// \param loc The source location for error reporting.
/// \param binary The input/output buffer containing the object file and
///               receiving the linked HSA code object.
/// \param path Optional path to the ROCm toolkit installation or LLD tool.
/// \param isLLDPath If true, the provided path is treated as the LLD tool path.
/// \return LogicalResult indicating success or failure.
LogicalResult linkBinary(Location loc, SmallVectorImpl<char> &binary,
                         std::optional<StringRef> path = std::nullopt,
                         bool isLLDPath = false);
} // namespace target
} // namespace amdgcn
} // namespace mlir::aster

#endif // ASTER_TARGET_BINARY_COMPILEASM_H
