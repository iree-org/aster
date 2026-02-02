//===- CodeGen.h - LSIR CodeGen -------------------------------------------===//
//
// Copyright 2025 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ASTER_DIALECT_LSIR_CODEGEN_CODEGEN_H
#define ASTER_DIALECT_LSIR_CODEGEN_CODEGEN_H

namespace mlir {
class ConversionTarget;
class DialectRegistry;
class RewritePatternSet;
namespace aster {
struct CodeGenConverter;
namespace lsir {
/// Register the dialects required for AMDGCN code generation.
void getDependentCodeGenDialects(DialectRegistry &registry);

/// Populate the conversion patterns for LSIR code generation.
void populateCodeGenPatterns(CodeGenConverter &converter,
                             RewritePatternSet &patterns,
                             ConversionTarget &target);
} // namespace lsir
} // namespace aster
} // namespace mlir

#endif // ASTER_DIALECT_LSIR_CODEGEN_CODEGEN_H
